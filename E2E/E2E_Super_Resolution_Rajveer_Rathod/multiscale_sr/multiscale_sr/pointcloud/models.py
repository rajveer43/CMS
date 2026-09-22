"""Set-to-set generator and a permutation-invariant discriminator.

Stage 3, as scoped in the implementation prompt: the generator receives LR
HITS (not an image) and emits HR hits, so no dense grid appears anywhere
inside either model — only at the ``pointcloud/collate.py`` boundary before
this module and the re-rasterization boundary after it (see
``train_pointcloud.py``).

Architecture choice: a Deep-Sets-style encoder (permutation-invariant by
construction: shared per-point MLP + a symmetric pool) rather than full
PointNet++ with farthest-point sampling / ball-query neighborhoods. This is a
deliberate scope cut, stated rather than hidden: FPS-based hierarchical
grouping is the more expressive choice for point clouds with real 3D
neighborhood structure, but a calorimeter hit set here is inherently 2D
(pixel grid coordinates) and small (~700 hits at HR, per the audit) — a
global permutation-invariant summary plus a cross-attention decoder is a
reasonable first architecture to test the *representation* before spending
the added complexity of hierarchical grouping on a set this small.

Cardinality: the audit found the generator must roughly double hit count
(LR64->HR is 45% of HR's hits) and can need to more than quintuple it (LR16).
Fixed top-K truncation is therefore wrong for the *output* side. Instead the
generator emits a fixed BUDGET of `hr_n_max` candidate hits, each with a
learned existence logit; at inference, hits above an existence threshold (or
the top-N by existence) are kept. Padding rows are supervised toward
"non-existent" exactly like the input mask marks them, so the model learns
cardinality rather than having it imposed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PermInvariantEncoder(nn.Module):
    """Deep-Sets-style encoder: per-point MLP + masked symmetric pool.

    Input ``(B, N, 4)`` = (channel, x, y, energy) plus a ``(B, N)`` mask.
    Coordinates are rescaled to [-1, 1] by the caller-provided ``grid_size``
    so the network sees a consistent scale across LR/HR resolutions sharing
    one architecture. Output is a single ``(B, d_model)`` summary per sample
    (mean + max pool, concatenated) that is permutation-invariant in the
    points: any row permutation of the input (with the mask permuted to
    match) produces the identical output, verified in
    ``tests/test_pointcloud.py::test_encoder_is_permutation_invariant``.
    """

    def __init__(self, d_model: int = 128, hidden: int = 128) -> None:
        super().__init__()
        # channel, x, y, energy (energy passed through log1p for dynamic range,
        # matching the log1p+z-score convention the grid pipeline already uses
        # for HR energies elsewhere in this project's normalization).
        self.point_mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model),
        )
        self.d_model = d_model

    @staticmethod
    def rescale_xy(x: Tensor, y: Tensor, grid_size: int) -> tuple[Tensor, Tensor]:
        """Shared coordinate normalization to ``[-1, 1]``, factored out so
        ``SetGenerator``'s query-seeding (Stage B) and this encoder's own
        feature construction cannot silently diverge on the rescaling
        formula."""
        x_n = (x / max(grid_size - 1, 1)) * 2 - 1
        y_n = (y / max(grid_size - 1, 1)) * 2 - 1
        return x_n, y_n

    def forward(
        self, hits: Tensor, mask: Tensor, grid_size: int, return_points: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Returns the pooled ``(B, 2*d_model)`` summary by default.

        With ``return_points=True``, ALSO returns the pre-pool per-point
        embeddings ``(B, N, d_model)`` (the pooled summary is still returned
        first, unchanged, so ``SetDiscriminator``'s existing call site needs
        no change). This is the fix for ``SetGenerator``'s conditioning
        bottleneck: cross-attending to these N embeddings instead of the
        single pooled vector gives the decoder real per-region structure
        instead of one number per feature summarizing the whole input (see
        ``SetGenerator``'s module docstring for the diagnosed failure this
        fixes). Sparse raster cosine overlap cannot establish whether a
        model conditions on its input; use a cross-Chamfer matrix instead.

        Permutation invariance for the per-point output means: permuting the
        input rows permutes ``point_emb`` identically (each row's embedding
        depends only on that row's own features, not on row order) — NOT that
        the per-point embeddings are identical to each other. The POOLED
        output is still invariant in the usual sense (permutation-invariant
        aggregate). See
        ``tests/test_pointcloud_stage3.py::test_encoder_per_point_output_permutes_with_input``.
        """
        channel, x, y, energy = hits.unbind(dim=-1)
        x_n, y_n = self.rescale_xy(x, y, grid_size)
        e_n = torch.log1p(energy.clamp_min(0.0))
        feat = torch.stack([channel, x_n, y_n, e_n], dim=-1)  # (B,N,4)

        point_emb = self.point_mlp(feat)  # (B,N,d_model)
        m = mask.unsqueeze(-1).to(point_emb.dtype)  # (B,N,1)

        # Masked mean: padding rows contribute 0 to the sum and are excluded
        # from the count, so they cannot dilute the summary regardless of how
        # many padding rows exist (unlike an unmasked mean, whose value would
        # depend on n_max rather than on the actual point set).
        summed = (point_emb * m).sum(dim=1)
        count = m.sum(dim=1).clamp_min(1.0)
        mean_pool = summed / count

        # Masked max: padding rows are driven to -inf before the max so they
        # can never win it, then samples with zero real points (count==0,
        # entirely padding) are zeroed rather than left at -inf.
        masked_for_max = point_emb.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        max_pool = masked_for_max.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2*d_model)
        if return_points:
            return pooled, point_emb
        return pooled


class SetGenerator(nn.Module):
    """LR hits -> HR hits (fixed budget with learned existence).

    Decodes a fixed budget of ``hr_n_max`` candidate output points via
    query embeddings cross-attending to the LR point set (a standard
    set-prediction-network pattern, e.g. DETR's object queries applied here
    to hits instead of boxes).

    **Stage B: queries are seeded from LR hit positions, not free parameters.**
    An earlier version used ``self.queries = nn.Parameter(torch.randn(
    hr_n_max, d_model))`` -- the SAME fixed vectors for every sample in every
    batch, with zero information about that sample's input. Even after Stage
    A gave the decoder a full-length attention memory (see below), the
    decoder still had to learn purely through attention weights which query
    should look at which part of the input, with no prior. ``_build_queries``
    instead seeds each query's initial embedding from an actual LR hit's
    ``(x, y)`` position (see its docstring for how ``hr_n_max > n_real_hits``
    is handled via round-robin tiling with a learned per-repeat
    perturbation), so a query starts with an explicit "I am near LR hit j"
    prior instead of a blank slate.

    **Stage A conditioning bottleneck fix (memory, not queries).** An earlier
    version of this class also pooled
    the entire LR point set (up to ~600 hits) down to ONE ``(B, d_model)``
    vector and had every output query cross-attend to that single token. That
    structurally restricts cross-attention: a one-token softmax always has
    weight one, so queries cannot select different input regions. Earlier
    sparse-raster cosine measurements did not establish conditioning failure:
    small positional shifts can remove all exact pixel overlap. Cross-Chamfer
    checks of the 3-epoch Stage A and A+B checkpoints instead identify their
    own HR targets well above chance. This version
    cross-attends to the FULL per-point LR embedding sequence (from
    ``PermInvariantEncoder(..., return_points=True)``), masked so padded LR
    slots are never attended to, with the pooled summary kept as one
    additional global-context token concatenated onto that sequence — this
    gives the decoder real per-region structure while still retaining cheap
    global context (total input energy, hit count) that per-point attention
    alone would have to re-derive.

    Each output slot predicts ``(x, y, energy, existence_logit)``. ``x``/``y``
    are ``sigmoid``-bounded to ``[0, grid_size-1]`` so they are always valid
    HR coordinates (the audit's round-trip test requires integral in-range
    coordinates; this generator rounds them at the collate boundary, in
    ``train_pointcloud.py``, not inside the model, so the model's own forward
    pass stays fully differentiable for training).
    """

    def __init__(
        self,
        hr_n_max: int,
        d_model: int = 128,
        hidden: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hr_n_max = hr_n_max
        self.d_model = d_model
        self.encoder = PermInvariantEncoder(d_model=d_model, hidden=hidden)
        self.summary_proj = nn.Linear(2 * d_model, d_model)

        # Stage B: query positional embedding, seeded from LR hit (x, y)
        # rather than a fixed input-independent nn.Parameter. Small MLP over
        # the [-1,1]-rescaled coordinate (PermInvariantEncoder.rescale_xy),
        # not sinusoidal encoding: this project's coordinates are already
        # continuous floats in a bounded, small (<=128) range, so an MLP
        # needs no frequency-basis machinery to represent them and stays
        # consistent with PermInvariantEncoder.point_mlp's own approach to
        # the same kind of feature.
        self.query_pos_mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, d_model),
        )
        # Small learned per-repeat perturbation (see _build_queries): ADDED
        # to the coordinate embedding, not a replacement for it, so the
        # decoder retains free capacity beyond the raw position prior (a
        # pure coordinate embedding risks the model outputting near-copies
        # of LR positions instead of learning genuine super-resolution --
        # see this class's docstring). Indexed by repeat count so repeats 0..
        # k-1 of the same LR hit get DISTINCT perturbations and can
        # specialize to different nearby HR hits (the audit's finding that
        # each LR hit's energy typically splits into several HR hits).
        # Sized generously (32) rather than to hr_n_max/lr_n_max's exact
        # worst-case ratio, since that ratio varies by scale (audit: ~2.2x at
        # 64x, ~5.5x at 16x) and the repeat INDEX is what matters, not a
        # tight upper bound -- indices beyond 32 wrap via modulo (see
        # _build_queries), which just reuses perturbation directions rather
        # than erroring.
        self._max_repeat_slots = 32
        self.repeat_perturbation = nn.Parameter(
            torch.randn(self._max_repeat_slots, d_model) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=hidden,
            batch_first=True, dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.n_channels = 3  # fixed by this project's data (see hits.py COL_CHANNEL)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True),
            # x, y, energy, existence_logit, n_channels channel-class logits
            nn.Linear(hidden, 4 + self.n_channels),
        )

    def _build_queries(self, lr_hits: Tensor, lr_mask: Tensor, hr_grid: int) -> Tensor:
        """Build the ``(B, hr_n_max, d_model)`` query sequence, seeded from
        LR hit positions (Stage B). Separated from ``forward`` so tests can
        call it directly and inspect the query tensor without mocking the
        decoder.

        Per sample, real LR hits (``lr_mask``) are gathered, tiled to fill
        exactly ``hr_n_max`` slots (repeating hits when ``hr_n_max >
        n_real_hits``, which is the normal case -- HR carries 2-5x MORE hits
        than LR depending on scale, see ``docs/POINTCLOUD_AUDIT.md``), and
        each repeat gets a distinct learned perturbation added on top of its
        coordinate embedding so repeated seeds can specialize rather than
        collapse to identical queries.

        Samples with ZERO real LR hits (``lr_mask.sum() == 0`` for that row)
        have no position to seed from; their ``hr_n_max`` queries fall back
        to the coordinate embedding of ``(0, 0)`` (a defined, finite,
        differentiable value -- not a special-cased branch) plus the same
        per-repeat perturbation, so the forward pass stays batchable and
        produces finite output rather than requiring a fully-padded sample to
        be handled specially by the caller.
        """
        b, n_lr = lr_hits.shape[0], lr_hits.shape[1]
        device = lr_hits.device
        x_lr, y_lr = lr_hits[..., 1], lr_hits[..., 2]

        # Per-sample count of real LR hits, floored at 1 so modulo indexing
        # below is always well-defined (a zero-hit sample uses count=1 and
        # index 0, i.e. "hit slot 0" -- masked out immediately after via
        # has_any_hit, see below).
        n_real = lr_mask.sum(dim=1)  # (B,)
        n_real_safe = n_real.clamp_min(1)

        # For each of the hr_n_max output slots, pick which LR hit (by index
        # within that sample's real hits) seeds it: slot k uses real hit
        # (k mod n_real_safe), and repeat_index = k // n_real_safe selects
        # the perturbation. This tiles hits round-robin (hit 0, hit 1, ...,
        # hit n-1, hit 0, hit 1, ...) rather than repeating each hit
        # n_max/n_real times contiguously, so nearby output slots draw from
        # DIFFERENT LR regions rather than all early slots exhausting one hit
        # before moving to the next -- a minor but deliberate choice, since
        # slot ORDER has no other meaning here (queries are consumed by
        # permutation-agnostic cross-attention).
        slot_idx = torch.arange(self.hr_n_max, device=device).unsqueeze(0)  # (1, hr_n_max)
        hit_rank = slot_idx % n_real_safe.unsqueeze(1)  # (B, hr_n_max): which real hit, by rank
        repeat_idx = (slot_idx // n_real_safe.unsqueeze(1)) % self._max_repeat_slots  # (B, hr_n_max)

        # Map each sample's real-hit RANKS (0..n_real-1) to actual indices
        # into the padded (B, N_lr) tensor. argsort of (~mask) puts real
        # (mask=True -> ~mask=False=0) hits first, in their original relative
        # order, so rank r's position is real_positions[:, r].
        real_positions = torch.argsort((~lr_mask).long(), dim=1, stable=True)  # (B, N_lr)
        hit_index = torch.gather(real_positions, 1, hit_rank)  # (B, hr_n_max): index into N_lr

        x_seed = torch.gather(x_lr, 1, hit_index)  # (B, hr_n_max)
        y_seed = torch.gather(y_lr, 1, hit_index)

        has_any_hit = (n_real > 0).unsqueeze(1)  # (B, 1)
        x_seed = torch.where(has_any_hit, x_seed, torch.zeros_like(x_seed))
        y_seed = torch.where(has_any_hit, y_seed, torch.zeros_like(y_seed))

        x_n, y_n = PermInvariantEncoder.rescale_xy(x_seed, y_seed, hr_grid)
        pos_emb = self.query_pos_mlp(torch.stack([x_n, y_n], dim=-1))  # (B, hr_n_max, d_model)

        perturb = self.repeat_perturbation[repeat_idx]  # (B, hr_n_max, d_model)
        return pos_emb + perturb

    def forward(
        self, lr_hits: Tensor, lr_mask: Tensor, lr_grid: int, hr_grid: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(hits, existence_logits, channel_logits)``.

        ``hits``: ``(B, hr_n_max, 4)`` = (channel, x, y, energy) — the SAME
        4-column layout ``PermInvariantEncoder`` (and hence
        ``SetDiscriminator``) expects, so generator output can be fed straight
        into the discriminator or re-rasterized without a reshape. Channel is
        ``argmax`` of a per-slot 3-way softmax over this project's fixed 3
        calorimeter channels (see ``hits.py``'s ``COL_CHANNEL`` convention) —
        discrete and therefore not differentiable through the argmax itself;
        gradient for channel assignment flows through the softmax
        cross-entropy term in ``losses.py`` instead, the same
        differentiable/non-differentiable split ``topk_hits`` documents for
        coordinates (values differentiable, indices not).

        Energy is ``softplus``'d so it is non-negative (calorimeter deposits
        cannot be negative — see the Stage 1 audit's finding that the
        *source* HR does carry a small negative pedestal, which this
        generator deliberately does not attempt to reproduce, consistent with
        the grid pipeline's own ``weighted_l1_loss`` clamping it out).

        ``existence_logits``: ``(B, hr_n_max)`` raw logits (pre-sigmoid); loss
        functions apply ``BCEWithLogitsLoss`` directly against the ground
        truth mask for numerical stability rather than thresholding here.

        ``channel_logits``: ``(B, hr_n_max, 3)`` raw per-channel logits,
        returned alongside ``hits`` (whose channel column is the detached
        argmax of these) so a training loop can apply cross-entropy against
        the ground-truth channel directly, the same differentiable-loss-input/
        detached-hit-output split ``topk_hits`` uses for coordinates.
        """
        b = lr_hits.shape[0]
        pooled, point_emb = self.encoder(lr_hits, lr_mask, lr_grid, return_points=True)
        summary_tok = self.summary_proj(pooled).unsqueeze(1)  # (B, 1, d_model)

        # Full per-point LR sequence + one global-summary token, so the
        # decoder has real per-region structure to attend to (see class
        # docstring) plus cheap global context. point_emb is already in
        # d_model (PermInvariantEncoder's point_mlp output dim), so it needs
        # no extra projection to match summary_tok's width.
        memory = torch.cat([point_emb, summary_tok], dim=1)  # (B, N_lr+1, d_model)
        # True = ignore, confirmed against PyTorch's actual behavior (not
        # assumed from the docstring, which does not state the polarity) --
        # see the empirical check referenced in this module's tests. The
        # summary token is never padding, so it gets an explicit False.
        real_lr = lr_mask.bool()
        summary_valid = torch.ones(b, 1, dtype=torch.bool, device=lr_mask.device)
        memory_key_padding_mask = ~torch.cat([real_lr, summary_valid], dim=1)

        queries = self._build_queries(lr_hits, lr_mask, hr_grid)  # (B, hr_n_max, d_model)
        decoded = self.decoder(
            tgt=queries, memory=memory, memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, hr_n_max, d_model)

        raw = self.head(decoded)  # (B, hr_n_max, 4 + n_channels)
        x = torch.sigmoid(raw[..., 0]) * (hr_grid - 1)
        y = torch.sigmoid(raw[..., 1]) * (hr_grid - 1)
        energy = F.softplus(raw[..., 2])
        existence_logits = raw[..., 3]
        channel_logits = raw[..., 4:]
        channel = channel_logits.argmax(dim=-1).to(raw.dtype).detach()  # non-differentiable, see docstring
        hits = torch.stack([channel, x, y, energy], dim=-1)
        return hits, existence_logits, channel_logits


class SetDiscriminator(nn.Module):
    """Permutation-invariant discriminator over (LR hits, HR/SR hits).

    Encodes each point set independently (shared ``PermInvariantEncoder``
    weights between LR and the HR/SR candidate, so the comparison is made in
    the same learned feature space) and classifies the concatenation. This
    judges the *distribution* of hits — is this a plausible HR hit set given
    this LR hit set — the analogue of the grid ``Discriminator``'s (lr, hr)
    patch classification, but with no spatial patches: a point cloud has no
    grid to take patches of, so the comparison is necessarily global rather
    than local, which is a real capability difference from the PatchGAN this
    replaces, not just a representation change.
    """

    def __init__(self, d_model: int = 128, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = PermInvariantEncoder(d_model=d_model, hidden=hidden)
        self.head = nn.Sequential(
            nn.Linear(4 * d_model, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, lr_hits: Tensor, lr_mask: Tensor, lr_grid: int,
        hr_hits: Tensor, hr_mask: Tensor, hr_grid: int,
    ) -> Tensor:
        lr_summary = self.encoder(lr_hits, lr_mask, lr_grid)  # (B, 2*d_model)
        hr_summary = self.encoder(hr_hits, hr_mask, hr_grid)  # (B, 2*d_model)
        return self.head(torch.cat([lr_summary, hr_summary], dim=-1)).squeeze(-1)  # (B,)
