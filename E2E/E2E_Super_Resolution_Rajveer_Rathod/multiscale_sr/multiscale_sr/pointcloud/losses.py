"""Point-cloud losses for the Stage 2 experiment: same CNN generator, only the
reconstruction loss changes.

All three losses here operate on the ``(energies, coords)`` pairs returned by
``pointcloud.hits.topk_hits`` — i.e. they are computed on the *existing* dense
generator's output, extracted as a fixed-K point set. Nothing here requires a
set-to-set model; that is Stage 3.

Conventions matched to ``engine.semd_loss`` deliberately, so a reviewer who
knows one recognises the other:

  * ``chunk`` bounds memory the same way ``semd_chunk`` does for SEMD — these
    losses are O(K^2) per sample, exactly like the dense SEMD expression.
  * Energies are expected on the SAME (raw, denormalized) scale ``semd_loss``
    uses, for the same reason: normalized tensors have a negative floor that
    is not meaningful as point-cloud "mass".
  * ``lambda_chamfer`` / ``lambda_sinkhorn`` are OFF by default (0.0) in
    ``train.py`` and guarded the same way ``lambda_semd`` is, so a disabled
    run is bit-for-bit identical to the existing pipeline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .hits import topk_hits


def _pairwise_sq_dist(a: Tensor, b: Tensor) -> Tensor:
    """(B,Ka,2) x (B,Kb,2) -> (B,Ka,Kb) squared Euclidean distance, chunked.

    Plain ``torch.cdist`` materializes the full (B,Ka,Kb) tensor at once;
    everything here does too (no chunking over the pair axis) because K is
    bounded by the caller's ``topk`` argument, exactly as ``semd_loss`` bounds
    memory via ``semd_topk`` rather than by chunking the distance matrix
    itself. Batch-dimension chunking is handled by the caller passing a
    smaller slice, matching ``semd_images``' ``chunk`` parameter.
    """
    return torch.cdist(a, b, p=2.0) ** 2


def chamfer_loss(
    pred_raw: Tensor,
    target_raw: Tensor,
    topk: int = 128,
    chunk: int = 1,
) -> Tensor:
    """Energy-weighted bidirectional Chamfer distance over top-K hits.

    Extracts the ``topk`` brightest pixels from ``pred_raw`` (differentiably,
    via ``topk_hits``) and from ``target_raw`` (no gradient needed — HR is the
    fixed target), then computes

        mean_i [ w_pred_i * min_j dist(pred_i, target_j)^2 ]
      + mean_j [ w_target_j * min_i dist(pred_i, target_j)^2 ]

    weighted by each hit's share of that sample's total top-K energy, so a
    sample's loss does not simply scale with its total deposited energy (the
    same reasoning as ``weighted_l1_loss``'s per-pixel weighting, applied
    per-hit instead of per-pixel).

    Known failure mode, stated up front rather than discovered in results:
    Chamfer's nearest-neighbour matching has no preference for using every
    target point once, so it is prone to producing clumped, blurry point sets
    — this is why Stage 2 also implements ``sinkhorn_loss`` as an alternative
    with better-behaved (one-to-one, mass-conserving) gradients.

    ``chunk`` splits the batch dimension into groups of ``chunk`` samples, the
    same memory/latency tradeoff ``semd_loss``'s ``chunk`` argument offers,
    since the K x K distance matrix is the dominant cost per sample.
    """
    if pred_raw.shape != target_raw.shape:
        raise ValueError(
            f"pred_raw {tuple(pred_raw.shape)} and target_raw {tuple(target_raw.shape)} "
            "must have the same shape"
        )
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")

    b = pred_raw.shape[0]
    total = pred_raw.new_zeros(())
    for start in range(0, b, chunk):
        end = min(start + chunk, b)
        pe, pc = topk_hits(pred_raw[start:end], topk)
        with torch.no_grad():
            te, tc = topk_hits(target_raw[start:end], topk)

        p_xy = pc[..., 1:3]  # (chunk, K, 2): (x, y)
        t_xy = tc[..., 1:3]
        d2 = _pairwise_sq_dist(p_xy, t_xy)  # (chunk, K, K)

        w_p = pe / pe.sum(dim=1, keepdim=True).clamp_min(1e-12)
        w_t = te / te.sum(dim=1, keepdim=True).clamp_min(1e-12)

        p_to_t = (d2.min(dim=2).values * w_p).sum(dim=1)
        t_to_p = (d2.min(dim=1).values * w_t).sum(dim=1)
        total = total + (p_to_t + t_to_p).sum()
    return total / b


def sinkhorn_loss(
    pred_raw: Tensor,
    target_raw: Tensor,
    topk: int = 128,
    eps: float = 4.0,
    iters: int = 30,
    chunk: int = 1,
) -> Tensor:
    """Entropy-regularised optimal transport over top-K hits, batched and chunked.

    Same top-K extraction as ``chamfer_loss``, but the assignment is a soft
    transport plan (log-domain Sinkhorn) rather than nearest-neighbour, which
    gives every point pressure to be matched by *something* rather than
    letting many-to-one collapses go unpenalised — the standard fix for
    Chamfer's clumping tendency.

    ``eps`` is in squared-pixel units (matching the cost, which is squared
    Euclidean on pixel coordinates); larger eps blurs the plan more but
    converges faster and more stably.

    **This loss is not zero even for a perfect reconstruction** at the default
    ``eps=4.0``: entropic regularisation spreads the transport plan across
    near-equidistant matches whenever ``eps`` is comparable to the point set's
    own spacing (empirically ~1-12px on this data's top-128 hits), so
    self-transport cost stays measurably above 0 (confirmed in
    ``tests/test_pointcloud.py::test_sinkhorn_self_transport_shrinks_toward_zero_as_eps_shrinks``).
    Track *relative* change in ``val_sinkhorn`` across epochs/checkpoints, not
    its absolute value, and do not compare across different ``eps`` settings.
    A smaller eps (e.g. 0.5-1.0) trades this bias for slower, less stable
    convergence — the same tradeoff ``scripts/pointcloud_audit.py`` makes
    differently (eps=1.0) because it runs Sinkhorn once for a measurement, not
    every training step.

    **The potentials are replaced each iteration, not accumulated** — this is
    the exact bug found and fixed in the Stage 1 audit script
    (``scripts/pointcloud_audit.py::sinkhorn_xy``), where accumulating drove
    the potentials to +/-inf within ~10 iterations. See
    ``tests/test_pointcloud.py::test_sinkhorn_converges_and_matches_exact_ot_on_a_known_case``
    for the regression test on that exact failure mode.
    """
    if pred_raw.shape != target_raw.shape:
        raise ValueError(
            f"pred_raw {tuple(pred_raw.shape)} and target_raw {tuple(target_raw.shape)} "
            "must have the same shape"
        )
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    b = pred_raw.shape[0]
    total = pred_raw.new_zeros(())
    for start in range(0, b, chunk):
        end = min(start + chunk, b)
        pe, pc = topk_hits(pred_raw[start:end], topk)
        with torch.no_grad():
            te, tc = topk_hits(target_raw[start:end], topk)

        p_xy = pc[..., 1:3]
        t_xy = tc[..., 1:3]
        cost = _pairwise_sq_dist(p_xy, t_xy)  # (chunk, K, K)

        log_wp = (pe / pe.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()
        log_wt = (te / te.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()

        f = torch.zeros_like(log_wp)
        g = torch.zeros_like(log_wt)
        for _ in range(iters):
            f = eps * (log_wp - torch.logsumexp((g.unsqueeze(1) - cost) / eps, dim=2))
            g = eps * (log_wt - torch.logsumexp((f.unsqueeze(2) - cost) / eps, dim=1))
        plan = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - cost) / eps)
        total = total + (plan * cost).sum(dim=(1, 2)).sum()
    return total / b


def masked_sinkhorn_loss(
    pred_hits: Tensor,
    target_hits: Tensor,
    target_mask: Tensor,
    eps: float = 4.0,
    iters: int = 30,
    lambda_energy_value: float = 1.0,
) -> Tensor:
    """Padded-batch counterpart to ``sinkhorn_loss``, for ``SetGenerator``'s
    output directly (see ``masked_chamfer_loss``'s docstring for why the two
    Stage 3 losses take no ``pred_mask``: every generator slot is a real
    prediction, existence is a separate learned score, not padding).

    Uses each point's OWN energy as its transport mass (matching
    ``sinkhorn_loss``'s ``log_wp``/``log_wt`` convention) rather than uniform
    mass, so — like ``masked_chamfer_loss``'s ``energy_weighted`` option —
    a bright target hit exerts more pull on the transport plan than a dim
    one. Unlike Chamfer's independent nearest-neighbour search per point,
    Sinkhorn's transport plan is a genuine joint assignment across the whole
    set, which is what gives it no-free-lunch pressure against the "spread
    the same total energy over more, dimmer hits" solution Chamfer permits
    (see ``masked_chamfer_loss``'s docstring for the checkpoint that produced
    exactly that failure).

    Padded prediction slots have no analogue here (see above), but padded
    TARGET rows are excluded from the transport by giving them a cost of
    +inf to every predicted point — mirroring how ``masked_chamfer_loss``
    excludes them from the nearest-target search, rather than letting a
    padding row at (0,0) with zero mass silently participate (zero mass
    would make it nearly costless to match, an easy sink for probability
    mass Sinkhorn would otherwise have to place on a real point).

    **``lambda_energy_value`` is not optional in practice, despite the
    default parity with ``masked_chamfer_loss``'s parameter of the same
    name.** Sinkhorn's transport masses (``log_wp``/``log_wt``) are each
    normalized to sum to 1 *within their own set* — so the loss only ever
    sees each hit's *fraction* of that sample's total energy, never its
    absolute value. A prediction whose peak drops from energy 30 to energy
    10 while every other hit drops proportionally looks like an *identical*
    mass distribution to Sinkhorn and scores near-zero on the position cost
    alone (confirmed empirically: with ``lambda_energy_value=0`` this
    function scored a smeared reconstruction as *better* than a sharp one on
    the exact case that motivated adding this parameter — see
    ``tests/test_pointcloud_stage3.py::test_masked_sinkhorn_prefers_sharp_reconstruction_over_smeared``).
    Adding the same per-pair ``|pred_energy - target_energy|`` term used in
    ``masked_chamfer_loss`` to the transport cost restores that pressure.
    """
    if pred_hits.shape[0] != target_hits.shape[0]:
        raise ValueError("batch size mismatch between pred and target")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    p_xy, p_e = pred_hits[..., 1:3], pred_hits[..., 3]
    t_xy, t_e = target_hits[..., 1:3], target_hits[..., 3]
    cost = _pairwise_sq_dist(p_xy, t_xy)  # (B, Np, Nt)
    if lambda_energy_value > 0.0:
        # Absolute energy-value pressure the mass-fraction-normalized
        # transport plan cannot see on its own — see docstring.
        e_gap = (p_e.unsqueeze(2) - t_e.unsqueeze(1)).abs()
        cost = cost + lambda_energy_value * e_gap
    cost = cost.masked_fill(~target_mask.unsqueeze(1), float("inf"))

    # Masses: predicted energies as-is (every slot real); target energies
    # zeroed at padding before normalizing, so padding contributes no mass
    # for the marginal constraint to satisfy.
    p_mass = p_e / p_e.sum(dim=1, keepdim=True).clamp_min(1e-12)
    t_e_masked = torch.where(target_mask, t_e, torch.zeros_like(t_e))
    t_mass = t_e_masked / t_e_masked.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_wp = p_mass.clamp_min(1e-12).log()
    log_wt = t_mass.clamp_min(1e-12).log()

    f = torch.zeros_like(log_wp)
    g = torch.zeros_like(log_wt)
    for _ in range(iters):
        f = eps * (log_wp - torch.logsumexp((g.unsqueeze(1) - cost) / eps, dim=2))
        g_input = (f.unsqueeze(2) - cost) / eps
        # Padded target columns are all -inf after the cost mask above;
        # logsumexp over an all -inf column would be -inf -> nan once it
        # feeds back into g's own update, so those columns are excluded from
        # g and just left at their previous (zero-mass-consistent) value.
        g_new = eps * (log_wt - torch.logsumexp(g_input, dim=1))
        g = torch.where(target_mask, g_new, g)
    plan = torch.exp(((f.unsqueeze(2) + g.unsqueeze(1) - cost) / eps).clamp(max=30.0))
    finite_cost = cost.masked_fill(~torch.isfinite(cost), 0.0)
    return (plan * finite_cost).sum(dim=(1, 2)).mean()


def masked_chamfer_loss(
    pred_hits: Tensor,
    target_hits: Tensor,
    target_mask: Tensor,
    energy_weighted: bool = True,
    alpha: float = 5.0,
    lambda_energy_value: float = 1.0,
) -> Tensor:
    """Chamfer distance between a generator's padded output and a padded
    target hit-list batch.

    Unlike ``chamfer_loss`` (which extracts top-K from a dense image),
    this operates directly on ``pointcloud/collate.py``'s padded ``(B, N, 4)``
    ``(channel, x, y, energy)`` layout, used by ``train_pointcloud.py`` for
    the Stage 3 set-to-set generator, where there is no dense image to extract
    from in the first place.

    ``pred_hits`` takes NO mask, deliberately: it must be ``SetGenerator``'s
    raw output, where every one of the ``hr_n_max`` slots is a real
    coordinate/energy prediction (existence is a separate learned SCORE per
    slot, supervised by ``existence_loss``, not an absence of a value the way
    ``target_mask`` marks unused rows in a fixed-size batch tensor). Passing a
    masked/externally-padded ``pred_hits`` here is a caller error: this
    function has no way to distinguish "generator predicts coordinate (0,0)"
    from "this slot is padding" — that ambiguity is exactly why an earlier
    version of this function accepted (and silently ignored) a ``pred_mask``
    argument, which produced wrong results whenever the caller assumed it did
    something. If you have a padded, externally-masked prediction (rather than
    a generator's own output), use ``chamfer_loss`` on a dense reconstruction
    of it instead, where masking is well-defined via the image's own zeros.

    ``target_hits``/``target_mask`` ARE masked (real HR hits vs. padding),
    since padded target rows must never be treated as points to match
    against — that mask is well-defined because target padding really is
    "this slot has nothing", unlike a generator's prediction slots.

    **Peak smearing (found in evaluation, not just theorized).** Training a
    model with the position-only version of this loss produced a real
    checkpoint whose predicted top-5 energies (10.9, 9.2, 7.4, 6.2, 6.1) were
    flatter than HR's (27.6, 16.6, 6.2, 6.0, 4.0) despite matching total
    energy closely — the same "spread energy across more hits" degenerate
    solution ``weighted_l1_loss`` was built to prevent on the dense grid, and
    the exact Chamfer failure mode (favors approximately-right position and
    magnitude per matched pair, with no pressure toward keeping bright pairs
    bright) flagged in this function's Stage 2 sibling before Stage 3 was
    ever trained. Two additions address it directly:

    - ``energy_weighted``: the TARGET-side term is weighted by
      ``1 + alpha * target_energy`` (same weighting formula and same default
      ``alpha`` as ``weighted_l1_loss``, so the two are tuned on a comparable
      scale), so a badly-matched bright HR hit costs far more than a
      badly-matched dim one — pressure toward preserving peaks, not just
      average position.
    - ``lambda_energy_value``: an explicit ``|pred_energy - target_energy|``
      term at each nearest-neighbour PAIR (both directions), on top of the
      pure position distance. The position-only distance has zero gradient
      pressure on energy VALUES at all — two hits at the same (x,y) with
      wildly different energy score identically on position distance alone.
    """
    if pred_hits.shape[0] != target_hits.shape[0]:
        raise ValueError("batch size mismatch between pred and target")

    p_xy = pred_hits[..., 1:3]
    t_xy = target_hits[..., 1:3]
    p_e = pred_hits[..., 3]
    t_e = target_hits[..., 3]
    d2 = _pairwise_sq_dist(p_xy, t_xy)  # (B, Np, Nt)

    # Per-pair energy-value gap, added to the position distance so a nearest
    # neighbour with the right (x,y) but the wrong energy is still penalized.
    if lambda_energy_value > 0.0:
        e_gap = (p_e.unsqueeze(2) - t_e.unsqueeze(1)).abs()  # (B, Np, Nt)
        d2 = d2 + lambda_energy_value * e_gap

    # Exclude padded target columns from the nearest-target search entirely,
    # rather than letting them win a min() at distance 0 (they are all at
    # coordinate (0,0), which is a valid pixel and would otherwise attract
    # every prediction near the origin).
    d2_masked = d2.masked_fill(~target_mask.unsqueeze(1), float("inf"))
    p_to_t = d2_masked.min(dim=2).values  # (B, Np)
    p_to_t = torch.where(torch.isfinite(p_to_t), p_to_t, torch.zeros_like(p_to_t))

    # p_to_t direction needs no pred-side mask (every predicted slot counts,
    # per the docstring above); t_to_p DOES need to skip padded targets, which
    # happens below via target_mask rather than by masking d2 a second time.
    t_to_p = d2.min(dim=1).values  # (B, Nt)
    t_to_p = torch.where(target_mask, t_to_p, torch.zeros_like(t_to_p))

    if energy_weighted:
        # Same formula as weighted_l1_loss: w = 1 + alpha * relu(energy).
        # relu is a no-op here since hit energies are already non-negative
        # (softplus'd by the generator, clamped by dense_to_hits' threshold
        # on the target side), kept for exact parity with the dense-grid
        # weighting so alpha means the same thing in both places.
        w = 1.0 + alpha * t_e.clamp_min(0.0)
        w = torch.where(target_mask, w, torch.zeros_like(w))
        t_to_p_weighted = (t_to_p * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
    else:
        t_to_p_weighted = t_to_p.sum(dim=1) / target_mask.sum(dim=1).clamp_min(1)

    return (p_to_t.mean(dim=1) + t_to_p_weighted).mean()


def existence_loss(existence_logits: Tensor, target_mask: Tensor) -> Tensor:
    """Binary cross-entropy: does this generator output slot correspond to a
    real hit? Trains the model to predict cardinality rather than imposing it
    — see ``models.SetGenerator``'s module docstring for why this replaces
    the fixed top-K assumption a dense-grid approach can rely on but a
    variable-cardinality point cloud cannot.

    ``target_mask`` must have the same shape as ``existence_logits`` and be
    aligned to predicted slots by ``matching.matched_supervision``. Unmatched
    predictions are supervised toward non-existence. This low-level BCE
    does not infer correspondence from target storage order.
    """
    if existence_logits.shape != target_mask.shape:
        raise ValueError(
            f"existence_logits {tuple(existence_logits.shape)} and target_mask "
            f"{tuple(target_mask.shape)} must match"
        )
    return F.binary_cross_entropy_with_logits(existence_logits, target_mask.float())


def channel_loss(channel_logits: Tensor, target_channel: Tensor, target_mask: Tensor) -> Tensor:
    """Cross-entropy on the predicted channel class, masked to real target hits.

    Targets and mask must be aligned to predicted slots by
    ``matching.matched_supervision``, as for ``existence_loss``.
    """
    if channel_logits.shape[:2] != target_channel.shape:
        raise ValueError("channel_logits/target_channel shape mismatch")
    ce = F.cross_entropy(
        channel_logits.reshape(-1, channel_logits.shape[-1]),
        target_channel.reshape(-1).round().long(),
        reduction="none",
    ).reshape(target_channel.shape)
    denom = target_mask.sum().clamp_min(1)
    return (ce * target_mask.float()).sum() / denom


def energy_conservation_loss(pred_raw: Tensor, target_raw: Tensor) -> Tensor:
    """Point-cloud analogue of ``engine.physics_loss_ratio``.

    Identical formulation (|sum(E_pred)/sum(E_true) - 1|, mean over the batch)
    applied to the raw dense tensors, not restricted to any top-K subset —
    total energy is a property of the whole image, and truncating to K hits
    before summing would make this loss track "how much energy is in the
    brightest K pixels" rather than total energy, silently changing its
    meaning as K is tuned. Kept as a separate function (rather than reusing
    ``physics_loss_ratio`` directly) only so this subpackage's loss surface is
    self-contained and documented for the point-cloud context; the underlying
    math is identical and this delegates to it.
    """
    from ..engine import physics_loss_ratio

    return physics_loss_ratio(pred_raw, target_raw)
