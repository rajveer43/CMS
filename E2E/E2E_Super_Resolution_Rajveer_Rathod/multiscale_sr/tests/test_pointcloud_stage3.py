"""Stage 3: padded batching, the set-to-set generator/discriminator, and the
losses that train them. Kept in a separate file from test_pointcloud.py
(Stage 1/2) since these exercise a materially different code path (padded
batches and models, not dense-image extraction)."""
import torch
import pytest

from multiscale_sr.pointcloud.hits import dense_to_hits
from multiscale_sr.pointcloud.collate import (
    dense_pair_to_padded_hits,
    pad_hits,
    unpad_hits,
)
from multiscale_sr.pointcloud.models import (
    PermInvariantEncoder,
    SetDiscriminator,
    SetGenerator,
)
from multiscale_sr.pointcloud.losses import (
    channel_loss,
    existence_loss,
    masked_chamfer_loss,
    masked_sinkhorn_loss,
)


def sparse_batch(b=3, c=3, h=16, w=16, density=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(b, c, h, w)
    mask = torch.rand(b, c, h, w, generator=g) < density
    x[mask] = torch.rand(int(mask.sum()), generator=g) * 100.0 + 1e-3
    return x


# --------------------------------------------------------------------------- #
# collate.py
# --------------------------------------------------------------------------- #
def test_pad_hits_round_trips_through_unpad():
    x = sparse_batch(b=4, h=12, w=12, density=0.1, seed=1)
    hits = dense_to_hits(x)
    padded, mask = pad_hits(hits, batch_size=4, n_max=64)
    back = unpad_hits(padded, mask)
    # Row order may differ (padding removed), so compare as sets of tuples.
    a = {tuple(round(v, 4) for v in row.tolist()) for row in hits}
    b = {tuple(round(v, 4) for v in row.tolist()) for row in back}
    assert a == b


def test_pad_hits_mask_matches_true_hit_count():
    x = torch.zeros(3, 1, 5, 5)
    x[0, 0, 0, 0] = 1.0
    x[0, 0, 1, 1] = 1.0
    x[1, 0, 2, 2] = 1.0
    # sample 2 has zero hits
    padded, mask = pad_hits(dense_to_hits(x), batch_size=3, n_max=8)
    assert mask.sum(dim=1).tolist() == [2, 1, 0]
    assert padded.shape == (3, 8, 4)
    assert padded[~mask].abs().sum() == 0  # padding rows are exactly zero


def test_pad_hits_truncates_to_brightest_when_over_n_max():
    x = torch.zeros(1, 1, 8, 8)
    for i, e in enumerate([5.0, 1.0, 9.0, 3.0]):
        x[0, 0, i, i] = e
    padded, mask = pad_hits(dense_to_hits(x), batch_size=1, n_max=2)
    assert mask.sum().item() == 2
    kept_energies = sorted(padded[0, mask[0], 3].tolist())
    assert kept_energies == [5.0, 9.0]  # the two brightest, not the first two


def test_pad_hits_rejects_bad_n_max():
    with pytest.raises(ValueError):
        pad_hits(torch.zeros(0, 5), batch_size=1, n_max=0)


def test_unpad_hits_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        unpad_hits(torch.zeros(2, 5, 4), torch.zeros(2, 6, dtype=torch.bool))


def test_dense_pair_to_padded_hits_shapes():
    lr = sparse_batch(b=2, h=8, w=8, density=0.2, seed=2)
    hr = sparse_batch(b=2, h=16, w=16, density=0.1, seed=3)
    lr_hits, lr_mask, hr_hits, hr_mask = dense_pair_to_padded_hits(lr, hr, lr_n_max=20, hr_n_max=40)
    assert lr_hits.shape == (2, 20, 4) and lr_mask.shape == (2, 20)
    assert hr_hits.shape == (2, 40, 4) and hr_mask.shape == (2, 40)


# --------------------------------------------------------------------------- #
# models.py
# --------------------------------------------------------------------------- #
def _random_hits(b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    channel = torch.randint(0, 3, (b, n), generator=g).float()
    x = torch.randint(0, 16, (b, n), generator=g).float()
    y = torch.randint(0, 16, (b, n), generator=g).float()
    energy = torch.rand(b, n, generator=g) * 10
    mask = torch.rand(b, n, generator=g) > 0.3
    hits = torch.stack([channel, x, y, energy], dim=-1)
    hits = hits * mask.unsqueeze(-1)  # zero out padding, matching pad_hits' convention
    return hits, mask


def test_encoder_is_permutation_invariant():
    torch.manual_seed(0)
    enc = PermInvariantEncoder(d_model=8, hidden=16)
    hits, mask = _random_hits(2, 10, seed=5)
    out1 = enc(hits, mask, grid_size=16)

    perm = torch.randperm(10)
    out2 = enc(hits[:, perm], mask[:, perm], grid_size=16)
    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


def test_encoder_ignores_padding_content():
    """Two batches differing only in the VALUES at masked-out rows must
    produce identical encodings -- padding content must never leak in."""
    torch.manual_seed(0)
    enc = PermInvariantEncoder(d_model=8, hidden=16)
    hits, mask = _random_hits(2, 10, seed=6)
    out1 = enc(hits, mask, grid_size=16)

    garbage = hits.clone()
    garbage[~mask] = torch.rand_like(garbage[~mask]) * 999
    out2 = enc(garbage, mask, grid_size=16)
    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


def test_encoder_handles_all_padding_sample():
    """A sample with zero real hits (fully masked) must not produce NaN/Inf."""
    enc = PermInvariantEncoder(d_model=8, hidden=16)
    hits = torch.zeros(1, 5, 4)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    out = enc(hits, mask, grid_size=16)
    assert torch.isfinite(out).all()


def test_encoder_per_point_output_permutes_with_input():
    """return_points=True: permuting input ROWS must permute the per-point
    embeddings the SAME way (each row's embedding is a pure function of that
    row's own features, independent of row order) -- not that the per-point
    embeddings are identical to each other, which is what the pooled-output
    invariance test checks instead."""
    torch.manual_seed(0)
    enc = PermInvariantEncoder(d_model=8, hidden=16)
    hits, mask = _random_hits(2, 10, seed=5)
    pooled1, points1 = enc(hits, mask, grid_size=16, return_points=True)

    perm = torch.randperm(10)
    pooled2, points2 = enc(hits[:, perm], mask[:, perm], grid_size=16, return_points=True)

    # Pooled summary: still invariant (order doesn't matter).
    torch.testing.assert_close(pooled1, pooled2, atol=1e-5, rtol=1e-5)
    # Per-point: row i of points2 should equal row perm[i] of points1 --
    # i.e. points1 permuted by `perm` equals points2, NOT points1==points2.
    torch.testing.assert_close(points1[:, perm], points2, atol=1e-5, rtol=1e-5)
    # And confirm it's genuinely NOT the degenerate case (rows differ from
    # each other) -- a buggy implementation that returns the pooled vector
    # broadcast to every row would pass the two checks above trivially.
    assert not torch.allclose(points1[:, 0], points1[:, 1], atol=1e-3)


def test_encoder_return_points_default_false_matches_pooled_only_call():
    """The new parameter must not change default behavior for existing
    callers (SetDiscriminator calls without return_points)."""
    torch.manual_seed(0)
    enc = PermInvariantEncoder(d_model=8, hidden=16)
    hits, mask = _random_hits(2, 10, seed=5)
    pooled_only = enc(hits, mask, grid_size=16)
    pooled_with_points, _ = enc(hits, mask, grid_size=16, return_points=True)
    torch.testing.assert_close(pooled_only, pooled_with_points, atol=1e-6, rtol=1e-6)


def test_generator_output_shapes_and_ranges():
    torch.manual_seed(0)
    gen = SetGenerator(hr_n_max=24, d_model=8, hidden=16, n_layers=1, n_heads=2)
    lr_hits, lr_mask = _random_hits(3, 12, seed=7)
    hits, exist_logits, chan_logits = gen(lr_hits, lr_mask, lr_grid=16, hr_grid=32)

    assert hits.shape == (3, 24, 4)
    assert exist_logits.shape == (3, 24)
    assert chan_logits.shape == (3, 24, 3)

    channel, x, y, energy = hits.unbind(dim=-1)
    assert (channel >= 0).all() and (channel <= 2).all()
    assert (x >= 0).all() and (x <= 31).all()
    assert (y >= 0).all() and (y <= 31).all()
    assert (energy >= 0).all()  # softplus
    assert torch.isfinite(hits).all()


def test_generator_ignores_padded_lr_content():
    """Padded LR rows (mask=False) must never influence the generator's
    output, regardless of what garbage values they hold -- the
    memory_key_padding_mask must actually be wired correctly (True=ignore,
    verified empirically against PyTorch's real behavior, not assumed from
    its docs, which do not state the boolean polarity)."""
    torch.manual_seed(0)
    gen = SetGenerator(hr_n_max=16, d_model=8, hidden=16, n_layers=1, n_heads=2)
    gen.eval()
    lr_hits, lr_mask = _random_hits(2, 10, seed=14)

    with torch.no_grad():
        hits1, exist1, chan1 = gen(lr_hits, lr_mask, lr_grid=16, hr_grid=32)

        garbage = lr_hits.clone()
        garbage[~lr_mask] = torch.rand_like(garbage[~lr_mask]) * 999
        hits2, exist2, chan2 = gen(garbage, lr_mask, lr_grid=16, hr_grid=32)

    torch.testing.assert_close(hits1, hits2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(exist1, exist2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(chan1, chan2, atol=1e-4, rtol=1e-4)


def test_generator_gradient_reaches_all_parameters():
    torch.manual_seed(0)
    gen = SetGenerator(hr_n_max=16, d_model=8, hidden=16, n_layers=1, n_heads=2)
    lr_hits, lr_mask = _random_hits(2, 8, seed=8)
    hits, exist_logits, chan_logits = gen(lr_hits, lr_mask, lr_grid=16, hr_grid=32)
    (hits.sum() + exist_logits.sum() + chan_logits.sum()).backward()
    missing = [n for n, p in gen.named_parameters() if p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"


def _distinguishable_lr_batch(b=2, n_hits=6, seed=20):
    """b samples, each with n_hits real hits at a DIFFERENT, deterministic
    set of positions -- used to check query-seeding is genuinely
    input-dependent (Stage B)."""
    lr_hits = torch.zeros(b, n_hits, 4)
    lr_mask = torch.ones(b, n_hits, dtype=torch.bool)
    for i in range(b):
        g = torch.Generator().manual_seed(seed + i)
        lr_hits[i, :, 0] = torch.randint(0, 3, (n_hits,), generator=g).float()
        # Offset each sample's positions into a disjoint range so no two
        # samples can coincidentally share a seed position.
        lr_hits[i, :, 1] = torch.randint(0, 10, (n_hits,), generator=g).float() + i * 20
        lr_hits[i, :, 2] = torch.randint(0, 10, (n_hits,), generator=g).float() + i * 20
        lr_hits[i, :, 3] = torch.rand(n_hits, generator=g) * 10 + 1.0
    return lr_hits, lr_mask


def test_build_queries_differs_across_samples_with_different_lr_input():
    """Stage B regression test: queries must be a function of the ACTUAL LR
    input, not fixed parameters shared identically across the batch (the
    pre-Stage-B behavior: self.queries was one nn.Parameter broadcast to
    every sample)."""
    torch.manual_seed(0)
    gen = SetGenerator(hr_n_max=20, d_model=8, hidden=16, n_layers=1, n_heads=2)
    lr_hits, lr_mask = _distinguishable_lr_batch(b=2, n_hits=6)

    with torch.no_grad():
        queries = gen._build_queries(lr_hits, lr_mask, hr_grid=64)

    assert not torch.allclose(queries[0], queries[1], atol=1e-4), (
        "queries are identical across samples with different LR input -- "
        "query construction is not actually input-dependent"
    )


def test_build_queries_ignores_padded_lr_content():
    """Mirrors test_generator_ignores_padded_lr_content but for query
    construction specifically: padded (mask=False) LR rows must never seed a
    query, regardless of what garbage they hold."""
    torch.manual_seed(0)
    gen = SetGenerator(hr_n_max=16, d_model=8, hidden=16, n_layers=1, n_heads=2)
    lr_hits, lr_mask = _random_hits(2, 10, seed=21)

    with torch.no_grad():
        q1 = gen._build_queries(lr_hits, lr_mask, hr_grid=32)
        garbage = lr_hits.clone()
        garbage[~lr_mask] = torch.rand_like(garbage[~lr_mask]) * 999
        q2 = gen._build_queries(garbage, lr_mask, hr_grid=32)

    torch.testing.assert_close(q1, q2, atol=1e-4, rtol=1e-4)


def test_build_queries_always_exactly_hr_n_max_regardless_of_real_hit_count():
    """Query count must be exactly hr_n_max whether a sample has very few
    real LR hits (near the round-robin-tiling extreme) or close to lr_n_max
    real hits (near no-repeat-needed) -- neither should change hr_n_max's
    meaning as a fixed output budget."""
    torch.manual_seed(0)
    hr_n_max = 24
    gen = SetGenerator(hr_n_max=hr_n_max, d_model=8, hidden=16, n_layers=1, n_heads=2)

    # Very few real hits: 1 hit out of 10 slots -> heavy round-robin repeat.
    few_hits = torch.zeros(1, 10, 4)
    few_hits[0, 0, 1:3] = torch.tensor([3., 7.])
    few_mask = torch.zeros(1, 10, dtype=torch.bool)
    few_mask[0, 0] = True

    # Many real hits: 10 out of 10 slots -> little/no repeat within budget.
    many_hits, many_mask = _random_hits(1, 10, seed=22)
    many_mask[:] = True  # force all real

    for hits, mask in [(few_hits, few_mask), (many_hits, many_mask)]:
        q = gen._build_queries(hits, mask, hr_grid=32)
        assert q.shape == (1, hr_n_max, 8)
        assert torch.isfinite(q).all()


def test_build_queries_handles_zero_real_hit_sample():
    """A fully-padded sample (lr_mask.sum() == 0) has nothing to seed queries
    from; this must produce finite output via a defined fallback, not
    NaN/Inf or a crash, and the full forward pass must complete."""
    torch.manual_seed(0)
    gen = SetGenerator(hr_n_max=12, d_model=8, hidden=16, n_layers=1, n_heads=2)
    lr_hits = torch.rand(1, 6, 4)  # content irrelevant, fully masked below
    lr_mask = torch.zeros(1, 6, dtype=torch.bool)

    q = gen._build_queries(lr_hits, lr_mask, hr_grid=32)
    assert torch.isfinite(q).all()
    assert not torch.isnan(q).any()

    hits, exist_logits, chan_logits = gen(lr_hits, lr_mask, lr_grid=16, hr_grid=32)
    assert torch.isfinite(hits).all()
    assert torch.isfinite(exist_logits).all()
    assert torch.isfinite(chan_logits).all()


def test_build_queries_repeat_perturbation_distinguishes_repeats_of_same_hit():
    """When hr_n_max > n_real_hits, the same LR hit seeds multiple query
    slots (round-robin) -- those repeats must NOT be identical, otherwise the
    decoder receives duplicate queries with no way to specialize each to a
    different nearby HR hit (the "one LR hit splits into several HR hits"
    story from the audit)."""
    torch.manual_seed(0)
    gen = SetGenerator(hr_n_max=12, d_model=8, hidden=16, n_layers=1, n_heads=2)

    lr_hits = torch.zeros(1, 6, 4)
    lr_hits[0, :3, 1] = torch.tensor([1., 5., 10.])
    lr_hits[0, :3, 2] = torch.tensor([1., 5., 10.])
    lr_mask = torch.zeros(1, 6, dtype=torch.bool)
    lr_mask[0, :3] = True

    q = gen._build_queries(lr_hits, lr_mask, hr_grid=32)
    # With 3 real hits and hr_n_max=12, slots 0 and 3 both seed from hit 0
    # (round-robin period 3) but are different repeat rounds.
    assert not torch.allclose(q[0, 0], q[0, 3], atol=1e-4)


def test_decoder_memory_has_real_length_not_a_single_token():
    """The actual regression test for the diagnosed bug, made structural
    rather than output-magnitude-based.

    An earlier version of SetGenerator pooled the whole LR set to one
    (B, d_model) vector and cross-attended to it as a length-1 "sequence." A
    real trained checkpoint on real data was found to ignore its input as a
    result (SR[i] closer to a DIFFERENT sample's HR than its own).

    Testing this via "do maximally-different inputs give different outputs"
    at RANDOM INIT is unreliable: even the length-1-memory architecture
    produces somewhat different outputs per input at initialization (the
    pooled vector itself differs per input, it just carries far less
    information) -- confirmed empirically: an untrained old-style model's
    min pairwise output distance across 4 maximally-different inputs was
    0.114, an untrained new-style model's was 0.275 -- directionally right
    but not a clean pass/fail boundary, since BOTH are comfortably nonzero.
    Random-init magnitude is not the right thing to assert on.

    The real, deterministic, seed-independent structural fact is about
    attention itself: softmax attention over a length-1 memory is
    MATHEMATICALLY GUARANTEED to produce a weight of exactly 1.0 for every
    query regardless of query or memory content (softmax of a single logit is
    always 1) -- so a length-1-memory decoder cannot express a different
    attention PATTERN per input at all, only a different memory VALUE. This
    test asserts the memory this generator's decoder attends to has real
    length (N_lr + 1 for the summary token), which is what makes non-trivial,
    input-varying attention patterns possible in the first place -- a
    necessary condition for the fix, verified directly against the model's
    internals rather than inferred from output statistics.
    """
    torch.manual_seed(0)
    gen = SetGenerator(hr_n_max=40, d_model=16, hidden=32, n_layers=2, n_heads=2)
    gen.eval()

    n_hits = 20
    lr_hits, lr_mask = _random_hits(3, n_hits, seed=15)

    captured = {}
    original_forward = torch.nn.TransformerDecoder.forward

    def spy(self, tgt, memory, **kwargs):
        captured["memory_shape"] = memory.shape
        captured["memory_key_padding_mask"] = kwargs.get("memory_key_padding_mask")
        return original_forward(self, tgt, memory, **kwargs)

    import unittest.mock
    with unittest.mock.patch.object(torch.nn.TransformerDecoder, "forward", spy):
        with torch.no_grad():
            gen(lr_hits, lr_mask, lr_grid=16, hr_grid=32)

    assert "memory_shape" in captured, "decoder was never called"
    b, mem_len, d = captured["memory_shape"]
    assert mem_len > 1, (
        f"decoder memory has length {mem_len} -- a length-1 memory makes "
        f"attention weights trivially 1.0 for every query regardless of "
        f"input, which is exactly the diagnosed bottleneck"
    )
    assert mem_len == n_hits + 1, (
        f"expected N_lr + 1 (per-point sequence + one global summary token) "
        f"= {n_hits + 1}, got {mem_len}"
    )
    assert captured["memory_key_padding_mask"] is not None, (
        "memory_key_padding_mask was not passed -- padded LR rows would be "
        "attended to as if they were real hits"
    )


def test_discriminator_output_shape_and_gradient():
    torch.manual_seed(0)
    disc = SetDiscriminator(d_model=8, hidden=16)
    lr_hits, lr_mask = _random_hits(3, 10, seed=9)
    hr_hits, hr_mask = _random_hits(3, 20, seed=10)
    score = disc(lr_hits, lr_mask, 16, hr_hits, hr_mask, 32)
    assert score.shape == (3,)
    score.sum().backward()
    missing = [n for n, p in disc.named_parameters() if p.grad is None]
    assert not missing


def test_generator_to_discriminator_is_wired_without_reshape():
    """The exact bug found during implementation: generator output must be
    directly consumable by the discriminator/encoder (same column layout)."""
    torch.manual_seed(0)
    gen = SetGenerator(hr_n_max=16, d_model=8, hidden=16, n_layers=1, n_heads=2)
    disc = SetDiscriminator(d_model=8, hidden=16)
    lr_hits, lr_mask = _random_hits(2, 8, seed=11)
    hits, _, _ = gen(lr_hits, lr_mask, lr_grid=16, hr_grid=32)
    hr_mask = torch.ones(2, 16, dtype=torch.bool)
    score = disc(lr_hits, lr_mask, 16, hits, hr_mask, 32)  # must not raise
    assert score.shape == (2,)


# --------------------------------------------------------------------------- #
# losses.py: masked_chamfer_loss / existence_loss / channel_loss
# --------------------------------------------------------------------------- #
def _random_pred_hits(b, n, seed=0):
    """Every slot has a real (non-padded) coordinate/energy, matching what
    SetGenerator actually outputs -- unlike _random_hits, which bakes in
    masked/zeroed rows and is unsuitable as a `pred_hits` fixture for exactly
    the reason explained in masked_chamfer_loss's docstring."""
    g = torch.Generator().manual_seed(seed)
    channel = torch.randint(0, 3, (b, n), generator=g).float()
    x = torch.randint(0, 16, (b, n), generator=g).float()
    y = torch.randint(0, 16, (b, n), generator=g).float()
    energy = torch.rand(b, n, generator=g) * 10 + 0.1  # never exactly 0
    return torch.stack([channel, x, y, energy], dim=-1)


def test_masked_chamfer_zero_when_pred_matches_target_exactly():
    pred = _random_pred_hits(2, 10, seed=12)
    mask = torch.ones(2, 10, dtype=torch.bool)  # every target slot real, matching pred 1:1
    assert masked_chamfer_loss(pred, pred, mask).item() == pytest.approx(0.0, abs=1e-4)


def test_masked_chamfer_ignores_padded_target_rows():
    """Padding a target with extra all-zero rows must not change the loss --
    zero-coordinate padding is a real pixel, so an unmasked comparison would
    be pulled toward the origin."""
    pred = torch.zeros(1, 2, 4)
    pred[0, 0] = torch.tensor([0., 5., 5., 1.])
    pred[0, 1] = torch.tensor([0., 5., 5., 1.])

    target_tight = torch.zeros(1, 2, 4)
    target_tight[0, 0] = torch.tensor([0., 5., 5., 1.])
    target_tight[0, 1] = torch.tensor([0., 5., 5., 1.])
    mask_tight = torch.ones(1, 2, dtype=torch.bool)

    target_padded = torch.zeros(1, 6, 4)
    target_padded[0, 0] = torch.tensor([0., 5., 5., 1.])
    target_padded[0, 1] = torch.tensor([0., 5., 5., 1.])
    mask_padded = torch.tensor([[True, True, False, False, False, False]])

    v_tight = masked_chamfer_loss(pred, target_tight, mask_tight)
    v_padded = masked_chamfer_loss(pred, target_padded, mask_padded)
    torch.testing.assert_close(v_tight, v_padded, atol=1e-5, rtol=1e-5)


def test_masked_chamfer_gradient_flows_to_pred():
    pred = torch.rand(2, 5, 4, requires_grad=True)
    target, target_mask = _random_hits(2, 5, seed=13)
    masked_chamfer_loss(pred, target, target_mask).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_masked_chamfer_rejects_batch_mismatch():
    a = torch.zeros(2, 3, 4)
    b = torch.zeros(3, 3, 4)
    with pytest.raises(ValueError):
        masked_chamfer_loss(a, b, torch.ones(3, 3, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# The peak-smearing bug: found in a REAL trained checkpoint (predicted top-5
# energies 10.9/9.2/7.4/6.2/6.1 vs HR's 27.6/16.6/6.2/6.0/4.0 -- flat where
# HR is peaked, despite near-identical total energy). Root cause: the
# position-only Chamfer distance has zero gradient pressure on energy VALUES
# -- two hits at the same (x,y) score identically regardless of energy gap.
# These tests pin the fix and would fail again if the energy-value term were
# ever dropped or defaulted to 0.
# --------------------------------------------------------------------------- #
def _peaked_case():
    """One bright peak + one dim hit -- the two-hit toy case that reproduces
    the checkpoint's failure at the smallest possible scale."""
    target = torch.tensor([[[0., 10., 10., 30.0], [0., 50., 50., 1.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    pred_sharp = torch.tensor([[[0., 10., 10., 30.0], [0., 52., 52., 1.0]]])
    pred_smeared = torch.tensor([[[0., 10., 10., 10.0], [0., 50., 50., 1.0]]])
    return target, mask, pred_sharp, pred_smeared


def test_masked_chamfer_position_only_prefers_smeared_reconstruction():
    """Documents the bug BEFORE the fix: with lambda_energy_value=0 and no
    energy weighting, the loss actually preferred the smeared checkpoint's
    failure mode. This is the regression the other tests below guard against
    -- if it ever stops holding, masked_chamfer_loss's defaults changed and
    the fix may have been silently reverted."""
    target, mask, pred_sharp, pred_smeared = _peaked_case()
    c_sharp = masked_chamfer_loss(pred_sharp, target, mask, energy_weighted=False, lambda_energy_value=0.0)
    c_smeared = masked_chamfer_loss(pred_smeared, target, mask, energy_weighted=False, lambda_energy_value=0.0)
    assert c_smeared < c_sharp  # the bug, preserved as documentation


def test_masked_chamfer_default_prefers_sharp_reconstruction():
    target, mask, pred_sharp, pred_smeared = _peaked_case()
    c_sharp = masked_chamfer_loss(pred_sharp, target, mask)  # defaults
    c_smeared = masked_chamfer_loss(pred_smeared, target, mask)
    assert c_sharp < c_smeared


def test_masked_chamfer_energy_weighting_alone_is_not_sufficient():
    """Energy-weighting the TARGET term (which hit matters more) is not the
    same as penalizing a wrong VALUE at the matched hit -- confirm both are
    needed, not just one."""
    target, mask, pred_sharp, pred_smeared = _peaked_case()
    c_sharp = masked_chamfer_loss(pred_sharp, target, mask, energy_weighted=True, lambda_energy_value=0.0)
    c_smeared = masked_chamfer_loss(pred_smeared, target, mask, energy_weighted=True, lambda_energy_value=0.0)
    assert c_smeared < c_sharp  # still wrong without the energy-value term


def test_masked_sinkhorn_mass_normalization_alone_misses_the_bug():
    """Sinkhorn's transport masses are normalized to sum=1 PER SAMPLE, so an
    absolute energy drop that preserves relative mass fractions is invisible
    without the energy-value cost term -- documents why lambda_energy_value
    isn't optional here despite defaulting the same as Chamfer's."""
    target, mask, pred_sharp, pred_smeared = _peaked_case()
    s_sharp = masked_sinkhorn_loss(pred_sharp, target, mask, eps=1.0, iters=100, lambda_energy_value=0.0)
    s_smeared = masked_sinkhorn_loss(pred_smeared, target, mask, eps=1.0, iters=100, lambda_energy_value=0.0)
    assert s_smeared < s_sharp  # the bug, preserved as documentation


def test_masked_sinkhorn_default_prefers_sharp_reconstruction():
    target, mask, pred_sharp, pred_smeared = _peaked_case()
    s_sharp = masked_sinkhorn_loss(pred_sharp, target, mask, eps=1.0, iters=100)
    s_smeared = masked_sinkhorn_loss(pred_smeared, target, mask, eps=1.0, iters=100)
    assert s_sharp < s_smeared


def test_masked_sinkhorn_padding_does_not_change_the_value():
    target_tight = torch.tensor([[[0., 10., 10., 30.0], [0., 50., 50., 1.0]]])
    mask_tight = torch.ones(1, 2, dtype=torch.bool)
    target_padded = torch.tensor([[[0., 10., 10., 30.0], [0., 50., 50., 1.0], [0., 0., 0., 0.]]])
    mask_padded = torch.tensor([[True, True, False]])
    pred = torch.tensor([[[0., 10., 10., 30.0], [0., 52., 52., 1.0]]])

    v_tight = masked_sinkhorn_loss(pred, target_tight, mask_tight, eps=1.0, iters=100)
    v_padded = masked_sinkhorn_loss(pred, target_padded, mask_padded, eps=1.0, iters=100)
    torch.testing.assert_close(v_tight, v_padded, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(v_padded)


def test_masked_sinkhorn_gradient_flows_to_energy_column():
    pred = torch.rand(2, 5, 4, requires_grad=True)
    target = torch.rand(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    masked_sinkhorn_loss(pred, target, mask, eps=1.0, iters=30).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad[..., 3] != 0).any()  # energy column receives gradient


def test_masked_sinkhorn_rejects_bad_args():
    a = torch.zeros(2, 3, 4)
    b = torch.zeros(3, 3, 4)
    with pytest.raises(ValueError):
        masked_sinkhorn_loss(a, b, torch.ones(3, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        masked_sinkhorn_loss(a, a, torch.ones(2, 3, dtype=torch.bool), eps=0.0)


def test_existence_loss_rewards_correct_predictions():
    mask = torch.tensor([[True, False, True]])
    correct_logits = torch.tensor([[10.0, -10.0, 10.0]])
    wrong_logits = torch.tensor([[-10.0, 10.0, -10.0]])
    assert existence_loss(correct_logits, mask).item() < existence_loss(wrong_logits, mask).item()


def test_existence_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        existence_loss(torch.zeros(2, 5), torch.zeros(2, 6, dtype=torch.bool))


def test_channel_loss_zero_when_logits_confidently_correct():
    target_channel = torch.tensor([[0.0, 2.0, 1.0]])
    mask = torch.tensor([[True, True, True]])
    logits = torch.full((1, 3, 3), -10.0)
    logits[0, 0, 0] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 1] = 10.0
    assert channel_loss(logits, target_channel, mask).item() == pytest.approx(0.0, abs=1e-3)


def test_channel_loss_ignores_padded_rows():
    target_channel = torch.tensor([[0.0, 2.0]])
    mask_all = torch.tensor([[True, True]])
    mask_partial = torch.tensor([[True, False]])
    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 0] = 10.0  # correct for row 0
    logits[0, 1, 1] = 10.0  # WRONG for row 1 (target is class 2)
    loss_all = channel_loss(logits, target_channel, mask_all)
    loss_partial = channel_loss(logits, target_channel, mask_partial)
    assert loss_partial.item() < loss_all.item()  # masking out the wrong row helps
    assert loss_partial.item() == pytest.approx(0.0, abs=1e-3)  # row 0 alone is correct


# Stage C: assignment must follow geometry, not target storage order.
from multiscale_sr.pointcloud.matching import hungarian_assignment, matched_supervision


def test_hungarian_recovers_shuffled_geometry_without_double_matching():
    pred = torch.tensor([[[0., 0., 0., 1.], [1., 10., 10., 1.], [2., 30., 30., 1.]]])
    target = pred[:, [1, 0, 2]]
    mask = torch.tensor([[True, True, False]])
    assignment = hungarian_assignment(pred, target, mask)
    assert assignment.tolist() == [[1, 0, -1]]
    real = assignment[assignment >= 0]
    assert real.unique().numel() == real.numel() == 2
    assert not torch.equal(assignment, torch.tensor([[0, 1, -1]]))


def test_matched_losses_are_shuffle_invariant_and_old_index_losses_are_not():
    pred = torch.tensor([[[0., 0., 0., 1.], [1., 10., 10., 1.], [2., 30., 30., 1.]]])
    target = pred.clone()
    mask = torch.tensor([[True, True, False]])
    existence = torch.tensor([[5., 2., -4.]], requires_grad=True)
    logits = torch.tensor([[[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]]], requires_grad=True)
    permutation = torch.tensor([2, 0, 1])
    shuffled, smask = target[:, permutation], mask[:, permutation]
    c1, m1 = matched_supervision(pred, target, mask)
    c2, m2 = matched_supervision(pred, shuffled, smask)
    assert torch.equal(c1, c2) and torch.equal(m1, m2)
    assert torch.equal(existence_loss(existence, m1), existence_loss(existence, m2))
    assert torch.equal(channel_loss(logits, c1, m1), channel_loss(logits, c2, m2))
    # Regression pins: the old call sites fail both invariance properties.
    assert not torch.isclose(existence_loss(existence, mask), existence_loss(existence, smask))
    assert not torch.isclose(channel_loss(logits, target[..., 0], mask), channel_loss(logits, shuffled[..., 0], smask))
    (existence_loss(existence, m1) + channel_loss(logits, c1, m1)).backward()
    assert torch.isfinite(existence.grad).all() and torch.isfinite(logits.grad).all()
    assert logits.grad[~m1].abs().sum() == 0


def test_hungarian_coincident_channels_ties_are_shuffle_invariant():
    pred = torch.tensor([[[0., 1., 1., 1.], [0., 1., 1., 1.], [0., 1., 1., 1.]]])
    target = pred.clone()
    target[0, :, 0] = torch.tensor([2., 0., 1.])
    mask = torch.ones(1, 3, dtype=torch.bool)
    c1, m1 = matched_supervision(pred, target, mask)
    c2, m2 = matched_supervision(pred, target[:, [2, 0, 1]], mask)
    assert torch.equal(c1, c2) and torch.equal(m1, m2)


@pytest.mark.parametrize('n_pred,n_target,n_real', [(4, 3, 0), (2, 4, 4), (4, 0, 0), (0, 3, 3)])
def test_hungarian_empty_and_excess_targets(n_pred, n_target, n_real):
    pred = torch.zeros(1, n_pred, 4, requires_grad=True)
    target = torch.zeros(1, n_target, 4)
    mask = torch.arange(n_target).unsqueeze(0) < n_real
    assignment = hungarian_assignment(pred, target, mask)
    assert assignment.shape == (1, n_pred)
    assert (assignment >= 0).sum() == min(n_pred, n_real)
    real = assignment[assignment >= 0]
    assert real.unique().numel() == real.numel()
    channels, matched = matched_supervision(pred, target, mask)
    assert channels.shape == matched.shape == (1, n_pred)
    assert not channels.requires_grad


def test_training_assignment_default_and_ablation():
    import train_pointcloud as training
    args = training.resolve_args(['--data-dir', '/tmp/test', '--scale', '64'])
    assert args.target_assignment == 'hungarian'
    pred = torch.tensor([[[0., 0., 0., 1.], [1., 10., 10., 1.]]])
    target = pred.flip(1)
    mask = torch.ones(1, 2, dtype=torch.bool)
    c, m = training._supervision_targets(pred, target, mask, args)
    assert torch.equal(c, pred[..., 0])
    args.target_assignment = 'index'
    c, m = training._supervision_targets(pred, target, mask, args)
    assert torch.equal(c, target[..., 0])


def test_validation_uses_matched_targets_when_target_rows_move(monkeypatch):
    import train_pointcloud as training
    args = training.resolve_args(['--data-dir', '/tmp/test', '--scale', '4', '--hr-size', '4'])
    pred = torch.tensor([[[0., 0., 0., 1.], [1., 2., 2., 1.], [2., 3., 3., 1.]]])
    mask = torch.tensor([[True, True, False]])
    exist = torch.tensor([[5., 2., -4.]])
    logits = torch.tensor([[[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]]])
    class FixedGenerator(torch.nn.Module):
        def forward(self, *unused):
            return pred, exist, logits
    batch = {'lr': torch.ones(1, 3, 4, 4), 'hr': torch.ones(1, 3, 4, 4)}
    monkeypatch.setattr(training, '_to_hits', lambda *unused: (pred, mask, pred, mask))
    before = training.evaluate(FixedGenerator(), [batch], torch.device('cpu'), args)
    perm = [2, 0, 1]
    monkeypatch.setattr(training, '_to_hits', lambda *unused: (pred, mask, pred[:, perm], mask[:, perm]))
    after = training.evaluate(FixedGenerator(), [batch], torch.device('cpu'), args)
    assert before['existence'] == after['existence']
    assert before['channel'] == after['channel']
    assert before['chamfer'] == pytest.approx(after['chamfer'])


def test_parallel_hungarian_is_identical_to_serial():
    rng = torch.Generator().manual_seed(9)
    pred = torch.rand(8, 24, 4, generator=rng)
    target = torch.rand(8, 19, 4, generator=rng)
    mask = torch.rand(8, 19, generator=rng) > 0.3
    mask[0] = False
    assert torch.equal(hungarian_assignment(pred, target, mask, workers=1),
                       hungarian_assignment(pred, target, mask, workers=4))
    with pytest.raises(ValueError, match="workers"):
        hungarian_assignment(pred, target, mask, workers=0)
