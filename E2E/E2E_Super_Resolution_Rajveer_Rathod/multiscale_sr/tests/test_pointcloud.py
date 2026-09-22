import numpy as np
from pathlib import Path
import pytest
import torch

from multiscale_sr.data.multiscale import downscale_hr, pad_to_size
from multiscale_sr.pointcloud.hits import (
    COL_BATCH,
    COL_CHANNEL,
    COL_ENERGY,
    COL_X,
    COL_Y,
    dense_to_hits,
    hit_counts,
    hits_to_dense,
    topk_hits,
)
from multiscale_sr.pointcloud.losses import (
    chamfer_loss,
    energy_conservation_loss,
    sinkhorn_loss,
)


def sparse_batch(b=3, c=3, h=16, w=16, density=0.03, seed=0):
    """A batch shaped like this data: mostly zeros, positive deposits."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(b, c, h, w)
    mask = torch.rand(b, c, h, w, generator=g) < density
    x[mask] = torch.rand(int(mask.sum()), generator=g) * 100.0 + 1e-3
    return x


def test_round_trip_is_exact():
    """hits_to_dense(dense_to_hits(x)) == x for threshold=0."""
    x = sparse_batch()
    hits = dense_to_hits(x, threshold=0.0)
    back = hits_to_dense(hits, x.shape)
    torch.testing.assert_close(back, x, rtol=0, atol=0)


def test_round_trip_exact_on_real_shape():
    """Same guarantee at the production 125->128 padded shape."""
    x = pad_to_size(sparse_batch(b=2, h=125, w=125, density=0.03, seed=7), 128)
    hits = dense_to_hits(x)
    torch.testing.assert_close(hits_to_dense(hits, x.shape), x, rtol=0, atol=0)


def test_energy_is_conserved_by_conversion():
    x = sparse_batch()
    hits = dense_to_hits(x)
    torch.testing.assert_close(hits[:, COL_ENERGY].sum(), x.sum())


def test_columns_carry_the_documented_meaning():
    x = torch.zeros(2, 3, 5, 7)
    x[1, 2, 3, 4] = 42.0  # batch 1, channel 2, y=3, x=4
    hits = dense_to_hits(x)
    assert hits.shape == (1, 5)
    assert hits[0, COL_BATCH].item() == 1
    assert hits[0, COL_CHANNEL].item() == 2
    assert hits[0, COL_X].item() == 4
    assert hits[0, COL_Y].item() == 3
    assert hits[0, COL_ENERGY].item() == 42.0


def test_threshold_drops_entries_below_it():
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, 0] = 0.5
    x[0, 0, 1, 1] = 5.0
    assert dense_to_hits(x, threshold=0.0).shape[0] == 2
    assert dense_to_hits(x, threshold=1.0).shape[0] == 1
    # Above every value -> empty, and empty must round-trip to all zeros.
    empty = dense_to_hits(x, threshold=100.0)
    assert empty.shape == (0, 5)
    torch.testing.assert_close(hits_to_dense(empty, x.shape), torch.zeros_like(x))


def test_negative_values_are_dropped_at_default_threshold():
    """Normalized tensors have a negative floor; document that it is discarded."""
    x = torch.full((1, 1, 3, 3), -0.7)
    x[0, 0, 1, 1] = 2.0
    hits = dense_to_hits(x)
    assert hits.shape[0] == 1
    assert hits[0, COL_ENERGY].item() == 2.0


def test_duplicate_hits_sum_rather_than_overwrite():
    """A generator may emit two hits in one cell; energy must not vanish."""
    hits = torch.tensor([
        [0.0, 0.0, 2.0, 1.0, 3.0],
        [0.0, 0.0, 2.0, 1.0, 4.0],
    ])
    out = hits_to_dense(hits, (1, 1, 4, 4))
    assert out[0, 0, 1, 2].item() == 7.0
    assert out.sum().item() == 7.0


def test_out_of_range_hits_raise():
    good = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
    hits_to_dense(good, (1, 1, 4, 4))  # does not raise
    for bad in (
        torch.tensor([[0.0, 0.0, 9.0, 1.0, 1.0]]),   # x outside
        torch.tensor([[0.0, 0.0, 1.0, 9.0, 1.0]]),   # y outside
        torch.tensor([[5.0, 0.0, 1.0, 1.0, 1.0]]),   # batch outside
        torch.tensor([[0.0, 7.0, 1.0, 1.0, 1.0]]),   # channel outside
    ):
        with pytest.raises(IndexError):
            hits_to_dense(bad, (1, 1, 4, 4))


def test_hit_counts_includes_trailing_empty_samples():
    x = torch.zeros(4, 1, 3, 3)
    x[0, 0, 0, 0] = 1.0
    x[0, 0, 1, 1] = 1.0
    x[1, 0, 2, 2] = 1.0
    counts = hit_counts(dense_to_hits(x), batch_size=4)
    assert counts.tolist() == [2, 1, 0, 0]
    assert hit_counts(dense_to_hits(torch.zeros(3, 1, 2, 2)), 3).tolist() == [0, 0, 0]


def test_topk_energies_are_differentiable_and_coords_are_not():
    x = sparse_batch(b=2, c=1, h=8, w=8, density=0.2, seed=3).requires_grad_(True)
    energies, coords = topk_hits(x, k=5)
    assert energies.shape == (2, 5)
    assert coords.shape == (2, 5, 3)
    assert energies.requires_grad
    assert not coords.requires_grad
    energies.sum().backward()
    # Exactly the selected pixels receive gradient.
    assert x.grad is not None
    assert int((x.grad != 0).sum()) == 10


def test_topk_selects_the_brightest_and_pads_when_short():
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, 0] = 9.0
    x[0, 0, 3, 3] = 5.0
    energies, coords = topk_hits(x, k=4)
    assert energies[0, :2].tolist() == [9.0, 5.0]
    # Only two positive pixels; the rest are inert zero-energy entries.
    assert energies[0, 2:].abs().sum().item() == 0.0
    assert coords[0, 0].tolist() == [0.0, 0.0, 0.0]   # (channel, x, y)
    assert coords[0, 1].tolist() == [0.0, 3.0, 3.0]
    # k larger than the whole grid still returns exactly k entries.
    e2, c2 = topk_hits(x, k=32)
    assert e2.shape == (1, 32) and c2.shape == (1, 32, 3)


def test_topk_is_consistent_with_dense_to_hits():
    """Top-K energies must be the K largest entries of the full hit list."""
    x = sparse_batch(b=1, c=3, h=12, w=12, density=0.1, seed=11)
    hits = dense_to_hits(x)
    expected = torch.sort(hits[:, COL_ENERGY], descending=True).values[:6]
    energies, _ = topk_hits(x, k=6)
    torch.testing.assert_close(energies[0], expected)


def test_downsampling_reduces_hit_count_on_sparse_data():
    """Sanity check for the audit's central question, on synthetic data."""
    x = pad_to_size(sparse_batch(b=2, h=125, w=125, density=0.03, seed=5), 128)
    hr_hits = hit_counts(dense_to_hits(x), 2).sum().item()
    lr_hits = hit_counts(dense_to_hits(downscale_hr(x, 16)), 2).sum().item()
    assert lr_hits < hr_hits  # 16x16 grid cannot hold as many entries


def test_sinkhorn_converges_and_matches_exact_ot_on_a_known_case():
    """Regression: accumulating the potentials instead of replacing them
    diverged to Inf within ~10 iterations. Also pins the value against a case
    with a known optimal plan."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pc_audit", Path(__file__).resolve().parent.parent / "scripts" / "pointcloud_audit.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    # Two hits each, equal mass, optimal matching pairs (3,3)->(4,4) and
    # (8,8)->(9,9): both distances are 2 in squared pixels, so exact OT = 2.0.
    a = torch.zeros(1, 1, 16, 16)
    a[0, 0, 3, 3] = 5.0
    a[0, 0, 8, 8] = 5.0
    b = torch.zeros(1, 1, 16, 16)
    b[0, 0, 4, 4] = 5.0
    b[0, 0, 9, 9] = 5.0
    ha, hb = dense_to_hits(a), dense_to_hits(b)

    value = audit.sinkhorn_xy(ha, hb, eps=0.05, iters=200)
    assert np.isfinite(value)
    assert value == pytest.approx(2.0, abs=0.1)

    # Identical sets: zero cost regardless of eps.
    assert audit.sinkhorn_xy(ha, ha, eps=0.05, iters=200) == pytest.approx(0.0, abs=1e-3)

    # Chamfer on the same pair: each hit is sqrt(2) from its match.
    assert audit.chamfer_xy(ha, hb) == pytest.approx(2 * 2 ** 0.5, abs=1e-4)


def test_chamfer_zero_for_identical_images():
    x = sparse_batch(b=2, c=1, h=12, w=12, density=0.1, seed=9)
    assert chamfer_loss(x, x, topk=10).item() == pytest.approx(0.0, abs=1e-5)


def test_chamfer_positive_and_symmetric_for_shifted_images():
    a = torch.zeros(1, 1, 16, 16)
    a[0, 0, 4, 4] = 3.0
    b = torch.zeros(1, 1, 16, 16)
    b[0, 0, 7, 4] = 3.0  # shifted 3px in x
    d_ab = chamfer_loss(a, b, topk=1).item()
    d_ba = chamfer_loss(b, a, topk=1).item()
    assert d_ab > 0
    assert d_ab == pytest.approx(d_ba, rel=1e-5)  # bidirectional -> order-invariant


def test_chamfer_gradient_reaches_only_selected_pixels():
    torch.manual_seed(2)
    pred = (torch.rand(2, 3, 16, 16) * 5).requires_grad_(True)
    target = torch.rand(2, 3, 16, 16) * 5
    chamfer_loss(pred, target, topk=6).backward()
    assert pred.grad is not None
    assert int((pred.grad != 0).sum()) == 2 * 6  # exactly K hits per sample


def test_chamfer_chunking_does_not_change_the_value():
    x = sparse_batch(b=4, c=1, h=10, w=10, density=0.2, seed=4)
    y = sparse_batch(b=4, c=1, h=10, w=10, density=0.2, seed=5)
    whole = chamfer_loss(x, y, topk=6, chunk=4)
    chunked = chamfer_loss(x, y, topk=6, chunk=1)
    torch.testing.assert_close(whole, chunked)


def test_chamfer_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        chamfer_loss(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4))
    with pytest.raises(ValueError):
        chamfer_loss(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8), chunk=0)


def test_sinkhorn_self_transport_shrinks_toward_zero_as_eps_shrinks():
    """Self-transport should approach 0 as eps -> 0 (entropic OT's exact-OT
    limit). It is NOT expected to be ~0 at a fixed eps regardless of point
    spacing: eps=1.0 is a genuine excess cost, not truncation error, whenever
    the point set's own scale (min pairwise distance) is comparable to eps --
    confirmed by iteration count alone (10 vs 1000) not moving the value at
    fixed eps, while shrinking eps does. See the well-separated case below for
    where eps=1.0 IS negligible relative to spacing."""
    x = sparse_batch(b=2, c=1, h=12, w=12, density=0.1, seed=9)
    losses = [sinkhorn_loss(x, x, topk=10, eps=e, iters=200).item()
              for e in (2.0, 1.0, 0.5, 0.1)]
    assert losses == sorted(losses, reverse=True)  # monotonically shrinks
    assert losses[-1] < 1e-3  # eps=0.1 is negligible next to ~1-12px spacing

    # Well-separated points (min spacing ~8px): eps=1.0 is already negligible,
    # and iters=10 already agrees with iters=200 -- true convergence, not a
    # coincidence of scale.
    a = torch.zeros(1, 1, 16, 16)
    a[0, 0, 3, 3] = 1.0
    a[0, 0, 11, 11] = 1.0
    v10 = sinkhorn_loss(a, a, topk=2, eps=1.0, iters=10).item()
    v100 = sinkhorn_loss(a, a, topk=2, eps=1.0, iters=100).item()
    assert v10 == pytest.approx(v100, abs=1e-6)
    assert v100 == pytest.approx(0.0, abs=1e-4)


def test_sinkhorn_is_finite_and_gradient_reaches_selected_pixels():
    torch.manual_seed(3)
    pred = (torch.rand(2, 3, 16, 16) * 5).requires_grad_(True)
    target = torch.rand(2, 3, 16, 16) * 5
    loss = sinkhorn_loss(pred, target, topk=6, eps=4.0, iters=30)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert int((pred.grad != 0).sum()) == 2 * 6


def test_sinkhorn_rejects_bad_args():
    x = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError):
        sinkhorn_loss(x, x, eps=0.0)
    with pytest.raises(ValueError):
        sinkhorn_loss(x, x, chunk=0)
    with pytest.raises(ValueError):
        sinkhorn_loss(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4))


def test_energy_conservation_matches_physics_loss_ratio():
    from multiscale_sr.engine import physics_loss_ratio

    torch.manual_seed(6)
    pred = torch.rand(3, 3, 10, 10) * 5
    target = torch.rand(3, 3, 10, 10) * 5
    torch.testing.assert_close(
        energy_conservation_loss(pred, target), physics_loss_ratio(pred, target)
    )


def test_rejects_wrong_rank():
    with pytest.raises(ValueError):
        dense_to_hits(torch.zeros(3, 4, 4))
    with pytest.raises(ValueError):
        topk_hits(torch.zeros(3, 4, 4), k=2)
    with pytest.raises(ValueError):
        topk_hits(torch.zeros(1, 1, 4, 4), k=0)
