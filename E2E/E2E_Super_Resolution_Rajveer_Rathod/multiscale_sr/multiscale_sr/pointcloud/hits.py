"""Dense image <-> hit-list conversion.

The foundation of the point-cloud path: everything downstream (datasets,
losses, models) is defined in terms of the hit layout fixed here.

Layout
------
A hit list is a single ``(N, 5)`` float tensor whose columns are::

    0: batch_idx   which sample in the batch this hit belongs to
    1: channel     which calorimeter channel (0..C-1)
    2: x           column index (width axis)
    3: y           row index (height axis)
    4: energy      the stored value at that pixel

Columns 0-3 are integer-valued but held as float so one tensor carries the
whole record; cast with ``.long()`` at indexing sites. The rows are ordered by
``torch.nonzero``'s row-major traversal (batch, channel, y, x), which makes
``dense_to_hits`` deterministic — but nothing downstream may *rely* on that
order, since a point cloud is by definition unordered.

Energies are stored exactly as they appear in the source tensor. This module
performs no normalization, no thresholding beyond the one the caller asks for,
and no coordinate rescaling, so that ``hits_to_dense`` is an exact inverse.
"""
from __future__ import annotations

import torch
from torch import Tensor

# Column positions, named so downstream code does not hard-code integers.
COL_BATCH = 0
COL_CHANNEL = 1
COL_X = 2
COL_Y = 3
COL_ENERGY = 4
N_COLS = 5


def dense_to_hits(img: Tensor, threshold: float = 0.0) -> Tensor:
    """Convert a dense ``(B,C,H,W)`` batch to a ``(N,5)`` hit list.

    Keeps entries with ``value > threshold``. The default ``0.0`` keeps every
    strictly-positive pixel and drops exact zeros — which, on this data, is
    ~97% of the grid. Note this also drops any *negative* values: the raw
    parquet energies are non-negative, but a normalized tensor has a negative
    background floor, so call this on raw (denormalized) energy unless you
    specifically intend to discard the floor (``weighted_l1_loss`` makes the
    same choice via ``clamp_min(0.0)``).

    Returns an empty ``(0,5)`` tensor when nothing passes the threshold, rather
    than raising — an all-empty jet is degenerate data, not a programming
    error, and the caller decides what to do about it.
    """
    if img.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got shape {tuple(img.shape)}")

    mask = img > threshold
    idx = mask.nonzero(as_tuple=False)  # (N,4): batch, channel, y, x
    if idx.numel() == 0:
        return img.new_zeros((0, N_COLS))

    energies = img[mask]  # same row-major order as idx
    hits = img.new_empty((idx.shape[0], N_COLS))
    hits[:, COL_BATCH] = idx[:, 0]
    hits[:, COL_CHANNEL] = idx[:, 1]
    hits[:, COL_X] = idx[:, 3]  # width axis
    hits[:, COL_Y] = idx[:, 2]  # height axis
    hits[:, COL_ENERGY] = energies
    return hits


def hits_to_dense(hits: Tensor, shape: tuple[int, int, int, int]) -> Tensor:
    """Rasterize a ``(N,5)`` hit list back to a dense ``(B,C,H,W)`` tensor.

    Exact inverse of ``dense_to_hits`` for ``threshold=0.0``: pixels with no
    hit are zero, and every hit's energy is written unmodified.

    Duplicate ``(batch, channel, x, y)`` entries are **summed**, not
    overwritten. That is what makes this usable for a generated point cloud,
    where a model may legitimately emit two hits in the same cell, and it keeps
    total energy invariant under rasterization. For a hit list that came from
    ``dense_to_hits`` there are no duplicates, so the round trip is unaffected.

    Coordinates must already be integral and inside the grid; this raises
    rather than silently clamping, because an out-of-range hit from a generator
    is a modelling bug whose energy would otherwise vanish without trace.
    """
    b, c, h, w = shape
    out = hits.new_zeros(shape) if hits.numel() else torch.zeros(
        shape, dtype=hits.dtype, device=hits.device
    )
    if hits.numel() == 0:
        return out
    if hits.dim() != 2 or hits.shape[1] != N_COLS:
        raise ValueError(f"Expected (N,{N_COLS}) hit list, got {tuple(hits.shape)}")

    bi = hits[:, COL_BATCH].round().long()
    ci = hits[:, COL_CHANNEL].round().long()
    xi = hits[:, COL_X].round().long()
    yi = hits[:, COL_Y].round().long()

    if (bi < 0).any() or (bi >= b).any() or (ci < 0).any() or (ci >= c).any():
        raise IndexError("hit batch/channel index outside the requested shape")
    if (xi < 0).any() or (xi >= w).any() or (yi < 0).any() or (yi >= h).any():
        raise IndexError("hit x/y coordinate outside the requested grid")

    # index_put_ with accumulate=True sums duplicates; plain assignment would
    # keep only the last write and silently lose energy.
    out.index_put_((bi, ci, yi, xi), hits[:, COL_ENERGY], accumulate=True)
    return out


def hit_counts(hits: Tensor, batch_size: int) -> Tensor:
    """Number of hits per sample, as a ``(batch_size,)`` long tensor.

    Samples with no hits get 0 — ``bincount`` alone would return a short
    tensor when the last samples are empty, so the length is pinned here.
    """
    if hits.numel() == 0:
        return torch.zeros(batch_size, dtype=torch.long, device=hits.device)
    return torch.bincount(
        hits[:, COL_BATCH].round().long(), minlength=batch_size
    )[:batch_size]


def topk_hits(img: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Differentiable top-K extraction per sample, for losses on a dense model.

    Returns ``(energies, coords)``:

    - ``energies``: ``(B, k)`` — **carries gradient back to** ``img``, which is
      what lets a set-based loss train the existing CNN generator without
      replacing it (the Stage 2 experiment).
    - ``coords``: ``(B, k, 3)`` of ``(channel, x, y)``, integral and detached.
      Positions are argmax indices; they are piecewise-constant in the input
      and have no useful gradient, so they are deliberately not differentiable.
      A loss that needs to *move* energy must do so through the energies and
      the ground metric, exactly as ``semd_loss`` does.

    Flattens across channels so K is a budget over the whole jet rather than
    per channel — matching ``semd_images``' top-K convention.

    When a sample has fewer than ``k`` positive pixels the remainder are real
    (zero-energy) entries rather than padding: they contribute nothing to an
    energy-weighted loss, so no mask is needed, and the shape stays static,
    which keeps this usable under ``torch.compile`` and avoids a per-sample
    branch.
    """
    if img.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got shape {tuple(img.shape)}")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")

    b, c, h, w = img.shape
    flat = img.reshape(b, -1)
    k_eff = min(k, flat.shape[1])
    energies, flat_idx = torch.topk(flat, k_eff, dim=1)

    # Unravel the flat index back to (channel, y, x) on the detached path.
    idx = flat_idx.detach()
    ci = idx // (h * w)
    rem = idx % (h * w)
    yi = rem // w
    xi = rem % w
    coords = torch.stack([ci, xi, yi], dim=-1).to(img.dtype)

    if k_eff < k:  # pad to the requested K with inert zero-energy hits
        pad = k - k_eff
        energies = torch.cat([energies, energies.new_zeros((b, pad))], dim=1)
        coords = torch.cat([coords, coords.new_zeros((b, pad, 3))], dim=1)
    return energies, coords
