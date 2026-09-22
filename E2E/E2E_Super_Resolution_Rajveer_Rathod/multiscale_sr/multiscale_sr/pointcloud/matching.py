"""Discrete geometric target assignment for existence/channel supervision."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch import Tensor


@torch.no_grad()
def hungarian_assignment(pred_hits: Tensor, target_hits: Tensor, target_mask: Tensor, *, workers: int = 4) -> Tensor:
    """Return target row indices per predicted slot, or -1 for unmatched slots.

    Minimize total squared (x,y) distance, independently per sample. Matching
    is detached and solved on CPU (up to four independent SciPy solves in
    parallel by default); reconstruction gradients still come from
    the unchanged Chamfer/Sinkhorn losses. No existence threshold is used.
    Each real target is assigned at most once. When targets outnumber slots,
    unassigned targets contribute nothing to existence/channel supervision,
    but remain available to the independent reconstruction losses.

    Canonically order targets by (x,y,channel,energy) before solving. Multiple
    channels can occupy the same coordinates: canonical ordering makes solver
    tie-breaking independent of arbitrary target storage order. Geometry alone
    does not resolve the physical channel ambiguity at coincident positions.
    """
    if pred_hits.ndim != 3 or target_hits.ndim != 3 or pred_hits.shape[-1] != 4 or target_hits.shape[-1] != 4:
        raise ValueError("pred_hits and target_hits must have shape (B,N,4)")
    if pred_hits.shape[0] != target_hits.shape[0] or target_mask.shape != target_hits.shape[:2]:
        raise ValueError("batch size or target_mask shape mismatch")
    if workers < 1:
        raise ValueError("workers must be positive")
    pred = pred_hits.detach().cpu().double().numpy()
    target = target_hits.detach().cpu().double().numpy()
    mask = target_mask.detach().cpu().bool().numpy()
    assigned = np.full(pred.shape[:2], -1, dtype=np.int64)
    def solve(b):
        real = np.flatnonzero(mask[b])
        if len(real) == 0 or pred.shape[1] == 0:
            return
        rows = target[b, real]
        order = np.lexsort((rows[:, 3], rows[:, 0], rows[:, 2], rows[:, 1]))
        real = real[order]
        cost = cdist(pred[b, :, 1:3], target[b, real, 1:3], metric="sqeuclidean")
        p, t = linear_sum_assignment(cost)
        assigned[b, p] = real[t]

    if workers == 1 or len(pred) < 2:
        for b in range(len(pred)):
            solve(b)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(pred))) as pool:
            # Consume results so solver exceptions propagate to the caller.
            list(pool.map(solve, range(len(pred))))
    return torch.as_tensor(assigned, device=pred_hits.device)


@torch.no_grad()
def matched_supervision(pred_hits: Tensor, target_hits: Tensor, target_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Return channel targets and existence mask in predicted-slot order."""
    assignment = hungarian_assignment(pred_hits, target_hits, target_mask)
    matched = assignment >= 0
    if target_hits.shape[1] == 0:
        return pred_hits.new_zeros(pred_hits.shape[:2]), matched
    channels = target_hits[..., 0].gather(1, assignment.clamp_min(0))
    return channels.masked_fill(~matched, 0), matched
