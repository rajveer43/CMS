"""Padded batching for variable-cardinality hit lists.

The audit (`docs/POINTCLOUD_AUDIT.md`) found the generator's job is not
"reposition existing energy" but "create most of the missing hits" — LR64
carries 45% of HR's hit count, LR16 only 18%. So both LR and HR sides of a
batch have a *different* hit count per sample, and both need padding to a
fixed tensor shape before they can be stacked.

Padding convention: pad with zero-energy, zero-coordinate rows and carry an
explicit boolean mask. A padded row's energy is 0, so it never contributes to
an energy-weighted loss even where the mask is accidentally ignored — the mask
is required for correctness (e.g. existence-score supervision, where a
zero-energy real hit is indistinguishable from padding without it), not merely
a performance optimization.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .hits import COL_BATCH, N_COLS, dense_to_hits


def pad_hits(hits: Tensor, batch_size: int, n_max: int) -> tuple[Tensor, Tensor]:
    """Convert a flat ``(N,5)`` hit list into a padded ``(B, n_max, 4)`` batch.

    Returns ``(padded, mask)``:

    - ``padded``: ``(B, n_max, 4)`` of ``(channel, x, y, energy)`` — batch index
      is dropped since it is now the tensor's own leading dimension.
    - ``mask``: ``(B, n_max)`` bool, ``True`` where the row is a real hit.

    Samples with more than ``n_max`` hits are **truncated to their brightest
    n_max**, not raised on — this mirrors ``topk_hits``' truncate-not-error
    convention and keeps batch shapes static, which padded batching requires
    regardless. Truncation silently drops the dimmest hits; call
    ``hit_counts`` on the source first if you need to know whether this is
    happening for your data (the audit's suggested ``N_max`` is chosen at
    HR's p99, so truncation should be rare for HR and essentially never for
    the LR side, which has far fewer hits per sample).
    """
    if n_max < 1:
        raise ValueError(f"n_max must be positive, got {n_max}")

    device, dtype = hits.device, hits.dtype
    padded = torch.zeros(batch_size, n_max, N_COLS - 1, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, n_max, device=device, dtype=torch.bool)

    if hits.numel() == 0:
        return padded, mask

    batch_idx = hits[:, COL_BATCH].round().long()
    for b in range(batch_size):
        rows = hits[batch_idx == b]
        if rows.shape[0] > n_max:
            keep = torch.topk(rows[:, 4], n_max).indices  # brightest n_max
            rows = rows[keep]
        k = rows.shape[0]
        if k == 0:
            continue
        padded[b, :k] = rows[:, 1:5]  # drop COL_BATCH, keep channel/x/y/energy
        mask[b, :k] = True
    return padded, mask


def unpad_hits(padded: Tensor, mask: Tensor) -> Tensor:
    """Inverse of ``pad_hits``: ``(B, n_max, 4)`` + mask -> flat ``(N,5)`` hit list.

    Re-attaches the batch index from the row's position, so the result has the
    same layout ``dense_to_hits`` produces and can be passed to
    ``hits_to_dense`` directly.
    """
    b, n_max, _ = padded.shape
    if mask.shape != (b, n_max):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match padded {tuple(padded.shape)}")

    batch_idx = torch.arange(b, device=padded.device).unsqueeze(1).expand(b, n_max)
    flat = padded[mask]
    flat_batch = batch_idx[mask].to(padded.dtype)
    return torch.cat([flat_batch.unsqueeze(1), flat], dim=1)


def dense_pair_to_padded_hits(
    lr_img: Tensor, hr_img: Tensor, lr_n_max: int, hr_n_max: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convenience wrapper: a dense (LR, HR) batch -> four padded/masked tensors.

    ``(lr_hits, lr_mask, hr_hits, hr_mask)``. This is the boundary conversion
    ``train_pointcloud.py`` calls once per batch — everything past this point
    (generator, discriminator, losses) works on hit lists only, never a dense
    image, matching the Stage 3 design: point clouds in, point clouds out.
    """
    b = lr_img.shape[0]
    lr_hits, lr_mask = pad_hits(dense_to_hits(lr_img), b, lr_n_max)
    hr_hits, hr_mask = pad_hits(dense_to_hits(hr_img), b, hr_n_max)
    return lr_hits, lr_mask, hr_hits, hr_mask
