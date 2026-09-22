"""Point-cloud representation for calorimeter jets (parallel to the grid path).

Calorimeter deposits are ~97% zeros on a dense grid (see ``weighted_l1_loss``
in ``multiscale_sr/engine.py``), so a hit list — an unordered set of
``(channel, x, y, energy)`` entries — carries the same information in a
fraction of the entries and has no "mostly empty grid" for a reconstruction
loss to be diluted across.

Nothing in this subpackage is imported by the grid pipeline. It is additive:
``multiscale_sr/`` outside this directory is unchanged, and its tests stay
green.
"""
from __future__ import annotations

from .hits import dense_to_hits, hits_to_dense, hit_counts, topk_hits
from .losses import (
    chamfer_loss,
    channel_loss,
    energy_conservation_loss,
    existence_loss,
    masked_chamfer_loss,
    masked_sinkhorn_loss,
    sinkhorn_loss,
)
from .collate import dense_pair_to_padded_hits, pad_hits, unpad_hits
from .models import PermInvariantEncoder, SetDiscriminator, SetGenerator

__all__ = [
    "dense_to_hits", "hits_to_dense", "hit_counts", "topk_hits",
    "chamfer_loss", "sinkhorn_loss", "energy_conservation_loss",
    "masked_chamfer_loss", "masked_sinkhorn_loss", "existence_loss", "channel_loss",
    "pad_hits", "unpad_hits", "dense_pair_to_padded_hits",
    "PermInvariantEncoder", "SetGenerator", "SetDiscriminator",
]
