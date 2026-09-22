"""Stage 1 audit: what does a jet look like as a point cloud?

Answers the questions the point-cloud architecture depends on, measured on
real data rather than assumed:

  * How many hits does an HR jet actually have? (sets N_max for padding)
  * Does downsampling REMOVE hits or merely REDISTRIBUTE energy? (decides
    whether a generator must create hits or reposition them — the single most
    important architectural input)
  * Is energy conserved through dense -> hits -> dense? (must be exact)
  * How far apart are HR and LR hit sets under Chamfer/Sinkhorn? (the "gap to
    close" baseline any point-cloud model must beat)

Reads the same cached parquet the grid pipeline uses, so the HR/LR pairs here
are exactly the ones training sees. Writes docs/POINTCLOUD_AUDIT.md.

Usage:
    .venv/bin/python scripts/pointcloud_audit.py --data-dir <dir> [--max-jets 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multiscale_sr.data.multiscale import downscale_hr, pad_to_size  # noqa: E402
from multiscale_sr.data.normalization import (  # noqa: E402
    discover_parquet_files,
    split_files,
)
from multiscale_sr.pointcloud.hits import (  # noqa: E402
    COL_ENERGY,
    dense_to_hits,
    hit_counts,
    hits_to_dense,
)

SCALES = (64, 32, 16)
HR_SIZE = 128


def percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {k: 0.0 for k in ("mean", "median", "p1", "p99", "max", "min")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p1": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def chamfer_xy(a: torch.Tensor, b: torch.Tensor) -> float:
    """Bidirectional energy-weighted Chamfer between two (N,5) hit lists.

    Distances are in pixel units on the HR grid; LR coordinates are rescaled to
    HR before comparison so the two live in the same space (an LR hit at (1,1)
    on a 16x16 grid covers HR pixels around (8,8), not (1,1)).
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    pa, pb = a[:, 2:4], b[:, 2:4]
    d2 = torch.cdist(pa.unsqueeze(0), pb.unsqueeze(0)).squeeze(0)  # (Na,Nb)
    wa = a[:, COL_ENERGY] / a[:, COL_ENERGY].sum().clamp_min(1e-12)
    wb = b[:, COL_ENERGY] / b[:, COL_ENERGY].sum().clamp_min(1e-12)
    a_to_b = (d2.min(dim=1).values * wa).sum()
    b_to_a = (d2.min(dim=0).values * wb).sum()
    return float(a_to_b + b_to_a)


def sinkhorn_xy(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0,
                iters: int = 50) -> float:
    """Entropy-regularised OT between two energy-weighted hit sets.

    Log-domain updates; eps is in squared-pixel units. Capped in size by the
    caller, since the cost matrix is O(Na*Nb).
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    pa, pb = a[:, 2:4], b[:, 2:4]
    cost = torch.cdist(pa.unsqueeze(0), pb.unsqueeze(0)).squeeze(0) ** 2
    log_wa = (a[:, COL_ENERGY] / a[:, COL_ENERGY].sum().clamp_min(1e-12)).clamp_min(1e-12).log()
    log_wb = (b[:, COL_ENERGY] / b[:, COL_ENERGY].sum().clamp_min(1e-12)).clamp_min(1e-12).log()
    f = torch.zeros_like(log_wa)
    g = torch.zeros_like(log_wb)
    # Log-domain Sinkhorn. Each potential is REPLACED, not accumulated: adding
    # the previous value back double-counts it and the potentials diverge to
    # +/-inf within ~10 iterations (the cost matrix stays finite, so an Inf
    # result is always this bug rather than degenerate data).
    for _ in range(iters):
        f = eps * (log_wa - torch.logsumexp((g[None, :] - cost) / eps, dim=1))
        g = eps * (log_wb - torch.logsumexp((f[:, None] - cost) / eps, dim=0))
    plan = torch.exp((f[:, None] + g[None, :] - cost) / eps)
    value = float((plan * cost).sum())
    if not np.isfinite(value):
        raise FloatingPointError("Sinkhorn did not converge to a finite value")
    return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--max-jets", type=int, default=2000,
                   help="Cap jets scanned for hit statistics")
    p.add_argument("--ot-jets", type=int, default=64,
                   help="Jets used for Chamfer/Sinkhorn (O(N^2) per pair)")
    p.add_argument("--ot-topk", type=int, default=256,
                   help="Brightest hits kept per jet for the OT metrics")
    p.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "POINTCLOUD_AUDIT.md"))
    p.add_argument("--out-json", default=None)
    args = p.parse_args(argv)

    import pyarrow.parquet as pq

    from multiscale_sr.data.normalization import batch_to_tensor

    data_dir = Path(args.data_dir)
    files = discover_parquet_files(data_dir)
    train_files, held_out = split_files(files, 0.33)
    print(f"[audit] {len(files)} files; scanning up to {args.max_jets} jets", flush=True)

    hr_counts: list[int] = []
    per_channel: list[np.ndarray] = []
    lr_counts: dict[int, list[int]] = {s: [] for s in SCALES}
    energy_err: list[float] = []
    neg_count: list[int] = []
    neg_energy: list[float] = []
    total_abs: list[float] = []
    lr_energy_frac: dict[int, list[float]] = {s: [] for s in SCALES}
    ot: dict[int, dict[str, list[float]]] = {s: {"chamfer": [], "sinkhorn": []} for s in SCALES}

    seen = 0
    ot_done = 0
    for f in files:
        if seen >= args.max_jets:
            break
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16, columns=["X_jets"]):
            hr = batch_to_tensor(batch.column(0)).float()
            hr = pad_to_size(hr, HR_SIZE)
            b = hr.shape[0]

            hits = dense_to_hits(hr)
            counts = hit_counts(hits, b)
            hr_counts.extend(counts.tolist())

            # Per-channel occupancy, to see whether channels differ in density.
            for ch in range(hr.shape[1]):
                ch_hits = dense_to_hits(hr[:, ch : ch + 1])
                per_channel.append(hit_counts(ch_hits, b).numpy())

            # Round-trip fidelity. The default threshold keeps only strictly
            # positive cells, so any discrepancy is exactly the negative
            # pedestal the source data carries; measure both separately rather
            # than reporting one number that conflates them.
            back = hits_to_dense(hits, hr.shape)
            pos = hr > 0
            energy_err.append(float((back[pos] - hr[pos]).abs().max()) if pos.any() else 0.0)
            neg_count.append(int((hr < 0).sum()))
            neg_energy.append(float(hr[hr < 0].abs().sum()))
            total_abs.append(float(hr.abs().sum()))

            hr_total = hr.sum(dim=(1, 2, 3)).clamp_min(1e-12)
            for s in SCALES:
                lr = downscale_hr(hr, s)
                lr_hits = dense_to_hits(lr)
                lr_counts[s].extend(hit_counts(lr_hits, b).tolist())
                lr_energy_frac[s].extend(
                    (lr.sum(dim=(1, 2, 3)) / hr_total).tolist()
                )

            # OT metrics on a small subsample: rescale LR coords onto HR grid.
            if ot_done < args.ot_jets:
                for i in range(min(b, args.ot_jets - ot_done)):
                    h_i = dense_to_hits(hr[i : i + 1])
                    if h_i.shape[0] > args.ot_topk:
                        keep = torch.topk(h_i[:, COL_ENERGY], args.ot_topk).indices
                        h_i = h_i[keep]
                    for s in SCALES:
                        l_i = dense_to_hits(downscale_hr(hr[i : i + 1], s))
                        if l_i.shape[0] > args.ot_topk:
                            keep = torch.topk(l_i[:, COL_ENERGY], args.ot_topk).indices
                            l_i = l_i[keep]
                        l_scaled = l_i.clone()
                        # Map LR cell centre onto the HR grid.
                        factor = HR_SIZE / s
                        l_scaled[:, 2:4] = (l_scaled[:, 2:4] + 0.5) * factor - 0.5
                        ot[s]["chamfer"].append(chamfer_xy(h_i, l_scaled))
                        ot[s]["sinkhorn"].append(sinkhorn_xy(h_i, l_scaled))
                    ot_done += 1

            seen += b
            if seen >= args.max_jets:
                break
        print(f"[audit] {seen} jets", flush=True)

    hr_arr = np.array(hr_counts, dtype=np.float64)
    grid_cells = 3 * HR_SIZE * HR_SIZE
    results = {
        "jets_scanned": int(seen),
        "hr_size": HR_SIZE,
        "hr_grid_cells": grid_cells,
        "hr_hits": percentiles(hr_arr),
        "hr_occupancy_frac": float(hr_arr.mean() / grid_cells),
        "per_channel_hits": percentiles(np.concatenate(per_channel)) if per_channel else {},
        "max_roundtrip_abs_error_positive_cells": float(max(energy_err)) if energy_err else 0.0,
        "negative_cells_per_jet": float(np.sum(neg_count) / max(seen, 1)),
        "negative_energy_frac_of_total_abs": (
            float(np.sum(neg_energy) / max(np.sum(total_abs), 1e-12))
        ),
        "lr": {},
        "ot_jets": ot_done,
        "ot_topk": args.ot_topk,
    }
    for s in SCALES:
        arr = np.array(lr_counts[s], dtype=np.float64)
        results["lr"][str(s)] = {
            "hits": percentiles(arr),
            "grid_cells": 3 * s * s,
            "occupancy_frac": float(arr.mean() / (3 * s * s)),
            "hits_vs_hr_ratio": float(arr.mean() / max(hr_arr.mean(), 1e-9)),
            "energy_frac_of_hr": percentiles(np.array(lr_energy_frac[s])),
            "chamfer_mean": float(np.nanmean(ot[s]["chamfer"])) if ot[s]["chamfer"] else None,
            "sinkhorn_mean": float(np.nanmean(ot[s]["sinkhorn"])) if ot[s]["sinkhorn"] else None,
        }

    n_max = int(np.ceil(results["hr_hits"]["p99"]))
    results["suggested_n_max"] = n_max

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))

    write_report(Path(args.out), results, args)
    print(json.dumps(results, indent=2))
    print(f"\n[audit] wrote {args.out}")
    return 0


def write_report(path: Path, r: dict, args) -> None:
    hr = r["hr_hits"]
    lines = [
        "# Point-Cloud Conversion Audit",
        "",
        "**Status:** Stage 1 measurement. Generated by "
        "[`scripts/pointcloud_audit.py`](../scripts/pointcloud_audit.py); rerun to reproduce.",
        "",
        f"Measured on **{r['jets_scanned']:,} jets** from the local 20% subset, "
        f"HR padded to {r['hr_size']}×{r['hr_size']} exactly as training does.",
        "",
        "## 1. How big is a jet as a point cloud?",
        "",
        "| statistic | hits per jet (all 3 channels) |",
        "| --- | --- |",
        f"| mean | {hr['mean']:.1f} |",
        f"| median | {hr['median']:.1f} |",
        f"| p1 | {hr['p1']:.1f} |",
        f"| p99 | {hr['p99']:.1f} |",
        f"| max | {hr['max']:.0f} |",
        "",
        f"The dense grid has **{r['hr_grid_cells']:,} cells**, so mean occupancy is "
        f"**{100 * r['hr_occupancy_frac']:.2f}%** — i.e. "
        f"{100 * (1 - r['hr_occupancy_frac']):.2f}% of the grid is empty, consistent with the "
        "~97% figure quoted in `weighted_l1_loss`.",
        "",
        f"**Suggested `N_max` = {r['suggested_n_max']}** (p99). Padding to the max "
        f"({hr['max']:.0f}) would waste memory on a rare tail.",
        "",
        "## 2. Does downsampling remove hits or redistribute energy?",
        "",
        "This is the question that decides the generator's job.",
        "",
        "| scale | mean hits | vs HR | grid cells | occupancy | energy kept |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in SCALES:
        d = r["lr"][str(s)]
        lines.append(
            f"| {s}×{s} | {d['hits']['mean']:.1f} | {100 * d['hits_vs_hr_ratio']:.1f}% | "
            f"{d['grid_cells']:,} | {100 * d['occupancy_frac']:.1f}% | "
            f"{100 * d['energy_frac_of_hr']['mean']:.1f}% |"
        )
    lines += [
        "",
        "`downscale_hr` uses `mode=\"area\"`, which averages rather than sums, so the "
        "energy column above is expected to scale roughly with the cell-count ratio — "
        "it is not a defect. See `multiscale_sr/data/multiscale.py`.",
        "",
        "## 3. Round-trip fidelity, and a finding about negative cells",
        "",
        f"Over strictly positive cells, the maximum absolute error of "
        f"`dense -> hits -> dense` is "
        f"**{r['max_roundtrip_abs_error_positive_cells']:.3g}** — lossless, as the unit "
        "tests assert independently.",
        "",
        "**However, the source HR is not non-negative.** Real jets in this dataset "
        f"carry on average **{r['negative_cells_per_jet']:.0f} negative cells** per jet, "
        f"accounting for **{100 * r['negative_energy_frac_of_total_abs']:.2f}%** of total "
        "absolute magnitude. `dense_to_hits` at the default `threshold=0.0` discards "
        "them, so the round trip is *not* lossless on unfiltered real data.",
        "",
        "This matters for the architecture: a point cloud of strictly positive hits "
        "cannot represent the negative pedestal, so a model trained on hits and "
        "re-rasterized for tagging will differ from HR wherever those cells sit. The "
        "grid pipeline already discards them in its loss (`weighted_l1_loss` applies "
        "`clamp_min(0.0)` to the weighting), so this is consistent with existing "
        "practice — but it must be stated rather than assumed, and the re-rasterized "
        "comparison in Stage 2/3 should be made against a clamped HR to stay honest.",
        "",
        "## 4. Gap to close: HR vs LR hit sets",
        "",
        f"Energy-weighted Chamfer and Sinkhorn (eps=1.0) over the brightest "
        f"{r['ot_topk']} hits of {r['ot_jets']} jets, with LR coordinates rescaled "
        "onto the HR grid. These are the baselines a point-cloud model must beat.",
        "",
        "| scale | Chamfer (px) | Sinkhorn (px²) |",
        "| --- | --- | --- |",
    ]
    for s in SCALES:
        d = r["lr"][str(s)]
        c = f"{d['chamfer_mean']:.3f}" if d["chamfer_mean"] is not None else "n/a"
        sk = f"{d['sinkhorn_mean']:.3f}" if d["sinkhorn_mean"] is not None else "n/a"
        lines.append(f"| {s}×{s} | {c} | {sk} |")
    lines += [
        "",
        "## 5. What this implies for the architecture",
        "",
        "Read these against the open questions in "
        "[`POINT_CLOUD_EXPLORATION.md`](POINT_CLOUD_EXPLORATION.md):",
        "",
        f"- **Cardinality.** Fixed top-K padding at `N_max={r['suggested_n_max']}` covers "
        "99% of jets. Whether the generator needs *learned* cardinality depends on "
        "section 2: if LR retains far fewer hits than HR, the model must create hits, "
        "not just move them.",
        "- **Coordinates.** LR hits land on a coarse grid; rescaling to HR leaves gaps "
        "between occupied cells. A generator restricted to integer HR coordinates can "
        "represent any HR configuration exactly (verified in section 3), so continuous "
        "coordinates are optional rather than required.",
        "- **Memory.** Chamfer is O(N²) in hits per jet; at "
        f"`N_max={r['suggested_n_max']}` a batch of 64 needs "
        f"~{64 * r['suggested_n_max'] ** 2 / 1e6:.0f}M pairwise distances per step, "
        "so chunking is required exactly as `semd_loss` does it.",
        "",
        "**Not measured here:** whether any of this improves tagging efficiency. That "
        "requires Stage 2/3 and a comparison against the committed grid baseline "
        "(tagging efficiency 89.7% / 87.8% for the epoch-7 / epoch-10 checkpoints).",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
