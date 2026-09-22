"""Presentation figures for the point-cloud SR pipeline.

Produces four figures that together tell the story documented in
docs/POINTCLOUD_RESULTS.md:

  1. jets_scatter.png     -- LR / SR / HR as actual POINT CLOUDS (scatter,
                             marker size by energy). The representation the
                             model actually works in, which a dense image
                             view hides.
  2. jets_raster.png      -- the same jets rasterized to dense images, in the
                             grid pipeline's house style, plus the SR-HR
                             residual. This is what the tagger sees.
  3. energy_profile.png   -- sorted per-hit energies, SR vs HR. Shows whether
                             the model reproduces HR's peaked distribution or
                             smears energy flat (the Fix 1 failure mode).
  4. chamfer_vs_tagging.png -- the headline result: Chamfer improves sharply
                             while tagging efficiency barely moves.

Intensity scales are shared across LR/SR/HR panels with HR setting the
ceiling, the same convention engine.py's sample panel uses, so a dim SR looks
dim instead of being silently rescaled to look bright.

Usage:
    python scripts/pointcloud_figures.py \
        --checkpoint <run>/checkpoints/best.pt \
        --data-dir <path>/datasets_20pct --out-dir <run>/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from multiscale_sr.data import get_dataloader  # noqa: E402
from multiscale_sr.utils import resolve_env, seed_everything  # noqa: E402

from multiscale_sr.pointcloud.collate import dense_pair_to_padded_hits  # noqa: E402
from multiscale_sr.pointcloud.hits import hits_to_dense  # noqa: E402
from multiscale_sr.pointcloud.models import SetGenerator  # noqa: E402
# Reuses the diagnostic script's generation-aware loader, so pre-Stage-B
# checkpoints (which store a plain `queries` parameter and will not load into
# the current SetGenerator) can be visualized too. Without this, figures could
# only ever be made for the newest architecture.
from cross_chamfer_diagnostic import load_generator as _load_gen  # noqa: E402

CMAP = "inferno"


def load_generator(path: str, device):
    gen, ck, arch = _load_gen(path, device)
    epoch = torch.load(path, map_location="cpu", weights_only=True)["epoch"]
    return gen, ck, epoch, arch


def _energy_img(dense: torch.Tensor) -> np.ndarray:
    """(3,H,W) raw -> log1p summed-channel energy map, as engine.py does."""
    return np.log1p(dense.sum(dim=0).cpu().numpy())


@torch.no_grad()
def get_batch(gen, ck, loader, device, n: int, threshold: float):
    batch = next(iter(loader))
    lr_img = batch["lr"][:n].to(device)
    hr_img = batch["hr"][:n].to(device)
    lr_hits, lr_mask, hr_hits, hr_mask = dense_pair_to_padded_hits(
        lr_img, hr_img, ck["lr_n_max"], ck["hr_n_max"],
    )
    pred_hits, exist_logits, _ = gen(lr_hits, lr_mask, ck["scale"], ck["hr_size"])
    keep = exist_logits > threshold
    return dict(
        lr_img=lr_img.cpu(), hr_img=hr_img.cpu(),
        lr_hits=lr_hits.cpu(), lr_mask=lr_mask.cpu(),
        hr_hits=hr_hits.cpu(), hr_mask=hr_mask.cpu(),
        pred_hits=pred_hits.cpu(), keep=keep.cpu(),
    )


def _scatter(ax, xs, ys, es, grid, vmax_e, title):
    if len(xs):
        # Energy is plotted on a LOG scale. Real jets put a large share of
        # their energy in a handful of hits, so a linear scale renders every
        # panel almost entirely black and hides the very difference these
        # figures exist to show. log1p keeps zero at zero and is the same
        # transform engine.py applies to its energy maps.
        le = np.log1p(es)
        lmax = max(float(np.log1p(vmax_e)), 1e-6)
        sizes = 3.0 + 70.0 * np.clip(le / lmax, 0.0, 1.0)
        ax.scatter(xs, ys, s=sizes, c=le, cmap=CMAP, vmin=0.0, vmax=lmax,
                   linewidths=0.0, alpha=0.85)
    ax.set_xlim(-1, grid); ax.set_ylim(-1, grid)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.set_facecolor("#08010f")


def fig_scatter(d, ck, out: Path, n: int):
    """LR / SR / HR as point clouds -- the model's native representation."""
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.5 * n))
    axes = np.atleast_2d(axes)
    for r in range(n):
        lm, hm = d["lr_mask"][r].bool(), d["hr_mask"][r].bool()
        lr = d["lr_hits"][r][lm].numpy()
        hr = d["hr_hits"][r][hm].numpy()
        sr = d["pred_hits"][r][d["keep"][r]].numpy()
        vmax_e = max(float(hr[:, 3].max()) if len(hr) else 1e-6, 1e-6)
        # LR lives on a coarser grid; scale its coords up purely for viewing.
        f = (ck["hr_size"] - 1) / max(ck["scale"] - 1, 1)
        _scatter(axes[r, 0], lr[:, 1] * f, lr[:, 2] * f, lr[:, 3], ck["hr_size"], vmax_e,
                 f"LR input ({len(lr)} hits)" if r == 0 else f"{len(lr)} hits")
        _scatter(axes[r, 1], sr[:, 1], sr[:, 2], sr[:, 3], ck["hr_size"], vmax_e,
                 f"SR predicted ({len(sr)} hits)" if r == 0 else f"{len(sr)} hits")
        _scatter(axes[r, 2], hr[:, 1], hr[:, 2], hr[:, 3], ck["hr_size"], vmax_e,
                 f"HR truth ({len(hr)} hits)" if r == 0 else f"{len(hr)} hits")
    fig.suptitle("Jets as point clouds - marker size and colour = hit energy",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_raster(d, ck, out: Path, n: int):
    """The same jets as dense images -- what the frozen tagger actually sees."""
    import torch.nn.functional as F
    hw = tuple(d["hr_img"].shape[-2:])
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.3 * n))
    axes = np.atleast_2d(axes)
    titles = ["LR (bicubic up)", "SR (re-rasterized)", "HR truth", "SR - HR"]
    for r in range(n):
        sr_hits = d["pred_hits"][r][d["keep"][r]]
        idx = torch.zeros(len(sr_hits), 1)
        sr_dense = hits_to_dense(torch.cat([idx, sr_hits], dim=1), (1, 3) + hw)[0]
        lr_up = F.interpolate(d["lr_img"][r : r + 1], size=hw, mode="bicubic",
                              align_corners=False).clamp_min(0.0)[0]
        imgs = [_energy_img(lr_up), _energy_img(sr_dense), _energy_img(d["hr_img"][r])]
        vmax = max(float(imgs[2].max()), 1e-6)
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap=CMAP, origin="lower", vmin=0.0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
        resid = imgs[1] - imgs[2]
        rmax = max(float(np.abs(resid).max()), 1e-6)
        axr = axes[r, 3]
        axr.imshow(resid, cmap="RdBu_r", origin="lower", vmin=-rmax, vmax=rmax)
        axr.set_xticks([]); axr.set_yticks([])
        if r == 0:
            axr.set_title(titles[3], fontsize=11)
    fig.suptitle("Same jets rasterized to dense images (log energy) - this is the tagger's view",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_energy_profile(d, out: Path, n: int, top: int = 40):
    """Sorted per-hit energies: does SR reproduce HR's peaked profile?"""
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for r in range(n):
        hm = d["hr_mask"][r].bool()
        hr_e = np.sort(d["hr_hits"][r][hm][:, 3].numpy())[::-1][:top]
        sr_e = np.sort(d["pred_hits"][r][d["keep"][r]][:, 3].numpy())[::-1][:top]
        ax = axes[0, r]
        ax.plot(hr_e, "o-", ms=3, lw=1.5, label="HR truth", color="#d62728")
        ax.plot(sr_e, "s-", ms=3, lw=1.5, label="SR predicted", color="#1f77b4")
        # Log y: HR spans ~0.5 to ~28 within a single jet, so a linear axis
        # compresses the whole low-energy tail into the baseline.
        ax.set_yscale("log")
        ax.set_xlabel(f"hit rank (top {top})", fontsize=9)
        if r == 0:
            ax.set_ylabel("hit energy", fontsize=9)
            ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Energy profile per jet - a flat SR curve means energy is smeared, not peaked",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_peak_smearing(d, out: Path):
    """The failure mode, stated as plainly as possible.

    Total energy is right; its DISTRIBUTION is not. The model spreads energy
    almost uniformly across ~750 hits instead of concentrating it in a few
    bright ones, which is what a jet tagger reads.
    """
    n = d["hr_mask"].shape[0]
    hr_tot, sr_tot, hr_peak, sr_peak = [], [], [], []
    for r in range(n):
        hm = d["hr_mask"][r].bool()
        h = d["hr_hits"][r][hm][:, 3].numpy()
        s_ = d["pred_hits"][r][d["keep"][r]][:, 3].numpy()
        hr_tot.append(h.sum()); sr_tot.append(s_.sum())
        hr_peak.append(h.max() if len(h) else 0.0)
        sr_peak.append(s_.max() if len(s_) else 0.0)

    x = np.arange(n); w = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    ax1.bar(x - w / 2, hr_tot, w, label="HR truth", color="#d62728")
    ax1.bar(x + w / 2, sr_tot, w, label="SR predicted", color="#1f77b4")
    # Titles are derived from the measured ratio, not hardcoded: these figures
    # are generated for every run, and some (pre-Stage-A) DO have peaks. A
    # fixed "collapsed" caption would mislabel those.
    tot_ratio = float(np.mean(np.array(sr_tot) / np.maximum(np.array(hr_tot), 1e-9)))
    peak_ratio = float(np.mean(np.array(sr_peak) / np.maximum(np.array(hr_peak), 1e-9)))
    ax1.set_title(f"Total energy per jet: SR/HR = {tot_ratio:.2f}x", fontsize=11)
    ax1.set_ylabel("summed hit energy", fontsize=10)

    ax2.bar(x - w / 2, hr_peak, w, label="HR truth", color="#d62728")
    ax2.bar(x + w / 2, sr_peak, w, label="SR predicted", color="#1f77b4")
    if peak_ratio < 0.25:
        peak_verdict = "collapsed"
    elif peak_ratio > 1.4:
        peak_verdict = "overshooting"
    else:
        peak_verdict = "comparable"
    ax2.set_title(f"Brightest single hit: {peak_verdict} (SR/HR = {peak_ratio:.2f}x)",
                  fontsize=11)
    ax2.set_ylabel("peak hit energy", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels([f"jet {i}" for i in x], fontsize=9)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    headline = ("Peak smearing: the right total energy, spread flat instead of peaked"
                if peak_ratio < 0.25 else
                "Energy distribution: total vs peak")
    fig.suptitle(headline, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_chamfer_vs_tagging(out: Path):
    """The headline: the trained objective improves, the real metric does not.

    Numbers are the measured results recorded in docs/POINTCLOUD_RESULTS.md.
    """
    stages = ["Chamfer\nonly\n(10ep)", "+ energy\nvalue\n(10ep)", "+ Stage\nA/B/C\n(5ep)"]
    chamfer = [None, 25.41, 19.90]   # val Chamfer; not recorded for the first run
    recovery = [-30.9, -25.6, -0.8]
    efficiency = [69.7, 70.9, 74.6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(stages))

    ax1.bar(x, efficiency, color="#1f77b4", width=0.55, label="point cloud")
    ax1.axhline(89.7, ls="--", lw=2, color="#2ca02c", label="grid GAN target (89.7%)")
    ax1.set_xticks(x); ax1.set_xticklabels(stages, fontsize=9)
    ax1.set_ylabel("tagging efficiency (%)", fontsize=10)
    ax1.set_ylim(0, 100); ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("Tagging efficiency vs the target", fontsize=11)

    colors = ["#d62728" if v < 0 else "#2ca02c" for v in recovery]
    ax2.bar(x, recovery, color=colors, width=0.55)
    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(44.8, ls="--", lw=2, color="#2ca02c", label="grid GAN (+44.8%)")
    for i, v in enumerate(recovery):
        ax2.text(i, v + (2 if v < 0 else -4), f"{v:+.1f}%", ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(stages, fontsize=9)
    ax2.set_ylabel("recovery fraction (%)", fontsize=10)
    ax2.set_ylim(-40, 55); ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.25)
    ax2.set_title("Recovery fraction: below 0 = worse than no model", fontsize=11)

    fig.suptitle("Three correct fixes moved the point-cloud objective, not the physics metric",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.33)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--existence-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary", action="store_true",
                   help="Also write the cross-run chamfer_vs_tagging.png summary")
    args = p.parse_args()

    env = resolve_env()
    seed_everything(args.seed)
    gen, ck, epoch, arch = load_generator(args.checkpoint, env.device)
    print(f"[figures] checkpoint epoch={epoch} scale={ck['scale']} arch={arch}")

    loader, _ = get_dataloader(
        path=Path(args.data_dir), split="test", env=env,
        batch_size=max(args.n_samples, 8), scale=ck["scale"],
        hr_size=ck["hr_size"], val_ratio=args.val_ratio,
    )
    d = get_batch(gen, ck, loader, env.device, args.n_samples, args.existence_threshold)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = args.n_samples
    fig_scatter(d, ck, out / "jets_scatter.png", n)
    fig_raster(d, ck, out / "jets_raster.png", n)
    fig_energy_profile(d, out / "energy_profile.png", n)
    fig_peak_smearing(d, out / "peak_smearing.png")
    # Cross-run summary: identical for every run (its numbers span all of
    # them), so it is written only where asked for rather than duplicated into
    # six directories.
    if args.summary:
        fig_chamfer_vs_tagging(out / "chamfer_vs_tagging.png")


if __name__ == "__main__":
    main()
