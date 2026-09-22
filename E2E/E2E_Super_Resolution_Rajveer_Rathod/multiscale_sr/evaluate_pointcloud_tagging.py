"""Tagging-efficiency evaluation for a Stage 3 point-cloud checkpoint.

Reuses the EXISTING frozen-tagger machinery (multiscale_sr/tagger.py) unchanged
-- no parallel tagger, per the implementation prompt's hard constraint -- by
re-rasterizing the SetGenerator's hit-list output to a dense image with
pointcloud.hits.hits_to_dense before scoring. This is the one conversion
train_pointcloud.py's module docstring describes as the tagging boundary.

Deliberately a SEPARATE script from classification_eval.py rather than a flag
added to it: classification_eval.py's main() loads a grid Generator and calls
it as generator(lr, target_hw) -> dense image directly (see its ckpt_args-based
construction), which is architecturally incompatible with SetGenerator's
(lr_hits, lr_mask, lr_grid, hr_grid) -> (hits, exist_logits, chan_logits)
signature. Forking that logic into two code paths inside one 900-line script
would be harder to review than this file, which mirrors its structure at the
points that must match (tagger training/scoring, AUC/efficiency/recovery math)
and diverges only at data materialization.

Usage:
    python evaluate_pointcloud_tagging.py --checkpoint <run>/checkpoints/best.pt \
        --data-dir ../datasets_20pct --max-samples 4000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from multiscale_sr.data import get_dataloader
from multiscale_sr.data.normalization import normalize
from multiscale_sr.tagger import eval_tagger_auc, roc_auc, train_tagger, tagger_scores
from multiscale_sr.utils import resolve_env, seed_everything

from multiscale_sr.pointcloud.collate import dense_pair_to_padded_hits
from multiscale_sr.pointcloud.hits import hits_to_dense
from multiscale_sr.pointcloud.models import SetGenerator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.33,
                    help="Must match the checkpoint's training --val-ratio, same "
                         "requirement classification_eval.py documents")
    p.add_argument("--max-samples", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tagger-width", type=int, default=32,
                    help="Same default as classification_eval.py, for comparable AUCs")
    p.add_argument("--tagger-epochs", type=int, default=15)
    p.add_argument("--test-frac", type=float, default=0.5,
                    help="Fraction of materialized samples held out for tagger evaluation")
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--existence-threshold", type=float, default=0.0,
                    help="Sigmoid threshold (as a pre-sigmoid logit) above which a "
                         "generated slot is kept as a real hit -- see "
                         "train_pointcloud._existing_hits_only")
    p.add_argument("--out-json", default=None)
    return p


@torch.no_grad()
def materialize_pointcloud_tagging_data(
    generator: SetGenerator, loader, stats, device: torch.device, args,
) -> dict[str, Tensor]:
    """Point-cloud analogue of engine.collect_tagging_tensors.

    Returns the SAME dict shape (hr/lr/sr each (N,3,H,W) normalized, y (N,))
    so downstream tagger training/scoring is byte-for-byte the grid pipeline's
    code, just called from here instead of classification_eval.py.

    LR is reported as a bicubic upsample of the raw LR image to HR size, the
    SAME convention collect_tagging_tensors uses -- LR hits are NOT
    re-rasterized through the point-cloud path, since the LR side is real
    sensor/downsample data, not something SetGenerator produces.
    """
    import torch.nn.functional as F

    generator.eval()
    hr_list, lr_list, sr_list, y_list = [], [], [], []
    seen = 0
    for batch in loader:
        if args.max_samples is not None:
            remaining = args.max_samples - seen
            if remaining <= 0:
                break
            batch = {k: v[:remaining] for k, v in batch.items()}

        lr_img = batch["lr"].to(device)
        hr_img = batch["hr"].to(device)
        lr_hits, lr_mask, _hr_hits, _hr_mask = dense_pair_to_padded_hits(
            lr_img, hr_img, args.lr_n_max, args.hr_n_max,
        )

        pred_hits, exist_logits, _chan_logits = generator(
            lr_hits, lr_mask, args.scale, args.hr_size,
        )
        sr_raw = _rasterize(pred_hits, exist_logits, hr_img.shape, args.existence_threshold)

        lr_up = F.interpolate(lr_img, size=hr_img.shape[-2:], mode="bicubic", align_corners=False)

        hr_list.append(normalize(hr_img, stats).cpu())
        lr_list.append(normalize(lr_up.clamp_min(0.0), stats).cpu())
        sr_list.append(normalize(sr_raw, stats).cpu())
        y_list.append(batch["y"].view(-1).cpu())
        seen += hr_img.shape[0]

    if not y_list:
        raise ValueError("Loader produced no samples")
    y = torch.cat(y_list)
    if y.unique().numel() < 2:
        raise ValueError("Materialized labels have fewer than 2 classes")
    return {
        "hr": torch.cat(hr_list), "lr": torch.cat(lr_list), "sr": torch.cat(sr_list), "y": y,
    }


def _rasterize(pred_hits, exist_logits, hr_shape, threshold) -> Tensor:
    """Padded generator output -> dense raw-energy image, keeping only slots
    whose existence logit clears `threshold` (0.0 = sigmoid > 0.5). Same
    function as train_pointcloud._existing_hits_only + hits_to_dense, kept
    local here so this script has no import-time dependency on train_pointcloud.py
    (a training entry point, not a library module)."""
    b, n, _ = pred_hits.shape
    keep = exist_logits > threshold
    batch_idx = torch.arange(b, device=pred_hits.device).unsqueeze(1).expand(b, n)
    flat = pred_hits[keep]
    flat_batch = batch_idx[keep].to(pred_hits.dtype)
    hits = torch.cat([flat_batch.unsqueeze(1), flat], dim=1)
    return hits_to_dense(hits, tuple(hr_shape))


def _split_indices(n: int, test_frac: float, seed: int) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(round(n * test_frac)))
    return perm[n_test:], perm[:n_test]


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.eval_seed)
    env = resolve_env()
    print(f"[env] {env}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ckpt_args = ckpt["args"]
    checkpoint_epoch = ckpt["epoch"]
    args.scale = ckpt_args["scale"]
    args.hr_size = ckpt_args["hr_size"]
    args.lr_n_max = ckpt_args["lr_n_max"]
    args.hr_n_max = ckpt_args["hr_n_max"]
    print(f"[pointcloud-eval] checkpoint epoch={ckpt['epoch']} scale={args.scale} "
          f"lr_n_max={args.lr_n_max} hr_n_max={args.hr_n_max}")

    generator = SetGenerator(
        hr_n_max=args.hr_n_max, d_model=ckpt_args["d_model"], hidden=ckpt_args["hidden"],
        n_layers=ckpt_args["n_layers"], n_heads=ckpt_args["n_heads"],
    ).to(env.device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    del ckpt

    # Stats: the dense grid pipeline's ChannelStats, computed fresh here (this
    # checkpoint never saved one, since SetGenerator normalizes coordinates/
    # energy internally rather than via ChannelStats -- see train_pointcloud.py's
    # data_and_normalization phase comment). Loading with the SAME data-dir/
    # val-ratio/scale the grid model used keeps the tagger's input scale
    # identical, which is what makes AUCs across the two approaches comparable.
    from multiscale_sr.data.normalization import (
        discover_parquet_files, split_files, stream_channel_stats_parquet,
    )
    files = discover_parquet_files(Path(args.data_dir))
    train_files, _ = split_files(files, args.val_ratio)
    stats = stream_channel_stats_parquet(train_files, batch_size=128, hr_size=args.hr_size)
    print(f"[data] stats: mean={stats.mean.tolist()} std={stats.std.tolist()}")

    test_loader, _ = get_dataloader(
        path=Path(args.data_dir), split="test", env=env, batch_size=args.batch_size,
        scale=args.scale, hr_size=args.hr_size, val_ratio=args.val_ratio,
    )

    print(f"[pointcloud-eval] materializing up to {args.max_samples} samples...")
    data = materialize_pointcloud_tagging_data(generator, test_loader, stats, env.device, args)
    y = data["y"]
    n = y.shape[0]
    train_idx, test_idx = _split_indices(n, args.test_frac, args.eval_seed)
    y_test = y[test_idx]
    print(f"[pointcloud-eval] n={n} (tagger-train={len(train_idx)} "
          f"tagger-test={len(test_idx)}) class balance: {torch.bincount(y).tolist()}")

    # Fixed HR tagger, scored on every source -- same measurement protocol as
    # classification_eval.py's primary_fixed_hr_tagger, so the numbers below
    # are directly comparable to the grid baseline's 89.7%/87.8% tagging
    # efficiency and 44.8%/35.1% recovery fraction.
    hr_tagger = train_tagger(
        data["hr"][train_idx], y[train_idx], env.device,
        width=args.tagger_width, epochs=args.tagger_epochs, seed=args.eval_seed,
    )
    scores = {src: tagger_scores(hr_tagger, data[src][test_idx], env.device) for src in ("hr", "lr", "sr")}
    auc = {}
    for src in ("hr", "lr", "sr"):
        _, _, a = roc_auc(scores[src], y_test.numpy())
        auc[src] = a
        print(f"  fixed-HR-tagger on {src.upper():2s}: AUC={a:.4f}")

    efficiency = (auc["sr"] / auc["hr"]) if auc["hr"] else float("nan")
    gap = auc["hr"] - auc["lr"]
    recovery = ((auc["sr"] - auc["lr"]) / gap) if abs(gap) > 1e-6 else float("nan")
    print(f"[pointcloud-eval] tagging efficiency (AUC_SR/AUC_HR) = {efficiency:.1%}")
    print(f"[pointcloud-eval] recovery fraction (LR->HR gap closed by SR) = {recovery:.1%}")

    results = {
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint_epoch, "scale": args.scale,
        "n": n, "n_train": len(train_idx), "n_test": len(test_idx),
        "auc": auc, "tagging_efficiency_sr_over_hr": efficiency,
        "recovery_fraction_lr_to_hr": recovery,
    }
    out = Path(args.out_json) if args.out_json else Path(args.checkpoint).parent.parent / "pointcloud_tagging_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[pointcloud-eval] wrote {out}")


if __name__ == "__main__":
    main()
