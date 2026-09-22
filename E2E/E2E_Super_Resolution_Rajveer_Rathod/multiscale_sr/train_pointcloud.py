"""Train the Stage 3 set-to-set point-cloud SR model.

Mirrors train.py's CLI/recovery conventions (resolve_args with config-under-
CLI precedence, Diagnostics phases, save_checkpoint recovery schema, same
metrics.jsonl file per run) so the two are easy to compare and operate the
same way, but the models/losses/data representation are entirely different:
LR hits -> SetGenerator -> HR hits, with no dense image inside the model (see
multiscale_sr/pointcloud/models.py's module docstring for why).

The dense grid pipeline (train.py, multiscale_sr/ outside pointcloud/) is
untouched by this file and stays the primary, tested path. This script reuses
its DataLoader (get_dataloader) only as a source of (LR image, HR image)
pairs — the boundary conversion to hits happens once per batch, in
_to_hits() below, via pointcloud.collate.dense_pair_to_padded_hits.

Usage:
    python train_pointcloud.py --config configs/scale_64.yaml --data-dir ../datasets \
        --hr-n-max 1300 --epochs 10

Re-rasterization for tagging: SR hits are converted back to a dense image with
pointcloud.hits.hits_to_dense before evaluation, so the EXISTING frozen
JetTagger and classification_eval.py work unchanged — no parallel tagger, per
the implementation prompt's hard constraint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

from multiscale_sr.data import detect_dataset_type, get_dataloader
from multiscale_sr.experiment import load_config, make_experiment_dir, save_config
from multiscale_sr.recovery import Diagnostics, save_checkpoint
from multiscale_sr.utils import resolve_env, seed_everything
from multiscale_sr.wandb_logger import WandbLogger, load_env

from multiscale_sr.pointcloud.collate import dense_pair_to_padded_hits
from multiscale_sr.pointcloud.hits import hits_to_dense
from multiscale_sr.pointcloud.losses import (
    channel_loss,
    energy_conservation_loss,
    existence_loss,
    masked_chamfer_loss,
    masked_sinkhorn_loss,
)
from multiscale_sr.pointcloud.models import SetDiscriminator, SetGenerator
from multiscale_sr.pointcloud.matching import matched_supervision

PACKAGE_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Stage 3 point-cloud SR model")
    p.add_argument("--config", type=str, default=None, help="YAML config; CLI flags override it")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--dataset-format", type=str, default=None, choices=["parquet", "hdf5"])
    p.add_argument("--scale", type=int, default=None, help="LR resolution: e.g. 64, 32, 16")
    p.add_argument("--hr-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32,
                    help="Smaller default than train.py's 64: hit lists are O(N) per sample "
                         "where N ~ hundreds, and Chamfer/BCE/CE terms are O(N^2) or O(B*N) per "
                         "step, a materially higher per-sample cost than a dense conv batch")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--d-lr-ratio", type=float, default=0.5)
    # LR/HR point-set budgets. Defaults come from the Stage 1 audit
    # (docs/POINTCLOUD_AUDIT.md): HR p99 hit count and roughly the 64x LR
    # equivalent; override per-scale, since coarser LR carries far fewer hits.
    p.add_argument("--lr-n-max", type=int, default=400,
                    help="Padding budget for LR hits (truncates the rest, see pad_hits)")
    p.add_argument("--hr-n-max", type=int, default=1300,
                    help="Generator output budget AND HR target padding budget "
                         "(suggested by the Stage 1 audit's p99 hit count)")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--lambda-chamfer", type=float, default=1.0,
                    help="Weight on masked_chamfer_loss (x,y,energy accuracy). Unlike Stage 2's "
                         "grid-pipeline flags, this is ON by default: it is a primary "
                         "reconstruction term here, not an optional add-on -- there is no L1 "
                         "equivalent on a variable-cardinality point cloud.")
    p.add_argument("--lambda-sinkhorn", type=float, default=0.0,
                    help="Weight on masked_sinkhorn_loss, blended alongside Chamfer (both may be "
                         "nonzero at once). Off by default: Sinkhorn is ~3x the compute of Chamfer "
                         "per step (iters x pairwise cost), so this is opt-in rather than replacing "
                         "Chamfer outright. See masked_sinkhorn_loss's docstring for why Chamfer's "
                         "many-to-one collapse (the diagnosed peak-smearing failure on a real "
                         "trained checkpoint -- predicted top-5 energies 10.9/9.2/7.4/6.2/6.1 vs "
                         "HR's 27.6/16.6/6.2/6.0/4.0) is a real risk Sinkhorn's joint assignment "
                         "resists better; try --lambda-sinkhorn 1.0 --lambda-chamfer 0.0 for a "
                         "Sinkhorn-only run, or a nonzero blend of both.")
    p.add_argument("--chamfer-energy-weighted", action="store_true", default=True,
                    help="Weight masked_chamfer_loss's target-side term by hit energy (see its "
                         "energy_weighted parameter). On by default -- this and "
                         "--lambda-energy-value are the fix for the diagnosed peak-smearing "
                         "failure; --no-chamfer-energy-weighted reproduces the pre-fix behavior "
                         "for comparison.")
    p.add_argument("--no-chamfer-energy-weighted", dest="chamfer_energy_weighted", action="store_false")
    p.add_argument("--chamfer-alpha", type=float, default=5.0,
                    help="Energy-weighting strength for --chamfer-energy-weighted, same default "
                         "and formula as the grid pipeline's --l1-weight-alpha")
    p.add_argument("--lambda-energy-value", type=float, default=1.0,
                    help="Per-matched-pair |pred_energy - target_energy| weight, added to both "
                         "masked_chamfer_loss and masked_sinkhorn_loss. This is the term that was "
                         "MISSING when the peak-smearing checkpoint was trained (position distance "
                         "alone has zero gradient on energy values) -- set to 0.0 to reproduce that "
                         "failure mode for comparison, but do not disable this for a real run.")
    p.add_argument("--sinkhorn-eps", type=float, default=4.0)
    p.add_argument("--sinkhorn-iters", type=int, default=30)
    p.add_argument("--lambda-existence", type=float, default=1.0,
                    help="Weight on existence_loss (does this output slot correspond to a real hit)")
    p.add_argument("--target-assignment", choices=["hungarian", "index"], default="hungarian",
                    help="Existence/channel target assignment; index reproduces the old baseline for ablation")
    p.add_argument("--lambda-channel", type=float, default=0.1,
                    help="Weight on channel_loss (predicted channel class cross-entropy)")
    p.add_argument("--lambda-energy", type=float, default=1.0,
                    help="Weight on energy_conservation_loss (re-rasterized SR vs HR total energy)")
    p.add_argument("--lambda-adv", type=float, default=0.1,
                    help="Peak adversarial weight; low default since the SetDiscriminator is a "
                         "newer, less-tuned component than the grid pipeline's PatchGAN")
    p.add_argument("--adv-warmup-epochs", type=int, default=3,
                    help="Same warmup rationale as train.py: let the generator learn geometry "
                         "before facing a discriminator (see docs/TRAINING_STABILITY.md)")
    p.add_argument("--adv-ramp-epochs", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.33)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="pointcloud_run")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--recovery-dir", default=None)
    p.add_argument("--run-fingerprint", default=None)
    p.add_argument("--checkpoint-seconds", type=float, default=300)
    p.add_argument("--experiments-root", type=str, default=str(PACKAGE_ROOT / "experiments_pointcloud"))
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--cache", action="store_true", default=True,
                    help="Reuses the grid pipeline's HR decode cache (see multiscale_sr/data/cache.py) "
                         "-- the LR/HR image pairs feeding hit conversion are the same ones train.py "
                         "uses, so this benefits from the same OOM fix (see cache.py's _FLUSH_BYTES)")
    p.add_argument("--no-cache", dest="cache", action="store_false")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default="multiscale-sr-pointcloud")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        cfg = load_config(Path(args.config))
        tokens = sys.argv[1:] if argv is None else argv
        explicit = {a.dest for a in parser._actions
                    if any(t.split("=", 1)[0] in a.option_strings for t in tokens)}
        for key, value in cfg.items():
            dest = key.replace("-", "_")
            if not hasattr(args, dest):
                continue
            if dest not in explicit:
                setattr(args, dest, value)
    if args.scale is None:
        parser.error("--scale is required (via CLI or --config)")
    if args.data_dir is None:
        parser.error("--data-dir is required (via CLI or --config)")
    for name in ("epochs", "batch_size", "lr_n_max", "hr_n_max", "d_model", "hidden",
                 "n_layers", "n_heads", "log_every", "sample_every"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.checkpoint_seconds <= 0:
        parser.error("checkpoint-seconds must be positive")
    return args


def _to_hits(lr_img: Tensor, hr_img: Tensor, args) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return dense_pair_to_padded_hits(lr_img, hr_img, args.lr_n_max, args.hr_n_max)


def _supervision_targets(pred_hits, hr_hits, hr_mask, args):
    if getattr(args, "target_assignment", "hungarian") == "index":
        return hr_hits[..., 0], hr_mask
    return matched_supervision(pred_hits, hr_hits, hr_mask)


@torch.no_grad()
def evaluate(generator, loader, device, args, max_batches=None) -> dict[str, float]:
    """Point-cloud analogue of engine.evaluate: geometric + energy metrics,
    computed the same way regardless of which loss terms are enabled, so
    comparisons accumulate across all runs (same rationale train.py/engine.py
    document for val_semd) -- in particular, val_chamfer AND val_sinkhorn are
    both always computed here even though only one may be the active training
    loss, so a Chamfer-trained and a Sinkhorn-trained checkpoint remain
    comparable on both metrics."""
    generator.eval()
    chamfer_sum = sinkhorn_sum = exist_sum = channel_sum = energy_sum = 0.0
    n_batches = 0
    for batch in loader:
        lr_img = batch["lr"].to(device)
        hr_img = batch["hr"].to(device)
        lr_hits, lr_mask, hr_hits, hr_mask = _to_hits(lr_img, hr_img, args)

        pred_hits, exist_logits, chan_logits = generator(
            lr_hits, lr_mask, args.scale, args.hr_size,
        )
        chamfer_sum += masked_chamfer_loss(
            pred_hits, hr_hits, hr_mask, energy_weighted=args.chamfer_energy_weighted,
            alpha=args.chamfer_alpha, lambda_energy_value=args.lambda_energy_value,
        ).item()
        sinkhorn_sum += masked_sinkhorn_loss(
            pred_hits, hr_hits, hr_mask, eps=args.sinkhorn_eps, iters=args.sinkhorn_iters,
            lambda_energy_value=args.lambda_energy_value,
        ).item()
        target_channel, target_exists = _supervision_targets(pred_hits, hr_hits, hr_mask, args)
        exist_sum += existence_loss(exist_logits, target_exists).item()
        channel_sum += channel_loss(chan_logits, target_channel, target_exists).item()

        sr_img = hits_to_dense(
            _existing_hits_only(pred_hits, exist_logits), tuple(hr_img.shape)
        )
        energy_sum += energy_conservation_loss(sr_img, hr_img).item()

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    if not n_batches:
        raise ValueError("Validation loader produced no samples")
    n = n_batches
    return {
        "chamfer": chamfer_sum / n, "sinkhorn": sinkhorn_sum / n, "existence": exist_sum / n,
        "channel": channel_sum / n, "energy": energy_sum / n,
    }


def _existing_hits_only(pred_hits: Tensor, exist_logits: Tensor, threshold: float = 0.0) -> Tensor:
    """Flatten a padded (B, N, 4) prediction to a flat (M,5) hit list, keeping
    only slots whose existence logit clears `threshold` (0.0 = sigmoid>0.5).
    This is the ONE place a hard existence decision is made; training losses
    (masked_chamfer_loss, existence_loss) never threshold, only this
    re-rasterization / final-output path does."""
    b, n, _ = pred_hits.shape
    keep = exist_logits > threshold
    batch_idx = torch.arange(b, device=pred_hits.device).unsqueeze(1).expand(b, n)
    flat = pred_hits[keep]
    flat_batch = batch_idx[keep].to(pred_hits.dtype)
    return torch.cat([flat_batch.unsqueeze(1), flat], dim=1)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    load_env(PACKAGE_ROOT)
    env = resolve_env()
    if args.recovery_dir or args.resume:
        from dataclasses import replace
        env = replace(env, num_workers=0, persistent_workers=False, prefetch_factor=None)
    print(f"[env] {env}")

    data_dir = Path(args.data_dir)
    dataset_type = args.dataset_format or detect_dataset_type(data_dir)
    dataset_name = data_dir.name if data_dir.name else dataset_type
    print(f"[data] format={dataset_type} dir={data_dir} scale={args.scale} hr={args.hr_size}")
    print(f"[pointcloud] lr_n_max={args.lr_n_max} hr_n_max={args.hr_n_max} "
          "(see docs/POINTCLOUD_AUDIT.md for how these were chosen)")

    paths = make_experiment_dir(
        Path(args.experiments_root), dataset_name, args.scale, args.run_name,
        run_dir=Path(args.run_dir) if args.run_dir else None, resume=bool(args.resume),
    )
    print(f"[exp] {paths.root}")
    recovery_dir = Path(args.recovery_dir) if args.recovery_dir else None
    diagnostics = Diagnostics(paths.root, env.device, recovery_dir).start()
    _ACTIVE_RESOURCES.append(diagnostics)

    config = vars(args).copy()
    config.update({"dataset_type": dataset_type, "device": str(env.device)})
    save_config(paths.config_yaml, config)

    diagnostics.phase("data_and_normalization", seed=args.seed)
    # Stats are computed for API compatibility with get_dataloader but never
    # applied: hit coordinates/energies are normalized inside
    # PermInvariantEncoder (log1p energy, [-1,1] coords) rather than via the
    # grid pipeline's ChannelStats z-score, since a point cloud has no fixed
    # per-channel pixel grid to compute a mean/std over in the same sense.
    train_loader, _ = get_dataloader(
        path=data_dir, split="train", env=env, batch_size=args.batch_size,
        scale=args.scale, hr_size=args.hr_size, dataset_type=dataset_type,
        val_ratio=args.val_ratio, use_cache=args.cache,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    val_loader, _ = get_dataloader(
        path=data_dir, split="val", env=env, batch_size=args.batch_size,
        scale=args.scale, hr_size=args.hr_size, dataset_type=dataset_type,
        val_ratio=args.val_ratio, use_cache=args.cache,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    generator = SetGenerator(
        hr_n_max=args.hr_n_max, d_model=args.d_model, hidden=args.hidden,
        n_layers=args.n_layers, n_heads=args.n_heads,
    ).to(env.device)
    discriminator = SetDiscriminator(d_model=args.d_model, hidden=args.hidden).to(env.device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr * args.d_lr_ratio, betas=(0.5, 0.999))

    start_epoch = 1
    best_val = math.inf
    history: list[dict] = []
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        generator.load_state_dict(ckpt["generator"], strict=True)
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["optimizer_g"])
        opt_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", math.inf)
        history = ckpt.get("history", [])
        global_step = ckpt.get("global_step", 0)
        print(f"[resume] from {args.resume} at epoch {start_epoch}")

    wb = WandbLogger(
        enabled=not args.no_wandb, project=args.wandb_project, entity=args.wandb_entity,
        run_name=f"{dataset_name}_{args.scale}x_{args.run_name}_pointcloud", config=config,
        dir=paths.root,
    )
    _ACTIVE_RESOURCES.append(wb)

    last_checkpoint = time.monotonic()

    def checkpoint(epoch, best=False):
        nonlocal last_checkpoint
        payload = {
            "recovery_version": 1, "epoch": epoch, "history": history,
            "global_step": global_step, "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(),
            "args": vars(args), "best_val": best_val,
        }
        diagnostics.phase("checkpoint", epoch=epoch, step=global_step)
        if best:
            save_checkpoint(payload, paths.best_ckpt, recovery_dir)
        save_checkpoint(payload, paths.latest_ckpt, recovery_dir)
        last_checkpoint = time.monotonic()

    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()
        e = epoch - 1
        adv_w = (0.0 if e < args.adv_warmup_epochs else
                 args.lambda_adv if args.adv_ramp_epochs <= 0 else
                 args.lambda_adv * min(1.0, (e - args.adv_warmup_epochs + 1) / args.adv_ramp_epochs))
        adv_active = adv_w > 0.0

        chamfer_run = sinkhorn_run = exist_run = channel_run = energy_run = g_run = d_run = 0.0
        seen = 0
        diagnostics.phase("training", epoch=epoch, step=global_step)

        train_iter = train_loader
        if args.max_train_batches is not None:
            from itertools import islice
            train_iter = islice(train_loader, args.max_train_batches)

        for step, batch in enumerate(train_iter, start=1):
            lr_img = batch["lr"].to(env.device)
            hr_img = batch["hr"].to(env.device)
            lr_hits, lr_mask, hr_hits, hr_mask = _to_hits(lr_img, hr_img, args)
            bs = lr_img.shape[0]

            pred_hits, exist_logits, chan_logits = generator(
                lr_hits, lr_mask, args.scale, args.hr_size,
            )

            if adv_active:
                discriminator.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                real_logits = discriminator(lr_hits, lr_mask, args.scale, hr_hits, hr_mask, args.hr_size)
                pred_mask_for_disc = torch.sigmoid(exist_logits.detach()) > 0.5
                fake_logits_d = discriminator(
                    lr_hits, lr_mask, args.scale, pred_hits.detach(), pred_mask_for_disc, args.hr_size,
                )
                d_loss = 0.5 * ((real_logits - 0.9) ** 2 + fake_logits_d ** 2).mean()
                if not torch.isfinite(d_loss).item():
                    raise FloatingPointError(f"Non-finite discriminator loss at epoch {epoch}, batch {step}")
                d_loss.backward()
                opt_d.step()
                d_val = d_loss.item()
            else:
                d_val = 0.0

            opt_g.zero_grad(set_to_none=True)
            # chamfer/sinkhorn both take lambda_energy_value: the per-pair
            # |pred_energy - target_energy| term that fixes the peak-smearing
            # failure (see masked_chamfer_loss's docstring for the real
            # trained checkpoint that motivated it). Guarded like lambda_semd:
            # Sinkhorn is skipped entirely (not computed-then-multiplied-by-0)
            # when its weight is 0, since its iters x pairwise-cost loop is
            # real added cost per step, not a free extra metric.
            chamfer_val = sinkhorn_val = 0.0
            g_loss = None
            if args.lambda_chamfer > 0.0:
                chamfer = masked_chamfer_loss(
                    pred_hits, hr_hits, hr_mask, energy_weighted=args.chamfer_energy_weighted,
                    alpha=args.chamfer_alpha, lambda_energy_value=args.lambda_energy_value,
                )
                chamfer_val = chamfer.item()
                g_loss = args.lambda_chamfer * chamfer
            if args.lambda_sinkhorn > 0.0:
                sinkhorn = masked_sinkhorn_loss(
                    pred_hits, hr_hits, hr_mask, eps=args.sinkhorn_eps, iters=args.sinkhorn_iters,
                    lambda_energy_value=args.lambda_energy_value,
                )
                sinkhorn_val = sinkhorn.item()
                term = args.lambda_sinkhorn * sinkhorn
                g_loss = term if g_loss is None else g_loss + term
            if g_loss is None:
                raise ValueError(
                    "At least one of --lambda-chamfer/--lambda-sinkhorn must be > 0: "
                    "with both at 0 there is no position/energy reconstruction term at all."
                )

            target_channel, target_exists = _supervision_targets(pred_hits, hr_hits, hr_mask, args)
            exist = existence_loss(exist_logits, target_exists)
            channel = channel_loss(chan_logits, target_channel, target_exists)
            sr_img = hits_to_dense(_existing_hits_only(pred_hits, exist_logits), tuple(hr_img.shape))
            energy = energy_conservation_loss(sr_img, hr_img)

            g_loss = (g_loss + args.lambda_existence * exist
                      + args.lambda_channel * channel + args.lambda_energy * energy)
            if adv_active:
                discriminator.requires_grad_(False)
                pred_mask_for_g = torch.sigmoid(exist_logits.detach()) > 0.5
                fake_logits_g = discriminator(lr_hits, lr_mask, args.scale, pred_hits, pred_mask_for_g, args.hr_size)
                g_loss = g_loss + adv_w * 0.5 * ((fake_logits_g - 1.0) ** 2).mean()

            if not torch.isfinite(g_loss).item():
                raise FloatingPointError(f"Non-finite generator loss at epoch {epoch}, batch {step}")
            g_loss.backward()
            opt_g.step()

            global_step += 1
            seen += bs
            chamfer_run += chamfer_val * bs
            sinkhorn_run += sinkhorn_val * bs
            exist_run += exist.item() * bs
            channel_run += channel.item() * bs
            energy_run += energy.item() * bs
            g_run += g_loss.item() * bs
            d_run += d_val * bs
            diagnostics.state.update(epoch=epoch, batch=step, step=global_step)

            if global_step % args.log_every == 0:
                wb.log({"step/g_loss": g_loss.item(), "step/d_loss": d_val,
                         "step/chamfer": chamfer.item(), "step/adv_weight": adv_w,
                         "epoch": epoch}, step=global_step)

            if time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                checkpoint(epoch)
                diagnostics.phase("training", epoch=epoch, step=global_step)

        if not seen:
            raise ValueError("Training loader produced no samples")
        denom = seen
        train_metrics = {
            "epoch": epoch, "train_g_loss": g_run / denom, "train_d_loss": d_run / denom,
            "train_chamfer": chamfer_run / denom, "train_sinkhorn": sinkhorn_run / denom,
            "train_existence": exist_run / denom,
            "train_channel": channel_run / denom, "train_energy": energy_run / denom,
            "adv_weight": adv_w,
        }
        diagnostics.phase("validation", epoch=epoch)
        val_metrics = evaluate(generator, val_loader, env.device, args, max_batches=args.max_val_batches)
        record = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        print(
            f"epoch {epoch:03d} g={record['train_g_loss']:.4f} d={record['train_d_loss']:.4f} "
            f"chamfer={record['train_chamfer']:.4f} sinkhorn={record['train_sinkhorn']:.4f} "
            f"exist={record['train_existence']:.4f} "
            f"channel={record['train_channel']:.4f} energy={record['train_energy']:.4f} "
            f"val_chamfer={record['val_chamfer']:.4f} val_sinkhorn={record['val_sinkhorn']:.4f} "
            f"val_energy={record['val_energy']:.4f} "
            f"advw={adv_w:.2f}"
        )
        with paths.metrics_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        wb.log(record, step=global_step)

        is_best = val_metrics["chamfer"] < best_val
        if is_best:
            best_val = val_metrics["chamfer"]
        checkpoint(epoch, best=is_best)
        if is_best:
            wb.log_artifact(paths.best_ckpt, name=f"model-{args.run_name}", aliases=["best"])

    diagnostics.phase("final_validation")
    final_eval = evaluate(generator, val_loader, env.device, args, max_batches=args.max_val_batches)
    final_eval.update({"best_val_chamfer": best_val, "scale": args.scale, "dataset": dataset_name})
    paths.eval_json.write_text(json.dumps(final_eval, indent=2), encoding="utf-8")
    wb.set_summary({"best_val_chamfer": best_val, "scale": args.scale, "dataset": dataset_name})
    wb.finish()
    print(f"[done] best val chamfer = {best_val:.5f} | eval -> {paths.eval_json}")


_ACTIVE_RESOURCES: list = []


def main() -> None:
    try:
        train(resolve_args())
    finally:
        for resource in reversed(_ACTIVE_RESOURCES):
            try:
                resource.close() if hasattr(resource, "close") else resource.finish()
            except Exception:
                pass
        _ACTIVE_RESOURCES.clear()


if __name__ == "__main__":
    main()
