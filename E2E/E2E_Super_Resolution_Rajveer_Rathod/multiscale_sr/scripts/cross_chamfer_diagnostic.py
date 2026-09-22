"""Cross-Chamfer conditioning diagnostic for point-cloud SR checkpoints.

**Why this exists (and what it replaces).** An earlier conditioning check
rasterized SR and HR hit lists to dense images and compared them with cosine
similarity. On ~98.5%-empty data (docs/POINTCLOUD_AUDIT.md) that metric is
categorically mismatched to the question: two point sets differing by only a
small positional shift per hit share NO exact pixel coincidence, so sparse
pixel-overlap cosine reads 0.0 for sets that are structurally near-identical.
It was therefore near-guaranteed to report "conditioning failure" regardless
of how well-conditioned the model actually was.

This script instead measures conditioning with the metric the model is
actually trained under: ``masked_chamfer_loss``, which operates on point
DISTANCE rather than pixel overlap. For a batch of B samples it builds the
full (B,B) cross matrix ``masked_chamfer_loss(SR[i], HR[j], HR_mask[j])`` and
asks the only question that matters for conditioning:

    does SR[i] identify HR[i] as its closest target, above chance?

Reported: mean diagonal (own-HR) vs mean off-diagonal (other-HR) Chamfer, and
the count of rows whose argmin is the own-HR column. Chance is 1/B rows.

**Architecture generations.** Checkpoints predating the Stage-B query fix
store a plain ``queries`` nn.Parameter and will NOT load into the current
``SetGenerator`` (which has ``query_pos_mlp``/``repeat_perturbation``
instead). ``StageAGenerator`` below reconstructs that earlier class so a true
apples-to-apples comparison across generations is possible; it is a frozen
copy for loading historical checkpoints ONLY, never for training. The
generation is auto-detected from the checkpoint's state-dict keys.

Usage:
    python scripts/cross_chamfer_diagnostic.py \
        --checkpoint <run>/checkpoints/best.pt [--checkpoint <run2>/...] \
        --data-dir /path/to/datasets_20pct --batch-size 16 --seeds 0 1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import Tensor, nn

# Same import bootstrap scripts/pointcloud_audit.py uses: this file is run as
# a script from the project root, not installed as a package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multiscale_sr.data import get_dataloader  # noqa: E402
from multiscale_sr.utils import resolve_env, seed_everything  # noqa: E402

from multiscale_sr.pointcloud.collate import dense_pair_to_padded_hits  # noqa: E402
from multiscale_sr.pointcloud.losses import masked_chamfer_loss  # noqa: E402
from multiscale_sr.pointcloud.models import PermInvariantEncoder, SetGenerator  # noqa: E402


class StageAGenerator(nn.Module):
    """Pre-Stage-B ``SetGenerator``: full-length attention memory (Stage A),
    but input-INDEPENDENT learned queries.

    Reconstructed verbatim from the Stage A generation of
    ``multiscale_sr/pointcloud/models.py`` so its checkpoints remain loadable
    after Stage B replaced ``self.queries`` with ``_build_queries``. Load-only
    for historical comparison; not exported from the package and not wired
    into training.
    """

    def __init__(
        self, hr_n_max: int, d_model: int = 128, hidden: int = 256,
        n_layers: int = 3, n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hr_n_max = hr_n_max
        self.d_model = d_model
        self.encoder = PermInvariantEncoder(d_model=d_model, hidden=hidden)
        self.summary_proj = nn.Linear(2 * d_model, d_model)
        self.queries = nn.Parameter(torch.randn(hr_n_max, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=hidden,
            batch_first=True, dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.n_channels = 3
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 4 + self.n_channels),
        )

    def forward(self, lr_hits: Tensor, lr_mask: Tensor, lr_grid: int, hr_grid: int):
        b = lr_hits.shape[0]
        pooled, point_emb = self.encoder(lr_hits, lr_mask, lr_grid, return_points=True)
        summary_tok = self.summary_proj(pooled).unsqueeze(1)
        memory = torch.cat([point_emb, summary_tok], dim=1)
        summary_valid = torch.ones(b, 1, dtype=torch.bool, device=lr_mask.device)
        memory_key_padding_mask = ~torch.cat([lr_mask.bool(), summary_valid], dim=1)
        queries = self.queries.unsqueeze(0).expand(b, -1, -1)
        decoded = self.decoder(
            tgt=queries, memory=memory, memory_key_padding_mask=memory_key_padding_mask,
        )
        raw = self.head(decoded)
        x = torch.sigmoid(raw[..., 0]) * (hr_grid - 1)
        y = torch.sigmoid(raw[..., 1]) * (hr_grid - 1)
        energy = torch.nn.functional.softplus(raw[..., 2])
        existence_logits = raw[..., 3]
        channel_logits = raw[..., 4:]
        channel = channel_logits.argmax(dim=-1).to(raw.dtype).detach()
        hits = torch.stack([channel, x, y, energy], dim=-1)
        return hits, existence_logits, channel_logits


def load_generator(path: str, device) -> tuple[nn.Module, dict, str]:
    """Load a checkpoint into whichever generator generation wrote it.

    Generation is detected from state-dict keys rather than from a version
    field, because these checkpoints predate any such field.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ck = ckpt["args"]
    sd = ckpt["generator"]
    stage_b = any(k.startswith("query_pos_mlp") for k in sd)
    cls = SetGenerator if stage_b else StageAGenerator
    gen = cls(
        hr_n_max=ck["hr_n_max"], d_model=ck["d_model"], hidden=ck["hidden"],
        n_layers=ck["n_layers"], n_heads=ck["n_heads"],
    ).to(device)
    # strict=True: a silent key mismatch here would mean comparing a
    # partially-random model, which is exactly the kind of quiet wrong answer
    # this diagnostic exists to avoid.
    gen.load_state_dict(sd, strict=True)
    gen.eval()
    return gen, ck, ("StageB+" if stage_b else "StageA")


@torch.no_grad()
def cross_chamfer(
    gen: nn.Module, ck: dict, loader, device, batch_size: int, skip: int = 0,
) -> tuple[Tensor, int]:
    """Return the (B,B) cross-Chamfer matrix M[i,j] = chamfer(SR[i], HR[j]).

    ``skip`` advances ``skip`` batches into the stream before measuring. The
    test split is deterministic and unshuffled (see data/factory.get_dataloader),
    so reseeding does NOT produce a different batch -- taking a different
    POSITION in the stream is what makes a second reading independent.
    """
    it = iter(loader)
    for _ in range(skip):
        next(it)
    batch = next(it)
    lr_img = batch["lr"][:batch_size].to(device)
    hr_img = batch["hr"][:batch_size].to(device)
    lr_hits, lr_mask, hr_hits, hr_mask = dense_pair_to_padded_hits(
        lr_img, hr_img, ck["lr_n_max"], ck["hr_n_max"],
    )
    pred_hits, _exist, _chan = gen(lr_hits, lr_mask, ck["scale"], ck["hr_size"])
    n = pred_hits.shape[0]
    m = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            m[i, j] = masked_chamfer_loss(
                pred_hits[i : i + 1], hr_hits[j : j + 1], hr_mask[j : j + 1],
            ).item()
    return m, n


def report(m: Tensor, batch_idx: int) -> dict:
    n = m.shape[0]
    diag = m.diagonal()
    off = m[~torch.eye(n, dtype=torch.bool)]
    matches = int((m.argmin(dim=1) == torch.arange(n)).sum())
    own, other = diag.mean().item(), off.mean().item()
    margin = (other - own) / other * 100 if other else float("nan")
    print(f"  batch #{batch_idx}  own-HR chamfer={own:8.3f}  other-HR={other:8.3f}  "
          f"margin={margin:5.1f}%  argmin-is-own: {matches}/{n} (chance ~1/{n})")
    return {"batch": batch_idx, "own": own, "other": other, "matches": matches, "n": n}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True,
                   help="Repeatable; each is evaluated and reported separately")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.33)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batches", type=int, nargs="+", default=[0, 1, 2],
                   help="Stream positions to measure, each a disjoint set of "
                        "jets; a finding that holds on only one batch is not "
                        "a finding")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = resolve_env()
    seed_everything(args.seed)
    print(f"[env] {env}")

    for path in args.checkpoint:
        gen, ck, gen_name = load_generator(path, env.device)
        print(f"\n=== {path}")
        print(f"    architecture={gen_name} scale={ck['scale']} hr_n_max={ck['hr_n_max']}")
        loader, _ = get_dataloader(
            path=Path(args.data_dir), split="test", env=env,
            batch_size=args.batch_size, scale=ck["scale"],
            hr_size=ck["hr_size"], val_ratio=args.val_ratio,
        )
        for b in args.batches:
            m, _n = cross_chamfer(gen, ck, loader, env.device, args.batch_size, skip=b)
            report(m, b)


if __name__ == "__main__":
    main()
