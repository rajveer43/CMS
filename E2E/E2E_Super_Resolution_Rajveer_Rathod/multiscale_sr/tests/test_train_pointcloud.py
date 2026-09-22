"""Real-subprocess smoke tests for train_pointcloud.py, matching the pattern
in tests/test_data_and_sweep.py::test_full_sweep_cli_smoke."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


def make_parquet(path, n=32, side=16):
    rng = np.random.default_rng(4)
    hr = rng.random((n, 3, side, side), dtype=np.float32)
    table = pa.table({
        "X_jets_LR": hr[:, :, ::2, ::2].tolist(), "X_jets": hr.tolist(),
        "pt": list(range(n)), "m0": [1.0] * n, "y": [(i // 2) % 2 for i in range(n)],
    })
    pq.write_table(table, path)


def _env(tmp_path):
    return {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "MULTISCALE_SR_DEVICE": "cpu", "MULTISCALE_SR_NUM_WORKERS": "0",
            "MPLCONFIGDIR": str(tmp_path / "mpl")}


def _project_root():
    return Path(__file__).resolve().parent.parent


def test_train_pointcloud_cli_smoke_runs_and_checkpoints(tmp_path):
    """Real subprocess: parquet -> hit conversion -> generator/discriminator
    training -> checkpoint -> eval.json, on the actual CLI entry point."""
    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        make_parquet(data / f"{i}.parquet", n=24, side=16)

    run_dir = tmp_path / "run"
    cmd = [
        sys.executable, "train_pointcloud.py",
        "--data-dir", str(data), "--scale", "8", "--hr-size", "16",
        "--epochs", "1", "--seed", "123", "--run-name", "pytest_smoke",
        "--experiments-root", str(tmp_path / "experiments"),
        "--run-dir", str(run_dir),
        "--cache-dir", str(tmp_path / "cache"),
        "--max-train-batches", "2", "--max-val-batches", "1", "--batch-size", "4",
        "--lr-n-max", "20", "--hr-n-max", "40",
        "--d-model", "8", "--hidden", "16", "--n-layers", "1", "--n-heads", "2",
        "--adv-warmup-epochs", "0", "--no-wandb",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=_env(tmp_path),
        cwd=_project_root(), timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    eval_json = run_dir / "eval.json"
    assert eval_json.exists(), result.stdout + result.stderr
    payload = json.loads(eval_json.read_text())
    for key in ("chamfer", "existence", "channel", "energy", "best_val_chamfer"):
        assert key in payload
        assert np.isfinite(payload[key])

    ckpt_dir = run_dir / "checkpoints"
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    ckpt = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=True)
    assert ckpt["recovery_version"] == 1
    assert ckpt["epoch"] == 1
    assert set(ckpt.keys()) >= {"generator", "discriminator", "optimizer_g", "optimizer_d", "history"}

    metrics = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics) == 1
    record = json.loads(metrics[0])
    assert record["epoch"] == 1
    for key in ("train_chamfer", "train_existence", "train_channel", "train_energy",
                "val_chamfer", "val_energy"):
        assert key in record and np.isfinite(record[key])


def test_train_pointcloud_resume_continues_from_checkpoint(tmp_path):
    """A second invocation with --resume must start at epoch+1 and keep
    accumulating global_step/history rather than restarting."""
    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        make_parquet(data / f"{i}.parquet", n=24, side=16)

    base_cmd = [
        sys.executable, "train_pointcloud.py",
        "--data-dir", str(data), "--scale", "8", "--hr-size", "16",
        "--seed", "123", "--run-name", "pytest_resume",
        "--experiments-root", str(tmp_path / "experiments"),
        "--cache-dir", str(tmp_path / "cache"),
        "--max-train-batches", "2", "--max-val-batches", "1", "--batch-size", "4",
        "--lr-n-max", "20", "--hr-n-max", "40",
        "--d-model", "8", "--hidden", "16", "--n-layers", "1", "--n-heads", "2",
        "--adv-warmup-epochs", "0", "--no-wandb",
    ]

    run1 = tmp_path / "run1"
    first = subprocess.run(
        base_cmd + ["--epochs", "1", "--run-dir", str(run1)],
        capture_output=True, text=True, env=_env(tmp_path), cwd=_project_root(), timeout=180,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    ckpt1 = torch.load(run1 / "checkpoints" / "latest.pt", map_location="cpu", weights_only=True)
    assert ckpt1["epoch"] == 1
    step_after_epoch_1 = ckpt1["global_step"]

    run2 = tmp_path / "run2"
    second = subprocess.run(
        base_cmd + ["--epochs", "2", "--run-dir", str(run2),
                    "--resume", str(run1 / "checkpoints" / "latest.pt")],
        capture_output=True, text=True, env=_env(tmp_path), cwd=_project_root(), timeout=180,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "[resume] from" in second.stdout

    ckpt2 = torch.load(run2 / "checkpoints" / "latest.pt", map_location="cpu", weights_only=True)
    assert ckpt2["epoch"] == 2
    assert ckpt2["global_step"] > step_after_epoch_1
    assert len(ckpt2["history"]) == 2  # epoch 1's record carried over + epoch 2's
