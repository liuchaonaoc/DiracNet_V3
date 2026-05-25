#!/usr/bin/env python3
"""Stage A training (Dirac PINN only — no NIST gradient)."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.losses.loss_schedule import stage_a_weights
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params, save_checkpoint
from pinn_art.training.stage_a_trainer import train_step
from pinn_art.training.train_state import create_train_state
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.utils.logging import get_logger


def _ensure_manifest(manifest: Path, cfg_path: Path) -> None:
    if manifest.exists():
        return
    log = get_logger()
    log.info("Manifest missing, running v3_prepare_hydrogenic.py ...")
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "v3_prepare_hydrogenic.py"),
            "--z-min", "1",
            "--z-max", "8",
            "--n-levels", "6",
            "--out", str(manifest),
        ],
        cwd=str(ROOT),
    )


def _epoch_indices(n: int, steps: int, batch_size: int, rng: np.random.Generator, shuffle: bool) -> list[list[int]]:
    order = np.arange(n)
    if shuffle:
        rng.shuffle(order)
    idx_lists = []
    for step in range(steps):
        start = (step * batch_size) % n
        idx = [(start + i) % n for i in range(batch_size)]
        idx_lists.append(idx)
    return idx_lists


def main():
    ap = argparse.ArgumentParser(description="PINN-ART Stage A trainer")
    ap.add_argument("--config", default="configs/v3_phase1_stage_a_z1_8.yaml")
    ap.add_argument("--resume", default=None, help="Path to stage_a_last.msgpack params")
    ap.add_argument("--prepare-data", action="store_true", help="Force regenerate manifest")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    log = get_logger()
    log.info("JAX backend=%s  devices=%s", jax.default_backend(), jax.devices())

    manifest = ROOT / cfg.dataset.manifest
    if args.prepare_data or not manifest.exists():
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "v3_prepare_hydrogenic.py"),
                "--z-min", "1",
                "--z-max", "8",
                "--n-levels", "6",
                "--out", str(manifest),
            ],
            cwd=str(ROOT),
        )
    _ensure_manifest(manifest, ROOT / args.config)

    ds = ManifestDataset(
        manifest,
        n_orb_max=int(cfg.model.n_orb_max),
        n_csf_max=int(getattr(cfg.model, "n_csf_max", 8)),
    )
    grid = make_radial_grid(
        float(cfg.grid.r_min),
        float(cfg.grid.r_max),
        int(cfg.grid.n_grid),
        str(cfg.grid.scheme),
    )

    key = jax.random.PRNGKey(int(cfg.seed))
    model, params = build_model_and_params(cfg, grid, key)
    resume_path = args.resume
    if resume_path is None:
        resume_path = getattr(cfg.training, "resume_from", None)
    if resume_path:
        resume_path = ROOT / resume_path
        params = load_params(resume_path)
        log.info("Resumed params from %s", resume_path)

    state = create_train_state(model, params, cfg, total_steps=100000)
    weights = stage_a_weights(cfg)

    n_epochs = int(cfg.stage_a.n_epochs)
    steps = int(cfg.training.steps_per_epoch)
    bs = int(cfg.stage_a.batch_size)
    shuffle = bool(getattr(cfg.training, "shuffle_each_epoch", True))
    ckpt_every = int(getattr(cfg.training, "ckpt_every_epochs", 0))
    val_every = int(getattr(cfg.training, "val_every_epochs", 25))

    ckpt_dir = ROOT / cfg.training.ckpt_dir
    log_dir = ROOT / cfg.training.log_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / "history.csv"

    write_header = not history_path.exists()
    rng_np = np.random.default_rng(int(cfg.seed))

    log.info(
        "Stage A: %d samples, %d epochs x %d steps, batch=%d, manifest=%s",
        len(ds), n_epochs, steps, bs, manifest,
    )

    warm_idx = list(range(min(bs, len(ds))))
    warm_batch = collate_batches([ds[i] for i in warm_idx], n_csf_max=int(cfg.model.n_csf_max))
    log.info("Compiling JIT train_step on %s (first call traces XLA, ~15-25s)...", jax.default_backend())
    t_compile = time.perf_counter()
    state, _ = train_step(state, warm_batch, grid, weights)
    jax.block_until_ready(state.params)
    compile_sec = time.perf_counter() - t_compile
    log.info("JIT compile done in %.1fs (steady-state ~0.05-0.2s/step expected)", compile_sec)

    with history_path.open("a", newline="") as hf:
        writer = csv.DictWriter(
            hf,
            fieldnames=["epoch", "loss", "pde", "ortho", "asym", "norm", "v_prior", "v_smooth"],
        )
        if write_header:
            writer.writeheader()

        for epoch in range(n_epochs):
            t_epoch = time.perf_counter()
            epoch_loss = 0.0
            last_metrics = {}
            for step_idx in range(steps):
                key, sub = jax.random.split(key)
                if shuffle:
                    perm = jax.random.permutation(sub, len(ds))
                    idx = [int(perm[i % len(ds)]) for i in range(bs)]
                else:
                    idx = [(step_idx * bs + i) % len(ds) for i in range(bs)]
                items = [ds[i] for i in idx]
                batch = collate_batches(items, n_csf_max=int(cfg.model.n_csf_max))
                state, metrics = train_step(state, batch, grid, weights)
                epoch_loss += float(metrics["loss"])
                last_metrics = {k: float(v) for k, v in metrics.items()}

            row = {
                "epoch": epoch,
                "loss": epoch_loss / steps,
                **{k: last_metrics.get(k, 0.0) for k in ["pde", "ortho", "asym", "norm", "v_prior", "v_smooth"]},
            }
            writer.writerow(row)
            hf.flush()

            epoch_sec = time.perf_counter() - t_epoch
            sec_per_step = epoch_sec / max(steps, 1)
            if (epoch + 1) % max(val_every, 1) == 0 or epoch < 3:
                log.info(
                    "epoch %4d  loss=%.4f  pde=%.4f  ortho=%.4e  norm=%.4f  "
                    "time=%.1fs (%.2fs/step)",
                    epoch,
                    row["loss"],
                    row["pde"],
                    row["ortho"],
                    row["norm"],
                    epoch_sec,
                    sec_per_step,
                )
            elif sec_per_step > 3.0:
                log.warning(
                    "epoch %4d slow: %.2fs/step — if backend=cpu, reinstall jax[cuda12]",
                    epoch,
                    sec_per_step,
                )

            if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
                ep_ckpt = ckpt_dir / f"epoch_{epoch + 1:04d}.msgpack"
                save_checkpoint(ep_ckpt, state, {"epoch": epoch + 1})

    ckpt = ckpt_dir / "stage_a_last.msgpack"
    save_checkpoint(ckpt, state, {"epoch": n_epochs, "manifest": str(manifest)})
    log.info("Training done. checkpoint=%s  history=%s", ckpt, history_path)


if __name__ == "__main__":
    main()
