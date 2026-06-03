#!/usr/bin/env python3
"""Stage B training — CI Slater calibration (freeze DeepONet from Stage A)."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from pinn_art.ci.racah_cache import RacahCache
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.losses.loss_schedule import stage_b_weights
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params, merge_params, save_checkpoint
from pinn_art.training.stage_b_trainer import create_stage_b_train_state, train_step
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.utils.logging import get_logger


def main():
    ap = argparse.ArgumentParser(description="PINN-ART Stage B trainer")
    ap.add_argument("--config", default="configs/v3_phase1_stage_b_z1_8.yaml")
    ap.add_argument("--resume", default=None, help="Stage A or Stage B checkpoint")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    log = get_logger()
    log.info("JAX backend=%s  devices=%s", jax.default_backend(), jax.devices())

    manifest = ROOT / cfg.dataset.manifest
    if not manifest.exists():
        log.error("Manifest missing: %s — run Stage A data prep first.", manifest)
        sys.exit(1)

    racah_path = ROOT / cfg.ci.racah_cache
    if not racah_path.exists():
        log.info("Racah cache missing, building ...")
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "v3_build_racah_cache.py"),
                "--manifest",
                str(manifest.relative_to(ROOT)),
                "--out",
                str(racah_path.relative_to(ROOT)),
                "--config",
                args.config,
            ],
            cwd=str(ROOT),
        )
    racah_cache = RacahCache.load(racah_path)
    k_list = tuple(int(k) for k in cfg.ci.k_list)

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

    resume_path = args.resume or getattr(cfg.training, "resume_from", None)
    if resume_path:
        resume_path = ROOT / resume_path
        loaded = load_params(resume_path)
        params = merge_params(params, loaded)
        log.info("Merged checkpoint into Stage B params: %s", resume_path)

    state = create_stage_b_train_state(model, params, cfg)
    weights = stage_b_weights(cfg)

    n_epochs = int(cfg.stage_b.n_epochs)
    steps = int(cfg.training.steps_per_epoch)
    bs = int(cfg.stage_b.batch_size)
    shuffle = bool(getattr(cfg.training, "shuffle_each_epoch", True))
    ckpt_every = int(getattr(cfg.training, "ckpt_every_epochs", 0))
    val_every = int(getattr(cfg.training, "val_every_epochs", 20))

    ckpt_dir = ROOT / cfg.training.ckpt_dir
    log_dir = ROOT / cfg.training.log_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / "history.csv"

    def collate_fn(items):
        return collate_batches(
            items,
            n_csf_max=int(cfg.model.n_csf_max),
            racah_cache=racah_cache,
            k_list=k_list,
        )

    write_header = not history_path.exists()
    rng_np = np.random.default_rng(int(cfg.seed))

    log.info(
        "Stage B: %d samples, %d epochs x %d steps, batch=%d, trainable=slater_log_scale",
        len(ds),
        n_epochs,
        steps,
        bs,
    )

    warm_batch = collate_fn([ds[i] for i in range(min(bs, len(ds)))])
    log.info("Compiling JIT train_step ...")
    t0 = time.perf_counter()
    state, _ = train_step(state, warm_batch, grid, weights)
    jax.block_until_ready(state.params)
    log.info("JIT compile done in %.1fs", time.perf_counter() - t0)

    with history_path.open("a", newline="") as hf:
        writer = csv.DictWriter(
            hf,
            fieldnames=["epoch", "loss", "slat", "offdiag", "e_csf", "pde"],
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
                batch = collate_fn([ds[i] for i in idx])
                state, metrics = train_step(state, batch, grid, weights)
                epoch_loss += float(metrics["loss"])
                last_metrics = {k: float(v) for k, v in metrics.items()}

            row = {
                "epoch": epoch,
                "loss": epoch_loss / steps,
                **{k: last_metrics.get(k, 0.0) for k in ["slat", "offdiag", "e_csf", "pde"]},
            }
            writer.writerow(row)
            hf.flush()

            if (epoch + 1) % max(val_every, 1) == 0 or epoch < 3:
                log.info(
                    "epoch %4d  loss=%.4f  slat=%.4e  offdiag=%.4e  e_csf=%.4e  pde=%.4e",
                    epoch,
                    row["loss"],
                    row["slat"],
                    row["offdiag"],
                    row["e_csf"],
                    row["pde"],
                )

            if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch + 1:04d}.msgpack", state, {"epoch": epoch + 1})

    ckpt = ckpt_dir / "stage_b_last.msgpack"
    save_checkpoint(ckpt, state, {"epoch": n_epochs, "manifest": str(manifest)})
    log.info("Stage B done. checkpoint=%s  history=%s", ckpt, history_path)


if __name__ == "__main__":
    main()
