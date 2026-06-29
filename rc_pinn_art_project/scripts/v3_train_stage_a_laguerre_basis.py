#!/usr/bin/env python3
"""Stage A Round 2 trainer — Differentiable Generalized-Laguerre Basis.

This is a thin shim around the canonical Stage-A trainer
(``v3_train_stage_a.py``).  It exists so the call-site for the Laguerre
experiment can be a single, well-known script name and so the per-step
metrics CSV is named distinctly.

What changes vs. the default Stage-A trainer:
  - Uses ``configs/v3_stage_a_laguerre_basis.yaml`` (per-orbital Laguerre
    coefficients, learnable ``λ_a``, bounded SIREN ``δQ``).
  - Logs three extra metrics each epoch: ``coeff_decay``, ``lambda_prior``,
    ``q_residual`` (in addition to pde / ortho / norm / asym / v_* / scf).
  - Hard-codes the Laguerre-specific checkpoints/log directory (overridable
    via ``--tag``).

What does NOT change:
  - ``ManifestDataset`` + ``train_step`` + ``build_model_and_params`` are
    reused verbatim.  The Laguerre toggles flow through the cfg → model
    constructor that was wired in R2a.

Usage (run on GPU box, see also prompts/17 §R3):
    python scripts/v3_train_stage_a_laguerre_basis.py \\
        --config configs/v3_stage_a_laguerre_basis.yaml \\
        --epochs 2000 --steps-per-epoch 5
"""

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
from pinn_art.losses.loss_schedule import stage_a_dfs_cfg, stage_a_weights
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params, save_checkpoint
from pinn_art.training.stage_a_trainer import train_step
from pinn_art.training.train_state import create_train_state
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.utils.logging import get_logger


# --- copied from v3_train_stage_a.py to keep behaviour identical ----------------


def _prepare_manifest(manifest: Path, cfg) -> None:
    log = get_logger()
    prepare = str(getattr(cfg.dataset, "prepare", "hydrogenic"))
    if prepare == "nist_z1_26_n10":
        log.info("Building Round-2 manifest via v3_prepare_nist_manifest.py ...")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "v3_prepare_nist_manifest.py"),
                "--z-min", "1", "--z-max", "26", "--n-levels", "10",
                "--out", str(manifest),
            ],
            cwd=str(ROOT),
        )
    else:
        log.info("Building hydrogenic manifest via v3_prepare_hydrogenic.py ...")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "v3_prepare_hydrogenic.py"),
                "--z-min", "1", "--z-max", "8", "--n-levels", "6",
                "--out", str(manifest),
            ],
            cwd=str(ROOT),
        )


def main():
    ap = argparse.ArgumentParser(
        description="Stage A Round 2 trainer (Laguerre basis ansatz)"
    )
    ap.add_argument("--config", default="configs/v3_stage_a_laguerre_basis.yaml")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--prepare-data", action="store_true")
    ap.add_argument("--z-min", type=int, default=None)
    ap.add_argument("--z-max", type=int, default=None)
    ap.add_argument("--ground-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    log = get_logger()
    log.info("JAX backend=%s  devices=%s", jax.default_backend(), jax.devices())
    log.info(
        "Laguerre config: use=%s K_max=%s learn_lambda=%s perturb_P=%s perturb_Q=%s",
        cfg.model.use_laguerre_basis, cfg.model.K_max,
        cfg.model.learn_lambda, cfg.model.perturb_scale_P, cfg.model.perturb_scale_Q,
    )

    manifest = ROOT / cfg.dataset.manifest
    if args.prepare_data or not manifest.exists():
        _prepare_manifest(manifest, cfg)

    ds = ManifestDataset(
        manifest,
        n_orb_max=int(cfg.model.n_orb_max),
        n_csf_max=int(getattr(cfg.model, "n_csf_max", 8)),
    )
    n_before = len(ds)
    mask = np.ones(n_before, dtype=bool)
    if args.z_min is not None:
        mask &= ds.df["Z"].to_numpy() >= args.z_min
    if args.z_max is not None:
        mask &= ds.df["Z"].to_numpy() <= args.z_max
    if args.ground_only and "is_ground" in ds.df.columns:
        mask &= ds.df["is_ground"].to_numpy().astype(bool)
    if not mask.all():
        ds.df = ds.df[mask].reset_index(drop=True)
        log.info("Subset filter: %d -> %d rows (z_min=%s z_max=%s ground_only=%s)",
                 n_before, len(ds), args.z_min, args.z_max, args.ground_only)

    grid = make_radial_grid(
        float(cfg.grid.r_min),
        float(cfg.grid.r_max),
        int(cfg.grid.n_grid),
        str(cfg.grid.scheme),
    )

    key = jax.random.PRNGKey(int(cfg.seed))
    model, params = build_model_and_params(cfg, grid, key)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    log.info("params initialised: %d leaves, %.2fM floats",
             len(jax.tree_util.tree_leaves(params)), n_params / 1e6)

    resume_path = args.resume or getattr(cfg.training, "resume_from", None)
    if resume_path:
        resume_path = ROOT / resume_path
        params = load_params(resume_path)
        log.info("Resumed params from %s", resume_path)

    state = create_train_state(model, params, cfg, total_steps=100000)
    weights = stage_a_weights(cfg)
    dfs_cfg = stage_a_dfs_cfg(cfg)
    log.info(
        "DFS: enabled=%s alpha_x=%.2f latter_tail=%s anchor_vprior=%s path_a=%s w_scf=%.3g",
        dfs_cfg[0], dfs_cfg[1], dfs_cfg[2], dfs_cfg[3], dfs_cfg[7],
        weights.get("scf", 0.0),
    )
    log.info(
        "Laguerre loss weights: coeff_decay=%s lambda_prior=%s q_residual=%s",
        weights.get("coeff_decay", 0.0),
        weights.get("lambda_prior", 0.0),
        weights.get("q_residual", 0.0),
    )

    n_epochs = int(args.epochs) if args.epochs is not None else int(cfg.stage_a.n_epochs)
    steps = int(cfg.training.steps_per_epoch)
    bs = int(args.batch_size) if args.batch_size is not None else int(cfg.stage_a.batch_size)
    shuffle = bool(getattr(cfg.training, "shuffle_each_epoch", True))
    ckpt_every = int(getattr(cfg.training, "ckpt_every_epochs", 0))
    val_every = int(getattr(cfg.training, "val_every_epochs", 25))

    ckpt_dir = ROOT / cfg.training.ckpt_dir
    log_dir = ROOT / cfg.training.log_dir
    if args.tag:
        ckpt_dir = ckpt_dir.parent / f"{ckpt_dir.name}_{args.tag}"
        log_dir = log_dir.parent / f"{log_dir.name}_{args.tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / "history.csv"

    write_header = not history_path.exists()
    rng_np = np.random.default_rng(int(cfg.seed))
    log.info("Stage A Laguerre: %d samples × %d epochs × %d steps × batch=%d",
             len(ds), n_epochs, steps, bs)

    warm_idx = list(range(min(bs, len(ds))))
    warm_batch = collate_batches(
        [ds[i] for i in warm_idx],
        n_csf_max=int(cfg.model.n_csf_max),
        k_max=int(getattr(cfg.model, "K_max", 9)),
        build_nodes=bool(getattr(cfg.model, "use_hybrid_head", False)),
        nodes_table_path=str(getattr(cfg.model, "nodes_table_path",
                                     "data_cache/laguerre_nodes_z1_26_n1_10.parquet")),
    )
    log.info("Compiling JIT train_step on %s (first call traces XLA)…",
             jax.default_backend())
    t_compile = time.perf_counter()
    state, _ = train_step(state, warm_batch, grid, weights, dfs_cfg)
    jax.block_until_ready(state.params)
    log.info("JIT compile done in %.1fs (steady-state ~0.05-0.3s/step on GPU)",
             time.perf_counter() - t_compile)

    fieldnames = ["epoch", "loss", "pde", "ortho", "asym", "norm",
                  "v_prior", "v_smooth", "scf",
                  "coeff_decay", "coeff_anchor", "lambda_prior", "q_residual"]
    with history_path.open("a", newline="") as hf:
        writer = csv.DictWriter(hf, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for epoch in range(n_epochs):
            t_epoch = time.perf_counter()
            epoch_loss = 0.0
            last = {}
            for step_idx in range(steps):
                key, sub = jax.random.split(key)
                if shuffle:
                    perm = jax.random.permutation(sub, len(ds))
                    idx = [int(perm[i % len(ds)]) for i in range(bs)]
                else:
                    idx = [(step_idx * bs + i) % len(ds) for i in range(bs)]
                batch = collate_batches(
                    [ds[i] for i in idx],
                    n_csf_max=int(cfg.model.n_csf_max),
                    k_max=int(getattr(cfg.model, "K_max", 9)),
                    build_nodes=bool(getattr(cfg.model, "use_hybrid_head", False)),
                    nodes_table_path=str(getattr(cfg.model, "nodes_table_path",
                                                 "data_cache/laguerre_nodes_z1_26_n1_10.parquet")),
                )
                state, metrics = train_step(state, batch, grid, weights, dfs_cfg)
                epoch_loss += float(metrics["loss"])
                last = {k: float(v) for k, v in metrics.items()}

            row = {"epoch": epoch, "loss": epoch_loss / steps}
            for k in fieldnames[1:]:
                row[k] = last.get(k, 0.0)
            writer.writerow(row)
            hf.flush()

            sec = time.perf_counter() - t_epoch
            sps = sec / max(steps, 1)
            if (epoch + 1) % max(val_every, 1) == 0 or epoch < 3:
                log.info(
                    "epoch %4d  loss=%.4f  pde=%.4f  ortho=%.4e  norm=%.4f  "
                    "cd=%.3e  ca=%.3e  λ=%.3e  q=%.3e  %.1fs (%.2fs/step)",
                    epoch, row["loss"], row["pde"], row["ortho"], row["norm"],
                    row["coeff_decay"], row["coeff_anchor"],
                    row["lambda_prior"], row["q_residual"],
                    sec, sps,
                )
            elif sps > 3.0:
                log.warning("epoch %4d slow: %.2fs/step — install jax[cuda12]?",
                            epoch, sps)

            if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
                ep_ckpt = ckpt_dir / f"epoch_{epoch + 1:04d}.msgpack"
                save_checkpoint(ep_ckpt, state, {"epoch": epoch + 1})

    ckpt = ckpt_dir / "stage_a_last.msgpack"
    save_checkpoint(ckpt, state, {"epoch": n_epochs, "manifest": str(manifest)})
    log.info("Laguerre training done.  ckpt=%s  history=%s", ckpt, history_path)


if __name__ == "__main__":
    main()