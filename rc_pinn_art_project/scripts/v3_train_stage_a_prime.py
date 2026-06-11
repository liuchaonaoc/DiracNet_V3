#!/usr/bin/env python3
"""Stage 3a' trainer — Z=3–4 with low-Z anchor mixing + early-stop guards.

Fixes catastrophic forgetting seen in Stage 3a by:
  - mixing ~25% Z<=2 batches each step (anchor replay)
  - lower learning rate (lr_mult)
  - periodic H anchor checks; save best ckpt; stop if guard fails

Example (full curriculum, after optional ground phase):

  cd rc_pinn_art_project && export PYTHONPATH=.
  python scripts/v3_train_stage_a_prime.py --phase full --tag p1z3_4_prime

See also: scripts/v3_p1_stage3a_prime.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np
import optax
from flax.training import train_state

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.losses.loss_schedule import stage_a_dfs_cfg, stage_a_weights
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.anchor_checks import metrics_from_train_state
from pinn_art.training.checkpoint import load_params, save_checkpoint
from pinn_art.training.stage_a_trainer import train_step
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.utils.logging import get_logger

DEFAULT_CFG = "configs/v3_stage_a_z1_26_n10.yaml"
DEFAULT_RESUME = "checkpoints/v3_stage_a_z1_26_n10_p1lowz/stage_a_last.msgpack"


def _filter_df(df, z_min, z_max, ground_only, li_only):
    mask = np.ones(len(df), dtype=bool)
    if z_min is not None:
        mask &= df["Z"].to_numpy() >= z_min
    if z_max is not None:
        mask &= df["Z"].to_numpy() <= z_max
    if li_only:
        mask &= df["Z"].to_numpy() == 3
    if ground_only and "is_ground" in df.columns:
        mask &= df["is_ground"].to_numpy().astype(bool)
    return df[mask].reset_index(drop=True)


def _create_state(model, params, cfg, lr_mult: float):
    opt_cfg = getattr(cfg, "optimizer", cfg)
    lr = float(getattr(opt_cfg, "lr_trunk", 3e-4)) * lr_mult
    wd = float(getattr(opt_cfg, "weight_decay", 1e-4))
    tx = optax.chain(
        optax.clip_by_global_norm(float(getattr(opt_cfg, "grad_clip", 1.0))),
        optax.adamw(lr, weight_decay=wd),
    )

    def apply_fn(params, batch, grid, **kw):
        return model.apply(params, batch, grid, **kw)

    return train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def _sample_mixed_indices(rng, bs, n_anchor_rows, n_main_rows, anchor_frac):
    n_a = 0
    if n_anchor_rows > 0 and anchor_frac > 0:
        n_a = min(n_anchor_rows, max(1, int(round(bs * anchor_frac))))
    n_m = bs - n_a
    if n_main_rows <= 0:
        n_a, n_m = bs, 0
    elif n_anchor_rows <= 0:
        n_a, n_m = 0, bs
    ia = rng.integers(0, max(n_anchor_rows, 1), size=n_a) if n_a else np.array([], dtype=np.int64)
    im = rng.integers(0, max(n_main_rows, 1), size=n_m) if n_m else np.array([], dtype=np.int64)
    return ia, im


def _phase_defaults(phase: str):
    if phase == "ground":
        return dict(z_min=3, z_max=4, ground_only=True, epochs=600, li_only=False)
    if phase == "li":
        return dict(z_min=3, z_max=3, ground_only=False, epochs=400, li_only=True)
    if phase == "full":
        return dict(z_min=3, z_max=4, ground_only=False, epochs=800, li_only=False)
    raise ValueError(f"unknown phase {phase!r}")


def main():
    ap = argparse.ArgumentParser(description="Stage 3a' trainer (anchor mix + early stop)")
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--resume", default=DEFAULT_RESUME)
    ap.add_argument("--phase", choices=["ground", "li", "full"], default="full")
    ap.add_argument("--tag", default="p1z3_4_prime")
    ap.add_argument("--z-min", type=int, default=None)
    ap.add_argument("--z-max", type=int, default=None)
    ap.add_argument("--ground-only", action="store_true", default=None)
    ap.add_argument("--li-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--anchor-frac", type=float, default=0.25, help="Fraction of batch from Z<=2")
    ap.add_argument("--lr-mult", type=float, default=0.3)
    ap.add_argument("--anchor-z-max", type=int, default=2, help="Anchor pool Z<=this")
    ap.add_argument("--check-every", type=int, default=25, help="H guard every N epochs")
    ap.add_argument("--warmup-epochs", type=int, default=3,
                    help="Skip guard during first N epochs (allow transient drift)")
    ap.add_argument("--max-h-exc-meV", type=float, default=200.0,
                    help="Max |H exc error| in meV before stopping (default 200; "
                         "p1lowz baseline ~-34, full warmup drift ~200)")
    ap.add_argument("--max-v-r1-diff", type=float, default=0.10,
                    help="Max |V(r~1) - (-1/r)| in Ha (default 0.10)")
    ap.add_argument("--patience", type=int, default=6,
                    help="Stop if |H err| not improved for this many checks (0=disable)")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--no-early-stop", action="store_true", help="Only log checks, never stop")
    args = ap.parse_args()

    defaults = _phase_defaults(args.phase)
    z_min = args.z_min if args.z_min is not None else defaults["z_min"]
    z_max = args.z_max if args.z_max is not None else defaults["z_max"]
    ground_only = args.ground_only if args.ground_only is not None else defaults["ground_only"]
    li_only = args.li_only or defaults.get("li_only", False)
    n_epochs = args.epochs if args.epochs is not None else defaults["epochs"]

    cfg = load_config(ROOT / args.config)
    log = get_logger()
    log.info("JAX backend=%s  devices=%s", jax.default_backend(), jax.devices())

    manifest = ROOT / cfg.dataset.manifest
    full_df = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max)).df
    anchor_df = full_df[full_df["Z"].to_numpy() <= args.anchor_z_max].reset_index(drop=True)
    main_df = _filter_df(full_df, z_min, z_max, ground_only, li_only)

    n_orb_max = int(cfg.model.n_orb_max)
    n_csf_max = int(getattr(cfg.model, "n_csf_max", 8))
    ds_anchor = ManifestDataset(manifest, n_orb_max=n_orb_max, n_csf_max=n_csf_max)
    ds_anchor.df = anchor_df
    ds_main = ManifestDataset(manifest, n_orb_max=n_orb_max, n_csf_max=n_csf_max)
    ds_main.df = main_df

    log.info(
        "Phase=%s  main Z=%s..%s ground_only=%s li_only=%s  rows main=%d anchor(Z<=%d)=%d",
        args.phase, z_min, z_max, ground_only, li_only, len(main_df), args.anchor_z_max, len(anchor_df),
    )

    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max),
        int(cfg.grid.n_grid), str(cfg.grid.scheme),
    )
    key = jax.random.PRNGKey(int(cfg.seed))
    model, params = build_model_and_params(cfg, grid, key)
    resume = ROOT / args.resume
    params = load_params(resume)
    log.info("Resumed from %s", resume)

    state = _create_state(model, params, cfg, lr_mult=args.lr_mult)
    weights = stage_a_weights(cfg)
    dfs_cfg = stage_a_dfs_cfg(cfg)
    steps = int(cfg.training.steps_per_epoch)
    bs = int(args.batch_size)

    ckpt_dir = ROOT / cfg.training.ckpt_dir
    ckpt_dir = ckpt_dir.parent / f"{ckpt_dir.name}_{args.tag}_{args.phase}"
    log_dir = ROOT / cfg.training.log_dir
    log_dir = log_dir.parent / f"{log_dir.name}_{args.tag}_{args.phase}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    history_path = log_dir / "history.csv"
    guard_path = log_dir / "anchor_guard.jsonl"
    rng_np = np.random.default_rng(int(cfg.seed) + hash(args.phase) % 10000)

    # JIT warm-up
    warm_items = [ds_main[i] for i in range(min(bs, len(main_df)))]
    if len(warm_items) < bs and len(anchor_df):
        warm_items += [ds_anchor[i] for i in range(min(bs - len(warm_items), len(anchor_df)))]
    log.info("JIT compile (lr_mult=%.2f, anchor_frac=%.2f)...", args.lr_mult, args.anchor_frac)
    state, _ = train_step(state, collate_batches(warm_items[:bs], n_csf_max=n_csf_max), grid, weights, dfs_cfg)
    jax.block_until_ready(state.params)

    best_err = float("inf")
    best_epoch = -1
    stale_checks = 0
    stopped_early = False
    stop_reason = ""
    guard_fail_count = 0

    with history_path.open("w", newline="") as hf, guard_path.open("w") as gf:
        writer = csv.DictWriter(
            hf,
            fieldnames=["epoch", "loss", "pde", "ortho", "asym", "norm", "v_prior", "v_smooth", "scf"],
        )
        writer.writeheader()

        for epoch in range(n_epochs):
            t0 = time.perf_counter()
            epoch_loss = 0.0
            last_metrics = {}
            for _ in range(steps):
                key, sub = jax.random.split(key)
                ia, im = _sample_mixed_indices(
                    rng_np, bs, len(anchor_df), len(main_df), args.anchor_frac,
                )
                items = [ds_anchor[int(i)] for i in ia] + [ds_main[int(i)] for i in im]
                batch = collate_batches(items, n_csf_max=n_csf_max)
                state, metrics = train_step(state, batch, grid, weights, dfs_cfg)
                epoch_loss += float(metrics["loss"])
                last_metrics = {k: float(v) for k, v in metrics.items()}

            row = {
                "epoch": epoch,
                "loss": epoch_loss / steps,
                **{k: last_metrics.get(k, 0.0) for k in [
                    "pde", "ortho", "asym", "norm", "v_prior", "v_smooth", "scf",
                ]},
            }
            writer.writerow(row)
            hf.flush()

            in_warmup = epoch < args.warmup_epochs
            do_check = (epoch + 1) % max(args.check_every, 1) == 0
            if do_check:
                am = metrics_from_train_state(
                    state, model, grid, manifest,
                    n_orb_max=n_orb_max,
                    n_csf_max=n_csf_max,
                    max_h_exc_meV=args.max_h_exc_meV,
                    max_v_r1_diff_ha=args.max_v_r1_diff,
                )
                rec = {
                    "epoch": epoch + 1,
                    "h_exc_ev": am.h_exc_ev,
                    "h_exc_err_meV": am.h_exc_err_meV,
                    "v_r1_diff_ha": am.v_r1_diff_ha,
                    "ok": am.ok,
                    "in_warmup": in_warmup,
                    "message": am.message,
                }
                gf.write(json.dumps(rec) + "\n")
                gf.flush()
                log.info("GUARD epoch %4d  warmup=%s  %s", epoch + 1, in_warmup, am.message)

                err_abs = abs(am.h_exc_err_meV)
                if err_abs < best_err:
                    best_err = err_abs
                    best_epoch = epoch + 1
                    stale_checks = 0
                    best_ckpt = ckpt_dir / "best_anchor.msgpack"
                    save_checkpoint(best_ckpt, state, {"epoch": epoch + 1, "guard": rec})
                    log.info("  -> new best |H err|=%.1f meV  saved %s", best_err, best_ckpt)
                else:
                    stale_checks += 1

                if not am.ok:
                    if in_warmup:
                        log.info("  -> warmup: guard FAIL ignored")
                    else:
                        guard_fail_count += 1
                        if not args.no_early_stop:
                            stopped_early = True
                            stop_reason = f"guard FAIL at epoch {epoch+1}: {am.message}"
                            log.warning("EARLY STOP: %s", stop_reason)
                            break

                if args.patience > 0 and stale_checks >= args.patience and not args.no_early_stop and not in_warmup:
                    stopped_early = True
                    stop_reason = (
                        f"no H err improvement for {args.patience} checks "
                        f"(best epoch {best_epoch}, |err|={best_err:.1f} meV)"
                    )
                    log.warning("EARLY STOP: %s", stop_reason)
                    break

            if (epoch + 1) % 25 == 0 or epoch < 3:
                log.info(
                    "epoch %4d  loss=%.3f  pde=%.4f  scf=%.4f  (%.1fs)",
                    epoch, row["loss"], row["pde"], row["scf"], time.perf_counter() - t0,
                )

            if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch + 1:04d}.msgpack", state, {"epoch": epoch + 1})

    last_ckpt = ckpt_dir / "stage_a_last.msgpack"
    save_checkpoint(last_ckpt, state, {"epoch": epoch + 1, "stopped_early": stopped_early})
    log.info("Saved %s", last_ckpt)

    if (ckpt_dir / "best_anchor.msgpack").exists():
        log.info("Best anchor ckpt: %s (epoch %d, |H err|=%.1f meV)", ckpt_dir / "best_anchor.msgpack", best_epoch, best_err)

    if stopped_early:
        log.info("Stopped early: %s", stop_reason)
        log.info("Recommend eval with: best_anchor.msgpack (or resume from it)")
    else:
        log.info("Completed all %d epochs.", n_epochs)

    log.info("history=%s  guard_log=%s", history_path, guard_path)


if __name__ == "__main__":
    main()
