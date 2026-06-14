#!/usr/bin/env python3
"""Stage 3a'' trainer — Path A (R^k→V_dfs 注入) B' experiment.

基于 v3_train_stage_a_prime.py 的 anchor-mix + early-stop 流程。
差异:
  - 从 E-prime 续训, 用 merge_params 把 E-prime params 合并到
    含 slater_log_scale=0 的新 init (Path A 注入点)
  - 启用 dfs.path_a_enabled=True
  - ci.enabled=True (slater_log_scale 才会被创建)
  - lr_mult 进一步降低 (lr 已在 config 里降到 0.1x E-prime)

Example:
  cd rc_pinn_art_project && export PYTHONPATH=.
  python scripts/v3_train_path_a_minimal.py --resume \
    checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack \
    --tag p1z3_4_bprime

注意:
  - 必须从 E-prime (return_ci=False ckpt) 续训 → 用 merge_params 自动加 slater_log_scale
  - 第一 epoch 后 slater_log_scale 会从 0 开始被梯度拉 (期望方向: 负)
  - V_dfs_aug = V_dfs + V_slater_corr 初期是 V_dfs + 0 = V_dfs, 与 E-prime 兼容
  - 但 forward 现在跑 return_ci=True → forward 慢 1.5x → 一 epoch 慢 1.5x
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
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from flax.training import train_state

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.losses.loss_schedule import stage_a_dfs_cfg, stage_a_weights
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.anchor_checks import metrics_from_train_state
from pinn_art.training.checkpoint import load_params, merge_params, save_checkpoint
from pinn_art.training.stage_a_trainer import train_step
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.utils.logging import get_logger

# B'' fix: He 1s² E_orb 监控常量
# He 1s² ground level_abs = 0 (定义), 1s orbital E_orb ≈ -2.847 Ha (NIST)
# 用 -2.85 Ha 为参考, 容差 0.15 Ha (约 4 eV, 比 §15 的 11 eV 退化要严得多)
HARTREE_EV = 27.211386245988
HE_1S2_NIST_EORB_HA = -2.847  # NIST He 1s² (闭壳) 1s orbital E_orb (Ha)
# E-prime baseline 实测 He 1s² E_orb ≈ -2.879 Ha (NIST 偏深 0.87 eV)
HE_1S2_NIST_TOTAL_HA = -2.847  # NIST He 1s² 1s² 总能量 (Ha) — §15 multi_electron 测


def _helium_anchor_check(state, model, grid, manifest, n_orb_max, n_csf_max,
                         max_dev_ha: float = 0.30):
    """B'' fix: 监控 He 1s² 1s² 多电子总能量, 防 §15 的 11 eV 退化.

    多电子总能量 = Σ_a ω_a * E_orb[a] (Ha), 与 NIST -2.847 Ha 比较.
    E-prime 实测: 2 * (-1.36) = -2.72 Ha → 偏浅 0.13 Ha
    B' baseline 实测: 2 * (-1.64) = -3.28 Ha → 偏深 0.43 Ha (退化 §15)
    阈值 0.30 Ha (~8.2 eV) 卡住 §15 退化.

    Returns: dict with keys
      - he_1s2_total_ha: float (He 1s² 总能量 = ω·E_orb 加权和, Ha)
      - he_1s2_err_meV: float ((pred - nist) * Ha_to_eV * 1000)
      - ok: bool (True if |err_ha| <= max_dev_ha)
      - message: str
    """
    import pandas as pd
    df = pd.read_parquet(str(manifest))
    sub = df[(df["Z"] == 2) & (df["ion_charge"] == 0) & (df["level_config"] == "1s2")]
    if len(sub) == 0:
        return {"he_1s2_total_ha": float("nan"), "he_1s2_err_meV": float("nan"),
                "ok": False, "message": "He 1s² not in manifest"}
    ds = ManifestDataset(manifest, n_orb_max=n_orb_max, n_csf_max=n_csf_max)
    item = ds[int(sub.index[0])]
    batch = collate_batches([item], n_csf_max=n_csf_max)
    out = model.apply(state.params, batch, grid, train=False)
    E_orb = np.asarray(out["E_orb"][0], dtype=np.float64)
    omega = np.asarray(batch["omega"][0], dtype=np.float64)
    mask = np.asarray(batch["orb_mask"][0], dtype=bool)
    # 多电子总能量 = Σ_a ω_a * E_orb[a] (mask 内)
    total_ha = float(np.sum(omega[mask] * E_orb[mask]))
    eorb_nist = HE_1S2_NIST_TOTAL_HA
    err_ha = total_ha - eorb_nist
    err_mev = err_ha * HARTREE_EV * 1000.0
    ok = abs(err_ha) <= max_dev_ha
    msg = (f"He 1s² total={total_ha:+.4f} Ha  NIST={eorb_nist:+.4f} Ha  "
           f"err={err_mev:+.1f} meV  {'PASS' if ok else 'FAIL'}")
    return {"he_1s2_total_ha": total_ha, "he_1s2_err_meV": err_mev,
            "ok": ok, "message": msg}

DEFAULT_CFG = "configs/v3_stage_a_prime3_path_a.yaml"
DEFAULT_RESUME = "checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack"


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
    # B'''' fix: 恢复 multi_transform, 给 slater_log_scale 单独 10x LR
    # 原因 (per §20): B''' 用单一 LR, slater_log_scale 210 epoch 几乎不动
    # slater_log_scale 是单标量, 梯度量级小, 必须 10x LR 才能学
    # 其他参数 (DeepONet trunk/branch) 用主 LR
    slater_lr_mult = float(getattr(opt_cfg, "slater_log_scale_lr_mult", 10.0))
    base_tx = optax.chain(
        optax.clip_by_global_norm(float(getattr(opt_cfg, "grad_clip", 1.0))),
        optax.adamw(lr, weight_decay=wd),
    )
    slater_tx = optax.chain(
        optax.clip_by_global_norm(float(getattr(opt_cfg, "grad_clip", 1.0))),
        optax.adamw(lr * slater_lr_mult, weight_decay=wd),
    )

    # optax.partition 需要 param_labels pytree (same shape as params)
    # True 表示用 slater_tx, False 表示用 base_tx
    def _make_labels(pytree):
        # params 结构: {'params': {'slater_log_scale': arr, 'DeepONetDirac_0': {...}}}
        if isinstance(pytree, dict):
            return {k: _make_labels(v) for k, v in pytree.items()}
        return False  # 默认 base_tx

    labels = _make_labels(params)
    # 标记 slater_log_scale 为 True
    if "params" in labels and "slater_log_scale" in labels["params"]:
        labels["params"]["slater_log_scale"] = True

    tx = optax.partition(
        {True: slater_tx, False: base_tx},
        labels,
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
        return dict(z_min=3, z_max=4, ground_only=True, epochs=300, li_only=False)
    if phase == "li":
        return dict(z_min=3, z_max=3, ground_only=False, epochs=200, li_only=True)
    if phase == "full":
        return dict(z_min=3, z_max=4, ground_only=False, epochs=600, li_only=False)
    raise ValueError(f"unknown phase {phase!r}")


def main():
    ap = argparse.ArgumentParser(description="Stage 3a'' trainer (Path A R^k→V_dfs inject)")
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--resume", default=DEFAULT_RESUME)
    ap.add_argument("--phase", choices=["ground", "li", "full"], default="full")
    ap.add_argument("--tag", default="p1z3_4_bprime")
    ap.add_argument("--z-min", type=int, default=None)
    ap.add_argument("--z-max", type=int, default=None)
    ap.add_argument("--ground-only", action="store_true", default=None)
    ap.add_argument("--li-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--anchor-frac", type=float, default=0.30, help="B': 0.25→0.30, 加强 H 锚点")
    ap.add_argument("--lr-mult", type=float, default=1.0, help="B': 不再二次乘 0.3, config 已有 0.1x lr")
    ap.add_argument("--anchor-z-max", type=int, default=2)
    ap.add_argument("--check-every", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="B': warmup=5 (forward 改成 return_ci=True, 初期 V 跳变要更宽容)")
    ap.add_argument("--max-h-exc-meV", type=float, default=5000.0,
                    help="B' fix v3: 300→5000 (前 100 epoch 期间不卡 H err, E-prime baseline 也是 100 epoch 才收敛)")
    ap.add_argument("--max-v-r1-diff", type=float, default=2.0,
                    help="B' fix: 0.15→2.0 (E-prime baseline 0.99 Ha, 不能用 0.15 卡死)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--max-he-1s2-dev-ha", type=float, default=0.20,
                    help="B'' fix: He 1s² E_orb 偏离 NIST 不超过 0.20 Ha (~5.4 eV, "
                         "严于 §15 的 11 eV 退化)")
    ap.add_argument("--log-slater-every", type=int, default=5,
                    help="B': 每 N epochs 记录 slater_log_scale 实际值")
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
    # B' 关键: ci.enabled=True 才会让 model 创建 slater_log_scale
    model, params_init = build_model_and_params(cfg, grid, key)
    log.info("Init params keys (含 slater_log_scale): %s",
             list(params_init["params"].keys()) if "params" in params_init else list(params_init.keys()))

    resume = ROOT / args.resume
    params_eprime = load_params(resume)
    log.info("Loaded E-prime ckpt from %s", resume)

    # B' 关键: merge_params — 把 E-prime 的 params 合并到含 slater_log_scale 的新 init
    # slater_log_scale=0 (默认 init), 其他 param 用 E-prime 的
    params = merge_params(params_init, params_eprime)
    log.info("Merged params (slater_log_scale 从 E-prime 缺失 → 保留 init=0)")

    # B'''' fix: 强制重设 slater_log_scale 到诊断最优值 -1.0
    # 原因: merge_params 会用 loaded ckpt 的 slater_log_scale 值覆盖 template init
    # (B'' best_anchor ckpt 含 slater_log_scale=-3.0, 会覆盖 -1.0)
    # → 训练从 -3.0 起步, 60 epoch 才 +3.4e-4, 整个训练实质无注入
    if "params" in params and "slater_log_scale" in params["params"]:
        params["params"]["slater_log_scale"] = jnp.full_like(
            params["params"]["slater_log_scale"], -1.0
        )
        log.info("B'''' fix: 重设 slater_log_scale = -1.0 (exp=0.37, 诊断最优值)")

    state = _create_state(model, params, cfg, lr_mult=args.lr_mult)
    weights = stage_a_weights(cfg)
    dfs_cfg = stage_a_dfs_cfg(cfg)
    log.info("dfs_cfg = %s", dfs_cfg)
    # B' fix v3: path_a_enabled 可关 (return_ci=True 改 P/Q 梯度, 需先验证 E-prime 复现)

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
    slater_path = log_dir / "slater_log_scale.jsonl"
    rng_np = np.random.default_rng(int(cfg.seed) + hash(args.phase) % 10000)

    # JIT warm-up
    warm_items = [ds_main[i] for i in range(min(bs, len(main_df)))]
    if len(warm_items) < bs and len(anchor_df):
        warm_items += [ds_anchor[i] for i in range(min(bs - len(warm_items), len(anchor_df)))]
    log.info("JIT compile (Path A ON, return_ci=True, anchor_frac=%.2f)...", args.anchor_frac)
    state, _ = train_step(state, collate_batches(warm_items[:bs], n_csf_max=n_csf_max), grid, weights, dfs_cfg)
    jax.block_until_ready(state.params)

    best_err = float("inf")
    best_epoch = -1
    stale_checks = 0
    stopped_early = False
    stop_reason = ""
    guard_fail_count = 0

    # 读 slater_log_scale 值的辅助
    def _read_slater(s):
        # slater_log_scale 在 params['params']['slater_log_scale'] (model 顶层 param)
        try:
            v = s.params["params"]["slater_log_scale"]
            return np.asarray(v)
        except Exception:
            return None

    with history_path.open("w", newline="") as hf, guard_path.open("w") as gf, slater_path.open("w") as sf:
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

            # B' 监控: slater_log_scale 实际值
            if (epoch + 1) % max(args.log_slater_every, 1) == 0 or epoch < 3:
                sls = _read_slater(state)
                if sls is not None:
                    rec_sls = {
                        "epoch": epoch + 1,
                        "slater_log_scale": sls.tolist(),
                        "exp_slater_log_scale": np.exp(sls).tolist(),
                    }
                    sf.write(json.dumps(rec_sls) + "\n")
                    sf.flush()
                    log.info("SLATER epoch %4d  log_scale=%s  exp=%s",
                             epoch + 1, [f"{x:+.3f}" for x in sls.tolist()],
                             [f"{x:.3f}" for x in np.exp(sls).tolist()])

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
                # B'' fix: 跑 He 1s² E_orb 监控 (防 §15 的 11 eV 退化)
                he = _helium_anchor_check(
                    state, model, grid, manifest, n_orb_max, n_csf_max,
                    max_dev_ha=args.max_he_1s2_dev_ha,
                )
                rec = {
                    "epoch": epoch + 1,
                    "h_exc_ev": am.h_exc_ev,
                    "h_exc_err_meV": am.h_exc_err_meV,
                    "v_r1_diff_ha": am.v_r1_diff_ha,
                    "he_1s2_total_ha": he["he_1s2_total_ha"],
                    "he_1s2_err_meV": he["he_1s2_err_meV"],
                    "ok": am.ok and he["ok"],
                    "in_warmup": in_warmup,
                    "message": am.message + "  |  " + he["message"],
                }
                gf.write(json.dumps(rec) + "\n")
                gf.flush()
                log.info("GUARD epoch %4d  warmup=%s  %s", epoch + 1, in_warmup, rec["message"])

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

                if not (am.ok and he["ok"]):
                    if in_warmup:
                        log.info("  -> warmup: guard FAIL ignored")
                    else:
                        guard_fail_count += 1
                        if not args.no_early_stop:
                            stopped_early = True
                            stop_reason = f"guard FAIL at epoch {epoch+1}: {rec['message']}"
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

    log.info("history=%s  guard_log=%s  slater_log=%s",
             history_path, guard_path, slater_path)


if __name__ == "__main__":
    main()
