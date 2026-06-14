#!/usr/bin/env python3
"""He 1s→2s 专项诊断：复用 build_prediction_cache，对 He 1s² ground 和 1s1 2s1
激发态跑两条独立 forward（use_ci=False 单粒子 / use_ci=True 含 Racah CI），
对比 (a) E_orb-only total、(b) omega-weighted sum、(c) E_csf 三种"总能量"，
计算 1s→2s 激发能与 NIST 偏差。

跑法（GPU）：
  cd rc_pinn_art_project && export PYTHONPATH=. && export JAX_PLATFORMS=cuda
  python scripts/v3_verify_he_1s2s.py \\
    --ckpt checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack \\
    --manifest data_cache/manifest_z_le_2.parquet \\
    --out logs/path_a_he_1s2s/eprime_he_1s2s.json

输出 JSON 含：
  paths: a_eorb_first / b_omega_weighted / c_csf  → 1s→2s 偏差
  delta_analysis: 三种"总能量"之间的修正量
  internal_E_orb_Ha: gnd_1s / exc_1s / exc_2s  → 原始 E_orb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from pinn_art.ci.racah_cache import RacahCache, build_cache_from_manifest_rows
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.evaluation.excitation_eval import build_prediction_cache
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params, merge_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid

HARTREE_EV = 27.211386245988
HARTREE_MEV = HARTREE_EV * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_stage_a_z1_26_n10.yaml")
    ap.add_argument("--ckpt", default="checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack")
    ap.add_argument("--manifest", default="data_cache/manifest_z_le_2.parquet")
    ap.add_argument("--out", default="logs/path_a_he_1s2s/eprime_he_1s2s.json")
    ap.add_argument("--use-nist-inject", action="store_true", default=False)
    args = ap.parse_args()

    print(f"[1/6] 加载 config: {args.config}")
    cfg = load_config(ROOT / args.config)
    print(f"[2/6] 构建 grid + 模型")
    grid = make_radial_grid(
        r_min=float(cfg.grid.r_min),
        r_max=float(cfg.grid.r_max),
        n_grid=int(cfg.grid.n_grid),
        scheme=str(cfg.grid.scheme),
    )
    n_orb_max = int(cfg.model.n_orb_max)
    n_csf_max = int(getattr(cfg.model, "n_csf_max", 8))
    model, params = build_model_and_params(cfg, grid, key=jax.random.PRNGKey(0))
    print(f"[3/6] 加载 ckpt: {args.ckpt}")
    ckpt_params = load_params(args.ckpt)
    # Stage A ckpt 缺 CI 模板；按 v3_eval_excitation_vs_nist.py 模式注入
    from pinn_art.models.pinn_art_model import _dummy_batch
    dummy = _dummy_batch(n_orb_max, n_csf_max)
    params_ci = model.init(jax.random.PRNGKey(1), dummy, grid, train=False, return_ci=True)
    params = merge_params(params_ci, ckpt_params)
    print(f"[4/6] 加载 manifest: {args.manifest}")
    ds = ManifestDataset(args.manifest, n_orb_max=n_orb_max, n_csf_max=n_csf_max)
    df = ds.df

    # 找 He I 1s² ground 和 1s1 2s1
    gnd_rows = df[(df["Z"] == 2) & (df["ion_charge"] == 0) & (df["level_config"] == "1s2") & (df["is_ground"] == True)]
    exc_rows = df[(df["Z"] == 2) & (df["ion_charge"] == 0) & (df["level_config"] == "1s1 2s1")]
    if len(gnd_rows) == 0 or len(exc_rows) == 0:
        raise RuntimeError("He I 1s² ground 或 1s1 2s1 缺失")
    gnd_idx, exc_idx = int(gnd_rows.index[0]), int(exc_rows.index[0])
    gnd_row = df.iloc[gnd_idx]
    exc_row = df.iloc[exc_idx]
    nist_exc_mev = float(exc_row["level_meV"])
    print(f"  He 1s² ground row={gnd_idx}, 1s1 2s1 row={exc_idx}")
    print(f"  NIST 1s→2s = {nist_exc_mev:.3f} meV ({nist_exc_mev/1000:.4f} eV)")

    # 准备 Racah cache（multi_electron CI 需要）
    ci_cfg = getattr(cfg, "ci", None)
    k_list = list(getattr(ci_cfg, "k_list", [0, 1, 2])) if ci_cfg else [0, 1, 2]
    M_max = int(getattr(ci_cfg, "M_max", 8))
    racah_path = ROOT / getattr(ci_cfg, "racah_cache", "data_cache/racah_cache_z1_26_n10.npz")
    if racah_path.exists():
        racah_cache = RacahCache.load(racah_path)
    else:
        cache = build_cache_from_manifest_rows(df.to_dict("records"), k_list=k_list, M_max=M_max)
        cache.save(racah_path)
        racah_cache = cache
    k_tuple = tuple(k_list)
    print(f"[5/6] Racah cache: {racah_path}  (parents={len(racah_cache.parent_ids)})")

    # 两个 unique key
    gnd_key = (2, 0, "1s2")
    exc_key = (2, 0, "1s1 2s1")
    keys = [gnd_key, exc_key]
    row_lookup = {gnd_key: gnd_idx, exc_key: exc_idx}

    # === 路径 a：use_ci=False（no CI, no R^k on-diag → 直接 E_orb）===
    print("[6/6] 跑 forward ×2 (no CI + with CI)...")
    sv_cache = build_prediction_cache(
        keys, ds, model, params, grid,
        n_csf_max=n_csf_max, use_ci=False, row_lookup=row_lookup,
    )
    # === 路径 b/c：use_ci=True（含 R^k on-diag + Racah C_ang full CI）===
    me_cache = build_prediction_cache(
        keys, ds, model, params, grid,
        n_csf_max=n_csf_max, racah_cache=racah_cache, k_list=k_tuple,
        use_ci=True, row_lookup=row_lookup,
    )

    # 提取预测
    def _totals(pred):
        E_orb = pred["E_orb_ha"]
        omega = pred["omega"]
        omask = pred["orb_mask"].astype(bool)
        # 路径 a：第 1 个 active orbital 的 E_orb（= 1s for He 1s² 闭壳；= 1s2 中第 1 active for 1s1 2s1）
        e_first = float(E_orb[np.where(omask)[0][0]]) if omask.any() else float("nan")
        # 路径 b：omega-weighted E_orb sum（2 电子都被算入）
        w = omega * omask
        e_omega = float(np.sum(E_orb * w)) if w.sum() > 0 else float("nan")
        # 路径 c：E_csf
        e_csf = float(pred.get("E_csf_ha", float("nan")))
        return {
            "E_orb_first_ha": e_first,
            "E_orb_first_meV": e_first * HARTREE_MEV,
            "E_orb_omega_weighted_ha": e_omega,
            "E_orb_omega_weighted_meV": e_omega * HARTREE_MEV,
            "E_csf_ha": e_csf,
            "E_csf_meV": e_csf * HARTREE_MEV if np.isfinite(e_csf) else float("nan"),
            "E_orb_all_ha": E_orb.tolist(),
            "omega": omega.tolist(),
            "orb_mask": omask.tolist(),
        }

    gnd_sv = _totals(sv_cache[gnd_key])
    exc_sv = _totals(sv_cache[exc_key])
    gnd_me = _totals(me_cache[gnd_key])
    exc_me = _totals(me_cache[exc_key])

    def _exc_diff(gnd_t, exc_t, key):
        e_g = gnd_t[key]
        e_e = exc_t[key]
        if not (np.isfinite(e_g) and np.isfinite(e_e)):
            return None
        pred = (e_e - e_g) * HARTREE_MEV
        err = pred - nist_exc_mev
        return {
            "gnd_meV": float(e_g * HARTREE_MEV),
            "exc_meV": float(e_e * HARTREE_MEV),
            "pred_exc_meV": float(pred),
            "nist_exc_meV": nist_exc_mev,
            "err_meV": float(err),
            "err_eV": float(err / 1000.0),
        }

    # 三种 total energy 解释：
    # a_E_orb_first : E_orb[1s]（H 风格 ground, E_orb 1s 唯一 active; 对 He 1s² ground 取 active1 即可；
    #                 1s1 2s1 取 active1=1s；差=0，无意义 → 改为 ground/exc 都用 omega-weighted）
    # b_omega_weighted: E_orb 加权求和（= Σ ω_a E_orb_a，含两个电子）
    # c_csf         : E_csf（含 R^k on-diag + Racah C_ang 完整 CI）

    # 对 He 1s² + 1s1 2s1，正确 ground total 应是 2 个电子的 E_orb 加权和。
    # 1s1 2s1 也是 2 个电子的加权和；差值就是 ΔE_orb 加权。
    # 用 omega-weighted E_orb 是无 CI 路径下"两个电子都被算入"的天然定义。

    paths = {
        "a_omega_weighted_E_orb": {
            "gnd": _exc_diff(gnd_sv, exc_sv, "E_orb_omega_weighted_meV"),
            "exc": _exc_diff(gnd_sv, exc_sv, "E_orb_omega_weighted_meV"),
        },
        "b_CI_with_Rk_diag_and_Racah": {
            "gnd": _exc_diff(gnd_me, exc_me, "E_csf_meV"),
            "exc": _exc_diff(gnd_me, exc_me, "E_csf_meV"),
        },
    }
    # 重新计算：a) 路径是 E_orb_omega_weighted (meV)
    a_path = {
        "pred_exc_meV": exc_sv["E_orb_omega_weighted_meV"] - gnd_sv["E_orb_omega_weighted_meV"],
        "nist_exc_meV": nist_exc_mev,
        "err_meV": (exc_sv["E_orb_omega_weighted_meV"] - gnd_sv["E_orb_omega_weighted_meV"]) - nist_exc_mev,
        "gnd_total_meV": gnd_sv["E_orb_omega_weighted_meV"],
        "exc_total_meV": exc_sv["E_orb_omega_weighted_meV"],
    }
    b_path = {
        "pred_exc_meV": exc_me["E_csf_meV"] - gnd_me["E_csf_meV"],
        "nist_exc_meV": nist_exc_mev,
        "err_meV": (exc_me["E_csf_meV"] - gnd_me["E_csf_meV"]) - nist_exc_mev,
        "gnd_total_meV": gnd_me["E_csf_meV"],
        "exc_total_meV": exc_me["E_csf_meV"],
    }

    print()
    print("=" * 70)
    print("He 1s→2s 诊断（4 路径 vs NIST = 19819.6 meV）")
    print()
    print(f"[a] E_orb omega-weighted (no CI) :")
    print(f"     gnd_total = {a_path['gnd_total_meV']:.1f} meV  exc_total = {a_path['exc_total_meV']:.1f} meV")
    print(f"     → 1s→2s = {a_path['pred_exc_meV']:.1f} meV  err = {a_path['err_meV']:+.1f} meV ({a_path['err_meV']/1000.0:+.4f} eV)")
    print()
    print(f"[b] E_csf (CI with R^k on-diag + Racah) :")
    print(f"     gnd_total = {b_path['gnd_total_meV']:.1f} meV  exc_total = {b_path['exc_total_meV']:.1f} meV")
    print(f"     → 1s→2s = {b_path['pred_exc_meV']:.1f} meV  err = {b_path['err_meV']:+.1f} meV ({b_path['err_meV']/1000.0:+.4f} eV)")
    print()
    rk_corr = b_path["err_meV"] - a_path["err_meV"]
    print(f"修正量拆分：")
    print(f"  a→b (CI 修正 = R^k on-diag + Racah C_ang) = {rk_corr:+.1f} meV")

    # === 写 JSON ===
    out = {
        "config": args.config,
        "ckpt": args.ckpt,
        "manifest": args.manifest,
        "nist_1s2s_exc_meV": nist_exc_mev,
        "nist_1s2s_exc_eV": nist_exc_mev / 1000.0,
        "gnd_row_idx": gnd_idx,
        "exc_row_idx": exc_idx,
        "gnd_row": {
            "level_config": gnd_row["level_config"],
            "group": gnd_row["group"],
            "level_abs_meV": float(gnd_row.get("level_abs_meV", 0)),
        },
        "exc_row": {
            "level_config": exc_row["level_config"],
            "group": exc_row["group"],
            "level_meV": float(exc_row.get("level_meV", 0)),
            "level_abs_meV": float(exc_row.get("level_abs_meV", 0)),
        },
        "paths": {
            "a_omega_weighted_E_orb": a_path,
            "b_full_CI_with_Rk_diag_and_Racah": b_path,
        },
        "delta_analysis": {
            "eorb_weighted_err_meV": a_path["err_meV"],
            "ci_correction_meV": rk_corr,
        },
        "internal_gnd_E_orb_Ha": gnd_sv["E_orb_all_ha"],
        "internal_exc_E_orb_Ha": exc_sv["E_orb_all_ha"],
        "internal_gnd_omega": gnd_sv["omega"],
        "internal_exc_omega": exc_sv["omega"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
