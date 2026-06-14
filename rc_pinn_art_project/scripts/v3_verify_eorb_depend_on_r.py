#!/usr/bin/env python3
"""E_orb 偏浅根因诊断：把 V_net(r) 与 -(Z-N+1)/r（Latter 渐近线）逐点对比，
分桶看 V_net 偏差和 E_orb 偏差的 r-依赖关系。

目的：He 1s² E_orb 偏浅 1.5 Ha，但 H 1s 偏深 18 meV。诊断"偏浅"根因
  - 核区 (r<0.5) : V_net 是否偏离 -Z/r 太多
  - 中间区 (0.5<r<5) : V_net 与 V_dfs 软约束的拟合
  - 渐近区 (r>10) : V_net 是否满足 Latter tail -(Z-N+1)/r

对 H 1s/2s, He 1s² (closed-shell), He⁺ 1s/2s, Li 1s²2s ground 跑 forward,
输出 4 列：
  r  |  V_net  |  -(Z-N+1)/r  |  diff = V_net - (-(Z-N+1)/r)  |  E_orb (该 active orbital)

跑法（GPU，5-10 分钟）:
  cd rc_pinn_art_project && export PYTHONPATH=. && export JAX_PLATFORMS=cuda
  python scripts/v3_verify_eorb_depend_on_r.py \\
    --ckpt checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack \\
    --manifest data_cache/manifest_z_le_2.parquet \\
    --out logs/path_a_eorb_r/eprime_eorb_r.json
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
from pinn_art.data.dataset import ManifestDataset
from pinn_art.evaluation.excitation_eval import build_prediction_cache
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params, merge_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid

HARTREE_EV = 27.211386245988
HARTREE_MEV = HARTREE_EV * 1000.0

# NIST 真空零点参考（Hartree）
NIST_HE_1S2_ABS_HA = -108845.545 / HARTREE_MEV  # -2.861681
NIST_HE_2S_ABS_HA = -4.0 / 27.211386245988     # -0.146986
NIST_HE_2P_ABS_HA = -4.0 / 27.211386245988     # -0.146986 (2s/2p 简并)
NIST_H_2S_ABS_HA = -0.125


def _diagnostic_radii(grid, n=12):
    """取对数均匀的 12 个诊断 r 点（覆盖 r=0.05 到 r=50）"""
    r = np.asarray(grid.r)
    # 选对数尺度均匀的 12 点
    log_r = np.log(r)
    samples = np.linspace(log_r[0] + 0.1, log_r[-1] - 0.1, n)
    return np.unique(np.searchsorted(r, np.exp(samples)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_stage_a_z1_26_n10.yaml")
    ap.add_argument("--ckpt", default="checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack")
    ap.add_argument("--manifest", default="data_cache/manifest_z_le_2.parquet")
    ap.add_argument("--out", default="logs/path_a_eorb_r/eprime_eorb_r.json")
    ap.add_argument("--n-radii", type=int, default=12)
    args = ap.parse_args()

    print(f"[1/5] 加载 config + grid + 模型")
    cfg = load_config(ROOT / args.config)
    grid = make_radial_grid(
        r_min=float(cfg.grid.r_min), r_max=float(cfg.grid.r_max),
        n_grid=int(cfg.grid.n_grid), scheme=str(cfg.grid.scheme),
    )
    n_orb_max = int(cfg.model.n_orb_max)
    n_csf_max = int(getattr(cfg.model, "n_csf_max", 8))
    model, params = build_model_and_params(cfg, grid, key=jax.random.PRNGKey(0))

    print(f"[2/5] 加载 ckpt (Stage A, 注入 CI 模板)")
    ckpt_params = load_params(args.ckpt)
    from pinn_art.models.pinn_art_model import _dummy_batch
    dummy = _dummy_batch(n_orb_max, n_csf_max)
    params_ci = model.init(jax.random.PRNGKey(1), dummy, grid, train=False, return_ci=True)
    params = merge_params(params_ci, ckpt_params)

    print(f"[3/5] 加载 manifest + Racah cache")
    ds = ManifestDataset(args.manifest, n_orb_max=n_orb_max, n_csf_max=n_csf_max)
    df = ds.df
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

    # 诊断 4 个体系
    targets = [
        ("H_1s",   {"Z": 1, "ion_charge": 0, "level_config": "1s1"}),
        ("H_2s",   {"Z": 1, "ion_charge": 0, "level_config": "2s1"}),
        ("Heplus_1s", {"Z": 2, "ion_charge": 1, "level_config": "1s1"}),
        ("Heplus_2s", {"Z": 2, "ion_charge": 1, "level_config": "2s1"}),
        ("He_1s2", {"Z": 2, "ion_charge": 0, "level_config": "1s2"}),
    ]
    keys = [(t["Z"], t["ion_charge"], t["level_config"]) for _, t in targets]
    row_lookup = {}
    for _, t in targets:
        k = (t["Z"], t["ion_charge"], t["level_config"])
        hits = df[(df["Z"] == k[0]) & (df["ion_charge"] == k[1]) & (df["level_config"] == k[2])]
        if len(hits) == 0:
            raise RuntimeError(f"manifest 缺 {k}")
        row_lookup[k] = int(hits.index[0])

    # 准备 z 渐近线：(Z - N + 1)/r   N = Z - ion_charge
    def _z_asymptote(Z, ion, r_grid):
        N = Z - ion
        z_tail = (Z - N + 1)  # 1 for neutrals with 1 e, 2 for He II, etc
        # 但 He 1s² 闭壳 N=2, Z=2 → z_tail = 2 - 2 + 1 = 1 ✓
        # He⁺ N=1, Z=2 → z_tail = 2 - 1 + 1 = 2 ✓
        # H N=1, Z=1 → z_tail = 1 - 1 + 1 = 1 ✓
        return -z_tail / r_grid

    # 准备 bare -Z/r
    def _z_bare(Z, r_grid):
        return -Z / r_grid

    # 计算 V_net + E_orb（不走 CI 路径）
    print(f"[4/5] 跑 forward (use_ci=False) ...")
    sv_cache = build_prediction_cache(
        keys, ds, model, params, grid,
        n_csf_max=n_csf_max, use_ci=False, row_lookup=row_lookup,
    )
    # He 1s² multi_electron CI 也跑一下
    me_cache = build_prediction_cache(
        keys, ds, model, params, grid,
        n_csf_max=n_csf_max, racah_cache=racah_cache, k_list=k_tuple,
        use_ci=True, row_lookup=row_lookup,
    )

    print(f"[5/5] 解析 + 写 JSON")
    r = np.asarray(grid.r)
    r_idx = _diagnostic_radii(grid, n=args.n_radii)

    out = {
        "config": args.config,
        "ckpt": args.ckpt,
        "manifest": args.manifest,
        "diagnostic_r": [float(r[i]) for i in r_idx],
        "systems": {},
    }

    for name, t in targets:
        k = (t["Z"], t["ion_charge"], t["level_config"])
        # 取 E_orb（sv_cache 已有）+ E_csf（me_cache 已有）
        E_orb = np.asarray(sv_cache[k]["E_orb_ha"])
        E_csf = me_cache[k].get("E_csf_ha", float("nan"))
        om = np.asarray(sv_cache[k]["omega"])
        omask = np.asarray(sv_cache[k]["orb_mask"]).astype(bool)
        omega_weighted = float(np.sum(E_orb * om * omask)) if (om * omask).sum() > 0 else float("nan")

        # 跑一次 forward 拿到 V_net（重做一遍以拿到 V）
        from pinn_art.data.collate import collate_batches
        idx = row_lookup[k]
        batch = collate_batches([ds[idx]], n_csf_max=n_csf_max)
        out_model = model.apply(params, batch, grid, train=False, return_ci=True)
        V = np.asarray(out_model["V"][0])  # [N_g]
        # 第一个 active orbital 的 E_orb
        first_active = int(np.where(omask)[0][0]) if omask.any() else 0
        E_orb_first = float(E_orb[first_active])

        # 计算 V_net 在 12 个诊断 r 点的值
        V_at_r = [float(V[i]) for i in r_idx]
        V_bare_at_r = [float(_z_bare(t["Z"], r[i])) for i in r_idx]
        V_asymp_at_r = [float(_z_asymptote(t["Z"], t["ion_charge"], r[i])) for i in r_idx]
        V_diff_bare = [a - b for a, b in zip(V_at_r, V_bare_at_r)]
        V_diff_asymp = [a - b for a, b in zip(V_at_r, V_asymp_at_r)]

        # 分桶 RMS：在 3 个区间 (r<0.5, 0.5<r<5, r>5) 计算 diff_rms
        r_mid1, r_mid2 = 0.5, 5.0
        diffs = np.array(V_diff_asymp)
        rms_core = float(np.sqrt(np.mean(diffs[r[r_idx] < r_mid1] ** 2))) if (r[r_idx] < r_mid1).any() else float("nan")
        rms_mid = float(np.sqrt(np.mean(diffs[(r[r_idx] >= r_mid1) & (r[r_idx] < r_mid2)] ** 2))) if ((r[r_idx] >= r_mid1) & (r[r_idx] < r_mid2)).any() else float("nan")
        rms_tail = float(np.sqrt(np.mean(diffs[r[r_idx] >= r_mid2] ** 2))) if (r[r_idx] >= r_mid2).any() else float("nan")

        out["systems"][name] = {
            "Z": t["Z"],
            "ion_charge": t["ion_charge"],
            "nele": t["Z"] - t["ion_charge"],
            "level_config": t["level_config"],
            "E_orb_first_active_ha": E_orb_first,
            "E_orb_first_active_eV": E_orb_first * HARTREE_EV,
            "E_orb_omega_weighted_ha": omega_weighted,
            "E_orb_omega_weighted_eV": omega_weighted * HARTREE_EV,
            "E_csf_ha": E_csf,
            "E_csf_eV": E_csf * HARTREE_EV if np.isfinite(E_csf) else float("nan"),
            "V_at_r": V_at_r,
            "V_bare_Zr_at_r": V_bare_at_r,
            "V_asymptote_at_r": V_asymp_at_r,
            "V_diff_from_asymptote": V_diff_asymp,
            "V_diff_from_bare": V_diff_bare,
            "rms_core_diff_from_asymp": rms_core,
            "rms_mid_diff_from_asymp": rms_mid,
            "rms_tail_diff_from_asymp": rms_tail,
        }

        print(f"  {name:12s} Z={t['Z']} ion={t['ion_charge']} N={t['Z']-t['ion_charge']}")
        print(f"     E_orb[1st active] = {E_orb_first*HARTREE_EV:+.3f} eV ({E_orb_first:+.6f} Ha)")
        print(f"     E_orb omega-weighted = {omega_weighted*HARTREE_EV:+.3f} eV")
        print(f"     E_csf = {E_csf*HARTREE_EV:+.3f} eV" if np.isfinite(E_csf) else f"     E_csf = NaN")
        print(f"     V(r) diff from latter asymptote: core(r<0.5)={rms_core:+.4f}  mid(0.5-5)={rms_mid:+.4f}  tail(r>5)={rms_tail:+.4f} Ha")
        # 打印 V_net 与两种参考线在 4 个代表点
        sample = [0, len(r_idx)//3, 2*len(r_idx)//3, len(r_idx)-1]
        print(f"     {'r':>8s}  {'V_net':>10s}  {'-(Z-N+1)/r':>12s}  {'diff':>10s}")
        for si in sample:
            i = r_idx[si]
            print(f"     {r[i]:>8.4f}  {V[i]:>+10.4f}  {V_asymp_at_r[si]:>+12.4f}  {V_diff_asymp[si]:>+10.4f}")

    # NIST 参考
    out["nist_reference"] = {
        "H_1s_abs_eV": -0.5 * HARTREE_EV,
        "H_2s_abs_eV": -0.125 * HARTREE_EV,
        "Heplus_1s_abs_eV": -2.0 * HARTREE_EV,
        "Heplus_2s_abs_eV": -0.5 * HARTREE_EV,
        "He_1s2_abs_eV": -108845.545 / 1000.0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")

    # 简短结论
    print()
    print("=" * 70)
    print("E_orb 偏浅根因诊断（核心结论）")
    print()
    print("对比 V(r) 在 3 个区间的 RMS 偏差（vs Latter tail 渐近线 -(Z-N+1)/r）")
    print()
    for name in [n for n, _ in targets]:
        s = out["systems"][name]
        print(f"  {name:12s}  core(r<0.5)={s['rms_core_diff_from_asymp']:+.4f}  mid={s['rms_mid_diff_from_asymp']:+.4f}  tail(r>5)={s['rms_tail_diff_from_asymp']:+.4f} Ha")
    print()
    print("判定规则:")
    print("  若 core 偏差 >> tail 偏差 → V_net 核区偏离 -Z/r 太多（密度加权 SCF 训练未约束核区）")
    print("  若 tail 偏差 >> core 偏差 → V_net 渐近线不满足 Latter tail（与 V_dfs 软约束冲突）")
    print("  若 mid 偏差最大 → 中间区 V_dfs 拟合偏移（scf_weight_mode=density 关注密度区）")


if __name__ == "__main__":
    main()
