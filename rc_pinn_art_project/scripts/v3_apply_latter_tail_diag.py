#!/usr/bin/env python3
"""He 1s² E_orb 偏浅 34 eV 根因验证：纯 forward 后期校正（不重训）。

诊断结论（v3_verify_eorb_depend_on_r.py）:
  - 单电子 V_net 完美 (diff < 0.04 Ha)
  - He 1s² V_net 是 -2Z/r 形状 + Latter tail sign 错
  - → V_net 没经过 V_dfs 屏蔽 + Latter tail

本脚本做的事（不训练）:
  1. 跑 model.forward 拿到原始 P/Q/V_net/E_orb
  2. 用 P/Q/omega 重新计算 rho, V_H, V_x (fermi_amaldi=True)
  3. 构建 V_rebuilt = -Z/r + (N-1)/N (V_H + V_x) + latter_tail
  4. 重新解 Dirac (期望值方式) 拿到 E_orb_rebuilt
  5. 对比: E_orb_orig vs E_orb_rebuilt vs E_csf vs NIST

如果 E_orb_rebuilt → NIST 接近 1 eV, 故障 = SCF 软约束未紧
  → 解法: 重训时加大 scf_weight + 确认 latter_tail 路径
如果 E_orb_rebuilt 仍然偏浅 30+ eV, 故障 = rho 本身就错
  → 解法: rho 重算 + 物理软约束

跑法:
  cd rc_pinn_art_project && export PYTHONPATH=. && export JAX_PLATFORMS=cuda
  python scripts/v3_apply_latter_tail_diag.py \\
    --ckpt checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack \\
    --manifest data_cache/manifest_z_le_2.parquet \\
    --out logs/path_a_latter_corr/eprime_latter_corr.json
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

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from pinn_art.physics.dfs_potential import (
    build_dfs_potential,
    electron_density,
    hartree_potential,
    slater_exchange_potential,
    fermi_amaldi_factor,
    latter_tail_correction,
)
from pinn_art.training.checkpoint import load_params, merge_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid

HARTREE_EV = 27.211386245988
HARTREE_MEV = HARTREE_EV * 1000.0


def _e_orb_with_rebuilt_V(
    P, Q, dPdr, dQdr, kappa, V, r, grid, orb_mask,
):
    """重新解 Dirac 期望值 <P|H|P> (用新 V)"""
    LP, LQ = dirac_apply(P, Q, dPdr, dQdr, V, kappa, r)
    E = orbital_energy_from_dirac(P, Q, LP, LQ, grid)
    return jnp.where(orb_mask, E, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_stage_a_z1_26_n10.yaml")
    ap.add_argument("--ckpt", default="checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack")
    ap.add_argument("--manifest", default="data_cache/manifest_z_le_2.parquet")
    ap.add_argument("--out", default="logs/path_a_latter_corr/eprime_latter_corr.json")
    args = ap.parse_args()

    print(f"[1/4] 加载 config + grid + 模型")
    cfg = load_config(ROOT / args.config)
    grid = make_radial_grid(
        r_min=float(cfg.grid.r_min), r_max=float(cfg.grid.r_max),
        n_grid=int(cfg.grid.n_grid), scheme=str(cfg.grid.scheme),
    )
    n_orb_max = int(cfg.model.n_orb_max)
    n_csf_max = int(getattr(cfg.model, "n_csf_max", 8))
    model, params = build_model_and_params(cfg, grid, key=jax.random.PRNGKey(0))

    print(f"[2/4] 加载 ckpt (注入 CI 模板)")
    ckpt_params = load_params(args.ckpt)
    from pinn_art.models.pinn_art_model import _dummy_batch
    dummy = _dummy_batch(n_orb_max, n_csf_max)
    params_ci = model.init(jax.random.PRNGKey(1), dummy, grid, train=False, return_ci=True)
    params = merge_params(params_ci, ckpt_params)

    print(f"[3/4] 加载 manifest + 跑 forward (5 体系)")
    ds = ManifestDataset(args.manifest, n_orb_max=n_orb_max, n_csf_max=n_csf_max)
    df = ds.df
    targets = [
        ("H_1s",     {"Z": 1, "ion_charge": 0, "level_config": "1s1"}),
        ("H_2s",     {"Z": 1, "ion_charge": 0, "level_config": "2s1"}),
        ("Heplus_1s", {"Z": 2, "ion_charge": 1, "level_config": "1s1"}),
        ("Heplus_2s", {"Z": 2, "ion_charge": 1, "level_config": "2s1"}),
        ("He_1s2",   {"Z": 2, "ion_charge": 0, "level_config": "1s2"}),
    ]

    out = {
        "config": args.config,
        "ckpt": args.ckpt,
        "manifest": args.manifest,
        "diagnostic_r": [float(grid.r[i]) for i in [0, 5, 10, 20, 50, 100, 200, min(400, len(grid.r)-1), min(600, len(grid.r)-1)]],
        "systems": {},
    }

    n_g = len(grid.r)
    diag_idx = [i for i in [0, 5, 10, 20, 50, 100, 200, 400, 600] if i < n_g]
    if len(diag_idx) < 9:
        print(f"[Info] grid 只有 {n_g} 点, diag 索引裁剪为 {diag_idx}")

    for name, t in targets:
        k = (t["Z"], t["ion_charge"], t["level_config"])
        hits = df[(df["Z"] == k[0]) & (df["ion_charge"] == k[1]) & (df["level_config"] == k[2])]
        if len(hits) == 0:
            print(f"  跳过 {name}: manifest 缺 {k}")
            continue
        idx = int(hits.index[0])
        batch = collate_batches([ds[idx]], n_csf_max=n_csf_max)
        out_model = model.apply(params, batch, grid, train=False, return_ci=True)

        V_orig = np.asarray(out_model["V"][0])        # [N_g]
        # 保留 batch 维: out_model 字典里的 P/Q 都是 [B, N_orb, N_g]
        P = np.asarray(out_model["wavefunctions"]["P"])         # [1, N_orb, N_g]
        Q = np.asarray(out_model["wavefunctions"]["Q"])
        dPdr = np.asarray(out_model["wavefunctions"]["dPdr"])
        dQdr = np.asarray(out_model["wavefunctions"]["dQdr"])
        kappa = np.asarray(batch["kappa"][:, :n_orb_max])       # [1, N_orb]
        omega_b = np.asarray(batch["omega"][:, :n_orb_max])     # [1, N_orb]
        orb_mask_b = np.asarray(batch["orb_mask"][:, :n_orb_max]).astype(bool)  # [1, N_orb]
        # 兼容 1D omega/orb_mask 用于原始 E_orb 提取
        omega = omega_b[0]
        orb_mask = orb_mask_b[0]
        Z = np.array([t["Z"]], dtype=np.float32)
        nele = np.array([t["Z"] - t["ion_charge"]], dtype=np.float32)

        # === 原始 E_orb (模型直接 forward 得到) ===
        E_orb_orig = np.asarray(out_model["E_orb"][0])  # [N_orb]
        first_active = int(np.where(orb_mask)[0][0]) if orb_mask.any() else 0
        E_orb_orig_first = float(E_orb_orig[first_active])
        omega_weighted_orig = float(np.sum(E_orb_orig * omega * orb_mask))

        # === 重建 V_dfs（开启 fermi_amaldi + latter_tail）===
        # 关键: 全部 [B, ...] 形状, 喂给 dfs_potential 的接口
        P_j = jnp.asarray(P)                # [1, N_orb, N_g]
        Q_j = jnp.asarray(Q)
        omega_j = jnp.asarray(omega_b)      # [1, N_orb]
        Z_j = jnp.asarray(Z)                # [1]
        nele_j = jnp.asarray(nele)          # [1]

        rho = electron_density(P_j, Q_j, omega_j)         # [1, N_g]
        V_nuc = -Z_j[:, None] / jnp.clip(grid.r[None, :], 1e-8)   # [1, N_g]
        V_H = hartree_potential(rho, grid)                          # [1, N_g]
        V_x = slater_exchange_potential(rho, grid, alpha_x=1.0)     # [1, N_g]
        sic = fermi_amaldi_factor(nele_j)                            # [1, 1]
        V_ee = sic * (V_H + V_x)                                     # [1, N_g]
        V_dfs_no_tail = (V_nuc + V_ee)                               # [1, N_g]
        V_dfs = latter_tail_correction(V_dfs_no_tail, Z_j, nele_j, grid)  # [1, N_g]
        # 转回 numpy [N_g] 便于混合
        V_dfs_raw_np = np.asarray(V_dfs_no_tail[0])     # 关闭 latter_tail 的版本
        V_dfs_np = np.asarray(V_dfs[0])                  # 开启 latter_tail 的版本

        # SCF 混合: V_rebuilt = V_orig + scf_alpha * (V_dfs - V_dfs_raw)
        # 当 scf_alpha=1: V_rebuilt 完全是 V_dfs (物理解)
        for scf_alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            V_rebuilt_np = V_orig + scf_alpha * (V_dfs_np - V_dfs_raw_np)
            E_rebuilt = _e_orb_with_rebuilt_V(
                P_j, Q_j,
                jnp.asarray(dPdr), jnp.asarray(dQdr),
                jnp.asarray(kappa),              # [1, N_orb]
                jnp.asarray(V_rebuilt_np[None, :]),  # [1, N_g]
                grid.r, grid,
                jnp.asarray(orb_mask_b),         # [1, N_orb]
            )
            E_rebuilt_arr = np.asarray(E_rebuilt)[0]  # [N_orb]
            E_first = float(E_rebuilt_arr[first_active])
            E_omega = float(np.sum(E_rebuilt_arr * omega * orb_mask))
            print(f"  {name:12s} scf_alpha={scf_alpha:.2f}  E_orb[1st]={E_first*HARTREE_EV:+.3f} eV  E_orb_omega={E_omega*HARTREE_EV:+.3f} eV")

        # 把 scf_alpha=0, 0.5, 1.0 三组结果存下来
        results_per_alpha = {}
        for scf_alpha in [0.0, 0.5, 1.0]:
            V_rebuilt_np = V_orig + scf_alpha * (V_dfs_np - V_dfs_raw_np)
            E_rebuilt = _e_orb_with_rebuilt_V(
                P_j, Q_j, jnp.asarray(dPdr), jnp.asarray(dQdr),
                jnp.asarray(kappa), jnp.asarray(V_rebuilt_np[None, :]),
                grid.r, grid, jnp.asarray(orb_mask_b),
            )
            E_rebuilt_arr = np.asarray(E_rebuilt)[0]
            E_first = float(E_rebuilt_arr[first_active])
            E_omega = float(np.sum(E_rebuilt_arr * omega * orb_mask))
            results_per_alpha[f"scf_alpha_{scf_alpha}"] = {
                "E_orb_first_active_ha": E_first,
                "E_orb_first_active_eV": E_first * HARTREE_EV,
                "E_orb_omega_weighted_ha": E_omega,
                "E_orb_omega_weighted_eV": E_omega * HARTREE_EV,
            }

        # E_csf (原始) = forward 自带，形状 [B, n_csf_max]
        E_csf_arr = out_model.get("E_csf")
        if E_csf_arr is not None:
            E_csf = float(np.asarray(E_csf_arr[0]).min())  # 取 ground state
        else:
            E_csf = float("nan")

        out["systems"][name] = {
            "Z": t["Z"],
            "ion_charge": t["ion_charge"],
            "nele": t["Z"] - t["ion_charge"],
            "level_config": t["level_config"],
            "E_orb_orig_first_ha": E_orb_orig_first,
            "E_orb_orig_first_eV": E_orb_orig_first * HARTREE_EV,
            "E_orb_orig_omega_ha": omega_weighted_orig,
            "E_orb_orig_omega_eV": omega_weighted_orig * HARTREE_EV,
            "E_csf_orig_ha": E_csf,
            "E_csf_orig_eV": E_csf * HARTREE_EV if np.isfinite(E_csf) else float("nan"),
            "V_orig_at_r": [float(V_orig[i]) for i in diag_idx],
            "V_dfs_at_r":   [float(V_dfs_np[i]) for i in diag_idx],
            "V_diff_Vdfs_minus_Vorig_at_r": [float(V_dfs_np[i] - V_orig[i]) for i in diag_idx],
            "results_per_scf_alpha": results_per_alpha,
        }

    out["nist_reference"] = {
        "H_1s_abs_eV": -0.5 * HARTREE_EV,
        "H_2s_abs_eV": -0.125 * HARTREE_EV,
        "Heplus_1s_abs_eV": -2.0 * HARTREE_EV,
        "Heplus_2s_abs_eV": -0.5 * HARTREE_EV,
        "He_1s2_abs_eV": -108.845545,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")

    # 简短结论
    print()
    print("=" * 80)
    print("He 1s² V_net 后期校正验证")
    print()
    print("期望: 当 scf_alpha=1.0 (V 完全是 V_dfs 物理解) 时:")
    print("  若 E_orb_rebuilt 接近 NIST → SCF 软约束未紧, 重训时加大 scf_weight")
    print("  若 E_orb_rebuilt 仍然偏浅 30+ eV → rho 本身就错, 需 rho 重算")
    print()
    for name in [n for n, _ in targets]:
        if name not in out["systems"]:
            continue
        s = out["systems"][name]
        r = s["results_per_scf_alpha"]
        e_orig = s["E_orb_orig_omega_eV"]
        e_a0 = r["scf_alpha_0.0"]["E_orb_omega_weighted_eV"]
        e_a1 = r["scf_alpha_1.0"]["E_orb_omega_weighted_eV"]
        print(f"  {name:12s}  orig={e_orig:+8.3f}  scf_alpha=0={e_a0:+8.3f}  scf_alpha=1={e_a1:+8.3f}  eV")


if __name__ == "__main__":
    main()
