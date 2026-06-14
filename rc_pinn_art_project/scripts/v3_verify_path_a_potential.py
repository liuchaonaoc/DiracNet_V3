#!/usr/bin/env python3
"""Path A 物理注入诊断：纯 forward, 把 R^k 注入 V_dfs 后期校正 V_rebuilt,
对比 E_orb_rebuilt vs NIST.

注意 (§11.2): E_orb 对 V 的修改免疫, 纯 forward 校正可能改不了 E_orb.
本脚本目的:
  1. 看 V_slater_corr(r) 的形态
  2. 看 V_rebuilt 注入 R^k 后是否仍"失效"(确认物理假设)
  3. 若 R^k 注入后 V 在 |ψ|² 峰值区变化 > 0.1 Ha → 才是有效物理修正

跑法:
  cd rc_pinn_art_project && export PYTHONPATH=. && export JAX_PLATFORMS=cuda
  python scripts/v3_verify_path_a_potential.py \\
    --ckpt checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack \\
    --manifest data_cache/manifest_z_le_2.parquet \\
    --out logs/path_a_Rk_inject/eprime_Rk_inject.json \\
    --alpha-list 0.0 0.5 1.0 2.0
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

from pinn_art.ci.slater_radial import compute_all_Rk_diagonal
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.dfs_potential import (
    electron_density,
    fermi_amaldi_factor,
    hartree_potential,
    latter_tail_correction,
    slater_exchange_potential,
)
from pinn_art.physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from pinn_art.physics.slater_correction import slater_correction_potential
from pinn_art.training.checkpoint import load_params, merge_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid

HARTREE_EV = 27.211386245988
HARTREE_MEV = HARTREE_EV * 1000.0


def _e_orb_rebuilt(P_j, Q_j, dPdr_j, dQdr_j, kappa_j, V_rebuilt, grid, orb_mask_j):
    """重新解 Dirac 期望值, 给定新 V."""
    LP, LQ = dirac_apply(P_j, Q_j, dPdr_j, dQdr_j, V_rebuilt, kappa_j, grid.r)
    E = orbital_energy_from_dirac(P_j, Q_j, LP, LQ, grid)
    return jnp.where(orb_mask_j, E, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_stage_a_z1_26_n10.yaml")
    ap.add_argument("--ckpt", default="checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack")
    ap.add_argument("--manifest", default="data_cache/manifest_z_le_2.parquet")
    ap.add_argument("--out", default="logs/path_a_Rk_inject/eprime_Rk_inject.json")
    ap.add_argument("--alpha-list", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    args = ap.parse_args()

    print(f"[1/4] 加载 config + grid + 模型")
    cfg = load_config(ROOT / args.config)
    grid = make_radial_grid(
        r_min=float(cfg.grid.r_min), r_max=float(cfg.grid.r_max),
        n_grid=int(cfg.grid.n_grid), scheme=str(cfg.grid.scheme),
    )
    n_orb_max = int(cfg.model.n_orb_max)
    n_csf_max = int(getattr(cfg.model, "n_csf_max", 8))
    k_list = list(getattr(getattr(cfg, "ci", None), "k_list", [0, 1, 2]))
    k_tuple = tuple(k_list)
    model, params = build_model_and_params(cfg, grid, key=jax.random.PRNGKey(0))

    print(f"[2/4] 加载 ckpt (注入 CI 模板, 启用 return_ci)")
    ckpt_params = load_params(args.ckpt)
    from pinn_art.models.pinn_art_model import _dummy_batch
    dummy = _dummy_batch(n_orb_max, n_csf_max)
    params_ci = model.init(jax.random.PRNGKey(1), dummy, grid, train=False, return_ci=True)
    params = merge_params(params_ci, ckpt_params)

    print(f"[3/4] 加载 manifest")
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
        "alpha_list": args.alpha_list,
        "diagnostic_r": [],
        "systems": {},
    }
    n_g = len(grid.r)
    diag_idx = [i for i in [0, 5, 10, 20, 50, 100, 200, min(400, n_g - 1), min(600, n_g - 1)] if 0 <= i < n_g]
    out["diagnostic_r"] = [float(grid.r[i]) for i in diag_idx]

    print(f"[4/4] 跑 5 体系 × {len(args.alpha_list)} 个 alpha (注入 R^k 强度)")
    for name, t in targets:
        k = (t["Z"], t["ion_charge"], t["level_config"])
        hits = df[(df["Z"] == k[0]) & (df["ion_charge"] == k[1]) & (df["level_config"] == k[2])]
        if len(hits) == 0:
            print(f"  跳过 {name}")
            continue
        idx = int(hits.index[0])
        batch = collate_batches([ds[idx]], n_csf_max=n_csf_max)
        out_model = model.apply(params, batch, grid, train=False, return_ci=True)

        V_orig = np.asarray(out_model["V"][0])
        P = np.asarray(out_model["wavefunctions"]["P"])        # [1, N, N_g]
        Q = np.asarray(out_model["wavefunctions"]["Q"])
        dPdr = np.asarray(out_model["wavefunctions"]["dPdr"])
        dQdr = np.asarray(out_model["wavefunctions"]["dQdr"])
        kappa = np.asarray(batch["kappa"][:, :n_orb_max])      # [1, N]
        omega_b = np.asarray(batch["omega"][:, :n_orb_max])    # [1, N]
        orb_mask_b = np.asarray(batch["orb_mask"][:, :n_orb_max]).astype(bool)  # [1, N]
        omega = omega_b[0]; orb_mask = orb_mask_b[0]
        Z = np.array([t["Z"]], dtype=np.float32)
        nele = np.array([t["Z"] - t["ion_charge"]], dtype=np.float32)
        first_active = int(np.where(orb_mask)[0][0]) if orb_mask.any() else 0

        # 1) 计算 R^k (来自 model 内部)
        Rk_model = out_model.get("Rk")  # [B, n_k, N_orb]
        if Rk_model is None:
            # fallback: 重新算
            Rk_model = compute_all_Rk_diagonal(
                jnp.asarray(P), jnp.asarray(Q), orb_mask_b, grid, k_list=k_tuple
            )
        # slater_log_scale 从 params 取
        try:
            slater_log_scale = np.asarray(params["params"]["slater_log_scale"])
        except (KeyError, TypeError):
            slater_log_scale = np.zeros(len(k_list), dtype=np.float32)
        # 归一化: Rk_model = 原始 R^k (不带 scale), slater_log_scale=0 → scale=1
        # 不, model 内部 Rk = compute_all_Rk_diagonal * exp(slater_log_scale), 已带 scale
        # 这里为纯 forward 实验, 我们用"未带 scale 的 Rk"+ slater_log_scale=0
        Rk_raw = compute_all_Rk_diagonal(
            jnp.asarray(P), jnp.asarray(Q), orb_mask_b, grid, k_list=k_tuple
        )
        slater_log_scale_j = jnp.zeros(len(k_list))

        # 2) 算 V_dfs (与 §11 一样)
        P_j, Q_j = jnp.asarray(P), jnp.asarray(Q)
        omega_j, Z_j, nele_j = jnp.asarray(omega_b), jnp.asarray(Z), jnp.asarray(nele)
        rho = electron_density(P_j, Q_j, omega_j)               # [1, N_g]
        V_nuc = -Z_j[:, None] / jnp.clip(grid.r[None, :], 1e-8)  # [1, N_g]
        V_H = hartree_potential(rho, grid)                        # [1, N_g]
        V_x = slater_exchange_potential(rho, grid, alpha_x=1.0)   # [1, N_g]
        sic = fermi_amaldi_factor(nele_j)                          # [1, 1]
        V_ee = sic * (V_H + V_x)                                   # [1, N_g]
        V_dfs_no_tail = (V_nuc + V_ee)                             # [1, N_g]
        V_dfs = latter_tail_correction(V_dfs_no_tail, Z_j, nele_j, grid)  # [1, N_g]
        V_dfs_np = np.asarray(V_dfs[0])

        # 3) 算 V_slater_corr
        V_slater = slater_correction_potential(
            P_j, Q_j, omega_j, Rk_raw, slater_log_scale_j, rho
        )  # [1, N_g]
        V_slater_np = np.asarray(V_slater[0])

        # 4) 多 α 扫描
        results_per_alpha = {}
        for scf_alpha in args.alpha_list:
            V_rebuilt_np = V_orig + scf_alpha * (V_dfs_np + V_slater_np - V_orig)
            E_rebuilt = _e_orb_rebuilt(
                P_j, Q_j, jnp.asarray(dPdr), jnp.asarray(dQdr),
                jnp.asarray(kappa), jnp.asarray(V_rebuilt_np[None, :]),
                grid, jnp.asarray(orb_mask_b),
            )
            E_rebuilt_arr = np.asarray(E_rebuilt)[0]
            E_first = float(E_rebuilt_arr[first_active])
            E_omega = float(np.sum(E_rebuilt_arr * omega * orb_mask))
            results_per_alpha[f"alpha_{scf_alpha}"] = {
                "E_orb_first_active_ha": E_first,
                "E_orb_first_active_eV": E_first * HARTREE_EV,
                "E_orb_omega_weighted_ha": E_omega,
                "E_orb_omega_weighted_eV": E_omega * HARTREE_EV,
            }
            print(f"  {name:12s} α={scf_alpha:.2f}  E_orb[1st]={E_first*HARTREE_EV:+.3f} eV  E_orb_ω={E_omega*HARTREE_EV:+.3f} eV")

        # V_orig 原始 E_orb
        E_orb_orig = np.asarray(out_model["E_orb"][0])
        first_active = int(np.where(orb_mask)[0][0]) if orb_mask.any() else 0
        E_orb_orig_first = float(E_orb_orig[first_active])
        omega_weighted_orig = float(np.sum(E_orb_orig * omega * orb_mask))

        # E_csf (CI)
        E_csf_arr = out_model.get("E_csf")
        E_csf = float(np.asarray(E_csf_arr[0]).min()) if E_csf_arr is not None else float("nan")

        out["systems"][name] = {
            "Z": t["Z"], "ion_charge": t["ion_charge"],
            "nele": t["Z"] - t["ion_charge"],
            "level_config": t["level_config"],
            "E_orb_orig_first_eV": E_orb_orig_first * HARTREE_EV,
            "E_orb_orig_omega_eV": omega_weighted_orig * HARTREE_EV,
            "E_csf_orig_eV": E_csf * HARTREE_EV if np.isfinite(E_csf) else float("nan"),
            "V_orig_at_r": [float(V_orig[i]) for i in diag_idx],
            "V_dfs_at_r":   [float(V_dfs_np[i]) for i in diag_idx],
            "V_slater_at_r": [float(V_slater_np[i]) for i in diag_idx],
            "V_dfs_plus_slater_at_r": [float(V_dfs_np[i] + V_slater_np[i]) for i in diag_idx],
            "results_per_alpha": results_per_alpha,
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
    print("Path A 物理注入诊断 (R^k→V_dfs 后期校正)")
    print()
    print("注意: §11.2 已证明 E_orb 对 V 修改免疫 (除非 P 在 ΔV 大处有大振幅)")
    print("本脚本若 α=1.0 仍 E_orb 不变 → 确认需要重训, 不能纯 forward 修")
    print()
    for name in [n for n, _ in targets]:
        if name not in out["systems"]:
            continue
        s = out["systems"][name]
        r = s["results_per_alpha"]
        e_orig = s["E_orb_orig_omega_eV"]
        e_a0 = r[f"alpha_{args.alpha_list[0]}"]["E_orb_omega_weighted_eV"]
        e_a1 = r[f"alpha_{args.alpha_list[-1]}"]["E_orb_omega_weighted_eV"]
        # V_slater RMS (vs 0) at diag points
        v_sl = np.array(s["V_slater_at_r"])
        v_sl_rms = float(np.sqrt(np.mean(v_sl ** 2)))
        print(f"  {name:12s}  E_orb orig={e_orig:+8.3f}  α=0={e_a0:+8.3f}  α={args.alpha_list[-1]}={e_a1:+8.3f} eV  V_slater RMS={v_sl_rms:+.4f} Ha")


if __name__ == "__main__":
    main()
