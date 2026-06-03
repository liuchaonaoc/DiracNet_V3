#!/usr/bin/env python3
"""PDE radial diagnostic: pde(r), cumulative PDE, core/mid/tail, grid sweep.

Standalone script — does not modify core training code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_art.constants import hartree_to_meV
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset, load_manifest
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from pinn_art.training.checkpoint import load_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid

# Manifest row indices (Z=1..8, n=1..6)
TARGETS = {
    "N_1s": 36,
    "O_1s": 42,
    "O_2s": 43,
}

# Radial zones by cumulative grid-index fraction (loglinear grid → core = small r)
ZONE_FRACS = {
    "core": (0.0, 0.20),      # smallest 20% of points
    "mid": (0.20, 0.60),      # middle 40%
    "tail": (0.60, 1.00),     # outer 40%
}


def pde_radial_profile(out: dict, batch: dict, grid, orb_idx: int = 0) -> dict:
    """Per-grid-point PDE integrand and cumulative integral for one orbital."""
    wf = out["wavefunctions"]
    P = wf["P"][:, orb_idx, :]
    Q = wf["Q"][:, orb_idx, :]
    dP = wf["dPdr"][:, orb_idx, :]
    dQ = wf["dQdr"][:, orb_idx, :]
    V = out["V"]
    kappa = batch["kappa"][:, orb_idx : orb_idx + 1]
    r = grid.r

    LP, LQ = dirac_apply(P[:, None, :], Q[:, None, :], dP[:, None, :], dQ[:, None, :], V, kappa, r)
    E = orbital_energy_from_dirac(P[:, None, :], Q[:, None, :], LP, LQ, grid)[:, 0]

    res_P = LP[:, 0, :] - E[:, None] * P
    res_Q = LQ[:, 0, :] - E[:, None] * Q
    dens = (res_P ** 2 + res_Q ** 2)[0]  # [N_g]

    dr = grid.dr
    contrib = dens * dr
    total = float(jnp.sum(contrib))
    cum = jnp.cumsum(contrib)
    cum_frac = cum / jnp.clip(total, 1e-30)

    # Scalar PDE as in dirac_pde_loss (single orbital)
    pde_scalar = total

    return {
        "r": np.asarray(r, dtype=np.float64),
        "dr": np.asarray(dr, dtype=np.float64),
        "pde_density": np.asarray(dens, dtype=np.float64),
        "contrib": np.asarray(contrib, dtype=np.float64),
        "cum_frac": np.asarray(cum_frac, dtype=np.float64),
        "pde_scalar": float(pde_scalar),
        "E_orb": float(E[0]),
    }


def zone_integrals(r: np.ndarray, contrib: np.ndarray, fracs: dict) -> dict[str, float]:
    n = len(r)
    total = float(np.sum(contrib)) + 1e-30
    out = {}
    for name, (a, b) in fracs.items():
        i0 = int(np.floor(a * n))
        i1 = int(np.floor(b * n))
        if i1 <= i0:
            i1 = min(i0 + 1, n)
        out[f"{name}_pde"] = float(np.sum(contrib[i0:i1]))
        out[f"{name}_frac"] = out[f"{name}_pde"] / total
        out[f"{name}_r_lo"] = float(r[i0])
        out[f"{name}_r_hi"] = float(r[min(i1 - 1, n - 1)])
    return out


def run_one(
    params,
    model,
    ds: ManifestDataset,
    row_idx: int,
    grid,
    n_csf_max: int,
) -> dict:
    batch = collate_batches([ds[row_idx]], n_csf_max=n_csf_max)
    out = model.apply(params, batch, grid, train=False, return_ci=False)
    prof = pde_radial_profile(out, batch, grid, orb_idx=0)
    zones = zone_integrals(prof["r"], prof["contrib"], ZONE_FRACS)
    return {**prof, **zones}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_phase1_stage_a_z1_8_phase3.yaml")
    ap.add_argument("--ckpt", default="checkpoints/v3_phase1_stage_a_z1_8_phase3/stage_a_last.msgpack")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-grids", default="256,512,1024")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg.training.log_dir / "pde_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_grids = [int(x) for x in args.n_grids.split(",")]
    ds = ManifestDataset(ROOT / cfg.dataset.manifest, n_orb_max=int(cfg.model.n_orb_max))
    mdf = load_manifest(ROOT / cfg.dataset.manifest)
    params = load_params(ROOT / args.ckpt)
    n_csf = int(getattr(cfg.model, "n_csf_max", 8))

  # Base grid params from config
    r_min = float(cfg.grid.r_min)
    r_max = float(cfg.grid.r_max)
    scheme = str(cfg.grid.scheme)

    all_rows = []
    profiles = {}

    for ng in n_grids:
        grid = make_radial_grid(r_min, r_max, ng, scheme)
        model, _ = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
        for label, row_idx in TARGETS.items():
            prof = run_one(params, model, ds, row_idx, grid, n_csf)
            z = int(mdf.iloc[row_idx]["Z"])
            el = str(mdf.iloc[row_idx]["element"])
            lc = str(mdf.iloc[row_idx]["level_config"])
            zones = {k: prof[k] for k in prof if k.endswith("_pde") or k.endswith("_frac") or k.endswith("_r_lo") or k.endswith("_r_hi")}
            all_rows.append({
                "target": label,
                "Z": z,
                "element": el,
                "level_config": lc,
                "row": row_idx,
                "n_grid": ng,
                "pde_scalar": prof["pde_scalar"],
                "E_orb_meV": float(hartree_to_meV(prof["E_orb"])),
                **zones,
            })
            profiles[(label, ng)] = prof

    sweep_df = pd.DataFrame(all_rows)
    sweep_df.to_csv(out_dir / "pde_grid_sweep_summary.csv", index=False)

    # Save radial profiles for n_grid=256 (base config)
    base_ng = n_grids[0] if 256 in n_grids else n_grids[0]
    for label in TARGETS:
        if (label, base_ng) not in profiles:
            continue
        p = profiles[(label, base_ng)]
        pd.DataFrame({
            "r": p["r"],
            "pde_density": p["pde_density"],
            "contrib": p["contrib"],
            "cum_frac": p["cum_frac"],
        }).to_csv(out_dir / f"pde_radial_{label}_n{base_ng}.csv", index=False)

    # Markdown report
    lines = [
        "# PDE 径向诊断报告",
        "",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Config grid: `r_min={r_min}`, `r_max={r_max}`, `scheme={scheme}`",
        f"- 目标: **N 1s** (row 36), **O 1s** (row 42), **O 2s** (row 43)",
        "",
        "## 区域定义（按径向网格点序号比例）",
        "",
        "| 区域 | 网格点范围 | 说明 |",
        "|------|------------|------|",
        f"| **core** | 前 20% 点 | 小 r，库仑奇点附近 |",
        f"| **mid** | 20%–60% | 中间半径 |",
        f"| **tail** | 60%–100% | 大 r，渐近区 |",
        "",
        "标量 PDE = Σ (`res_P²` + `res_Q²`) × `dr`（与 `dirac_pde_loss` 单轨道一致）。",
        "",
        "## 1. Grid sweep：标量 PDE 与分区占比",
        "",
        sweep_df.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## 2. 各目标 @ n_grid=256：分区 PDE",
        "",
    ]

    sub256 = sweep_df[sweep_df["n_grid"] == base_ng]
    for _, row in sub256.iterrows():
        lines.append(f"### {row['target']} ({row['element']} {row['level_config']})")
        lines.append("")
        lines.append(f"- **pde_scalar** = {row['pde_scalar']:.4g}")
        lines.append(f"- **core** ({row['core_r_lo']:.3e}–{row['core_r_hi']:.3e} a.u.): "
                     f"pde={row['core_pde']:.4g}, **{100*row['core_frac']:.1f}%** of total")
        lines.append(f"- **mid** ({row['mid_r_lo']:.3e}–{row['mid_r_hi']:.3e}): "
                     f"pde={row['mid_pde']:.4g}, **{100*row['mid_frac']:.1f}%**")
        lines.append(f"- **tail** ({row['tail_r_lo']:.3e}–{row['tail_r_hi']:.3e}): "
                     f"pde={row['tail_pde']:.4g}, **{100*row['tail_frac']:.1f}%**")
        lines.append("")

    lines.extend([
        "## 3. Cumulative PDE 关键点（@ n_grid=256）",
        "",
        "累积分数 = 从内径向外积分 `contrib` 占总量比例。",
        "",
    ])

    for label in TARGETS:
        key = (label, base_ng)
        if key not in profiles:
            continue
        p = profiles[key]
        r, cf = p["r"], p["cum_frac"]
        lines.append(f"### {label}")
        lines.append("")
        for pct in [0.5, 0.8, 0.9, 0.95, 0.99]:
            idx = int(np.searchsorted(cf, pct))
            idx = min(idx, len(r) - 1)
            lines.append(f"- **{100*pct:.0f}%** 的 PDE 来自 r ≤ **{r[idx]:.4g}** a.u.")
        lines.append("")

    lines.extend([
        "## 4. Grid sweep 解读要点",
        "",
    ])

    for label in TARGETS:
        g = sweep_df[sweep_df["target"] == label]
        pde_vals = g.set_index("n_grid")["pde_scalar"]
        lines.append(f"- **{label}**: " + ", ".join(f"n={k} → {v:.4g}" for k, v in pde_vals.items()))
    lines.append("")
    lines.append("详细径向曲线见 `pde_radial_*_n256.csv`。")
    lines.append("")

    report_path = out_dir / "PDE_DIAGNOSTIC_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {out_dir / 'pde_grid_sweep_summary.csv'}")
    print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
