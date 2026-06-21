#!/usr/bin/env python3
"""Three-way energy comparison: NIST (analytic hydrogenic) vs cFAC
(single-electron Dirac) vs PINN-ART Stage-A, on the manifest rows of
manifest_hydrogenic_z1_26_n10.parquet (260 rows: Z=1..26 × n=1..10).

For each row we compute the orbital energy E_orb in Hartree:
    * NIST ground truth: -Z^2 / (2 n^2)  (analytic hydrogenic, no Dirac correction)
    * cFAC:                parsed from the `# energy` comment in cf_Z<Z>_n<N>_PQ.dat
    * PINN-ART:            E_orb[slot] returned by model.apply on the manifest row

Per-row errors (ΔE in meV) are written to a CSV; per-Z RMSE/max-abs
errors are summarized; and three 1:1 plots are saved.

Inputs:
    --ckpt PATH        PINN-ART Stage-A checkpoint
    --config PATH      Stage-A YAML config
    --manifest PATH    Hydrogenic manifest (must have Z, level_config, has_nist_level)
    --fac-dir PATH     Directory holding cf_Z<Z>_n<N>_PQ.dat (default: energy_batch/)
    --out-dir PATH     Directory for outputs (default: energy_batch/energy_compare)

Outputs (under --out-dir):
    energy_comparison.csv      full 260-row table
    per_Z_stats.csv            per-Z RMSE/max/mean statistics
    energy_1to1.png            three 1:1 plots (PINN vs NIST, FAC vs NIST, PINN vs FAC)
    per_Z_RMSE.png             per-Z RMSE / max |ΔE| bar charts
    error_histograms.png       |ΔE| and signed-ΔE histograms

Example:
    python compare_energy_3way.py
    python compare_energy_3way.py --ckpt /path/to/new_ckpt.msgpack
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path("/home/chaos/workspace2/DiracNet_V3")
PROJ = ROOT / "rc_pinn_art_project"
JOBS = ROOT / "cfac_jobs" / "energy_batch"

sys.path.insert(0, str(PROJ))

import numpy as np
import pandas as pd
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinn_art.constants import hartree_to_meV
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.training.checkpoint import load_params


def parse_fac_energy(z: int, n: int, fac_dir: Path) -> float | None:
    """Read `# energy = -1.36058544E+01` (in eV) and convert to Hartree."""
    p = fac_dir / f"cf_Z{z}_n{n}_PQ.dat"
    if not p.exists():
        return None
    with p.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith("# energy"):
                m = re.search(r"=\s*(-?\d+\.\d+E[+-]\d+)", ln)
                if m:
                    return float(m.group(1)) / 27.211386245988
    return None


def run_pinn(model, params, ds, grid, n_csf_max: int, df: pd.DataFrame) -> list[float]:
    """Return one PINN E_orb per manifest row, in Hartree."""
    E_list = []
    for i in range(len(ds)):
        batch = collate_batches([ds[i]], n_csf_max=n_csf_max)
        out = model.apply(params, batch, grid, train=False, return_ci=False)
        E_h = np.asarray(out["E_orb"][0])    # [N_orb]
        sh = np.asarray(batch["shell_table"])[0]
        n_target = int(str(df.iloc[i]["level_config"])[:-2])
        slot = 0
        for a in range(E_h.shape[0]):
            if int(sh[a, 0]) == n_target and int(sh[a, 1]) == 0:
                slot = a
                break
        E_list.append(float(E_h[slot]))
    return E_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=PROJ / "checkpoints/v3_stage_a_laguerre_basis_c_c2/stage_a_last.msgpack")
    ap.add_argument("--config", type=Path,
                    default=PROJ / "configs/v3_stage_a_laguerre_basis_c.yaml")
    ap.add_argument("--manifest", type=Path,
                    default=PROJ / "data_cache/manifest_hydrogenic_z1_26_n10.parquet")
    ap.add_argument("--fac-dir", type=Path,
                    default=JOBS)
    ap.add_argument("--out-dir", type=Path,
                    default=JOBS / "energy_compare")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Manifest ---
    df = pd.read_parquet(args.manifest)
    if "has_nist_level" not in df.columns:
        raise RuntimeError(f"Manifest {args.manifest} has no has_nist_level column")
    n_total = len(df)

    # --- 2. NIST ground truth: analytic hydrogenic ---
    n_q = df["level_config"].str.extract(r"(\d+)")[0].astype(float)
    df["E_nist_hartree"] = -df["Z"].astype(float) ** 2 / (2 * n_q ** 2)

    # --- 3. FAC: parse every PQ.dat ---
    df["E_fac_hartree"] = [
        parse_fac_energy(int(r.Z), int(r.level_config[:-2]), args.fac_dir)
        for r in df.itertuples()
    ]

    # --- 4. PINN: run checkpoint ---
    print(f"[load] ckpt={args.ckpt}")
    print(f"[load] config={args.config}")
    cfg = load_config(args.config)
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max),
        int(cfg.grid.n_grid), str(cfg.grid.scheme),
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    params = load_params(args.ckpt)
    ds = ManifestDataset(args.manifest,
                         n_orb_max=int(cfg.model.n_orb_max),
                         n_csf_max=int(cfg.model.n_csf_max))
    df["E_pinn_hartree"] = run_pinn(model, params, ds, grid,
                                    int(cfg.model.n_csf_max), df)

    # --- 5. ΔE in meV ---
    H_to_meV = hartree_to_meV(1.0)
    df["ΔE_PINN_meV"] = (df["E_pinn_hartree"] - df["E_nist_hartree"]) * H_to_meV
    df["ΔE_FAC_meV"] = (df["E_fac_hartree"] - df["E_nist_hartree"]) * H_to_meV
    df["ΔE_PINN_FAC_meV"] = (df["E_pinn_hartree"] - df["E_fac_hartree"]) * H_to_meV
    df["E_nist_meV"] = df["E_nist_hartree"] * H_to_meV

    csv_path = args.out_dir / "energy_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"saved -> {csv_path}")

    # --- 6. Statistics ---
    def stats(name, vals):
        vals = vals.dropna()
        if len(vals) == 0:
            return {"name": name, "n": 0, "mean_meV": float("nan"),
                    "median_meV": float("nan"), "std_meV": float("nan"),
                    "max_abs_meV": float("nan"), "rmse_meV": float("nan")}
        return {
            "name": name,
            "n": int(len(vals)),
            "mean_meV": float(vals.mean()),
            "median_meV": float(vals.median()),
            "std_meV": float(vals.std()),
            "max_abs_meV": float(vals.abs().max()),
            "rmse_meV": float(np.sqrt(np.mean(vals ** 2))),
        }

    overall = [
        stats("PINN vs NIST", df["ΔE_PINN_meV"]),
        stats("FAC  vs NIST", df["ΔE_FAC_meV"]),
        stats("PINN vs FAC",  df["ΔE_PINN_FAC_meV"]),
    ]
    print("\n== Overall energy error statistics (meV) ==")
    for s in overall:
        print(f"  {s['name']:14s}  N={s['n']:3d}  "
              f"mean={s['mean_meV']:+9.2f}  median={s['median_meV']:+9.2f}  "
              f"std={s['std_meV']:8.2f}  max|Δ|={s['max_abs_meV']:10.2f}  "
              f"RMSE={s['rmse_meV']:10.2f}")

    # Per-Z
    rows = []
    for z, g in df.groupby("Z"):
        rows.append({
            "Z": int(z),
            "N": int(len(g)),
            "PINN_mean_meV": g["ΔE_PINN_meV"].mean(),
            "PINN_max_abs_meV": g["ΔE_PINN_meV"].abs().max(),
            "PINN_rmse_meV": np.sqrt(np.mean(g["ΔE_PINN_meV"] ** 2)),
            "FAC_mean_meV": g["ΔE_FAC_meV"].mean(),
            "FAC_max_abs_meV": g["ΔE_FAC_meV"].abs().max(),
            "FAC_rmse_meV": np.sqrt(np.mean(g["ΔE_FAC_meV"] ** 2)),
        })
    per_z = pd.DataFrame(rows)
    per_z_path = args.out_dir / "per_Z_stats.csv"
    per_z.to_csv(per_z_path, index=False)
    print(f"saved -> {per_z_path}")
    print("\n== Per-Z summary (head) ==")
    print(per_z.head(10).to_string(index=False))

    # --- 7. Plots ---
    def one_to_one(ax, x_h, y_h, x_label, y_label, title):
        x = np.asarray(x_h) * H_to_meV
        y = np.asarray(y_h) * H_to_meV
        ax.scatter(x, y, s=12, alpha=0.55, edgecolor='none')
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.7, label="y=x")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel(f"{x_label} (meV)")
        ax.set_ylabel(f"{y_label} (meV)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    one_to_one(axes[0], df["E_nist_hartree"], df["E_pinn_hartree"],
               "E_NIST (analytic)", "E_PINN", "PINN vs NIST")
    one_to_one(axes[1], df["E_nist_hartree"], df["E_fac_hartree"],
               "E_NIST (analytic)", "E_FAC", "cFAC vs NIST")
    one_to_one(axes[2], df["E_fac_hartree"], df["E_pinn_hartree"],
               "E_FAC", "E_PINN", "PINN vs cFAC")
    fig.tight_layout()
    fig.savefig(args.out_dir / "energy_1to1.png", dpi=110)
    plt.close(fig)
    print(f"saved -> {args.out_dir / 'energy_1to1.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    z_vals = per_z["Z"].to_numpy()
    axes[0].bar(z_vals, per_z["PINN_rmse_meV"], color="tab:red", alpha=0.7, label="PINN")
    axes[0].bar(z_vals, per_z["FAC_rmse_meV"], color="tab:blue", alpha=0.5, label="cFAC")
    axes[0].set_xlabel("Z")
    axes[0].set_ylabel("RMSE ΔE (meV)")
    axes[0].set_title("Per-Z RMSE  (PINN vs NIST, FAC vs NIST)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(z_vals, per_z["PINN_max_abs_meV"], color="tab:red", alpha=0.7, label="PINN")
    axes[1].bar(z_vals, per_z["FAC_max_abs_meV"], color="tab:blue", alpha=0.5, label="cFAC")
    axes[1].set_xlabel("Z")
    axes[1].set_ylabel("max |ΔE| (meV)")
    axes[1].set_title("Per-Z max |ΔE|")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "per_Z_RMSE.png", dpi=110)
    plt.close(fig)
    print(f"saved -> {args.out_dir / 'per_Z_RMSE.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(np.abs(df["ΔE_PINN_meV"]), bins=50, color="tab:red", alpha=0.7, label="PINN")
    axes[0].hist(np.abs(df["ΔE_FAC_meV"]), bins=50, color="tab:blue", alpha=0.5, label="cFAC")
    axes[0].set_xlabel("|ΔE| (meV)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Distribution of |ΔE| vs analytic")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(df["ΔE_PINN_meV"], bins=50, color="tab:red", alpha=0.7, label="PINN - NIST")
    axes[1].hist(df["ΔE_FAC_meV"], bins=50, color="tab:blue", alpha=0.5, label="cFAC - NIST")
    axes[1].axvline(0, color="k", linestyle="--", lw=0.7)
    axes[1].set_xlabel("ΔE (meV)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Signed ΔE vs analytic (all rows)")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "error_histograms.png", dpi=110)
    plt.close(fig)
    print(f"saved -> {args.out_dir / 'error_histograms.png'}")

    # Top-N worst rows for PINN vs NIST (handy for diagnosing)
    worst = df.nlargest(10, "ΔE_PINN_meV", keep="all")[
        ["Z", "level_config", "E_nist_hartree", "E_pinn_hartree",
         "E_fac_hartree", "ΔE_PINN_meV", "ΔE_FAC_meV"]
    ]
    worst_path = args.out_dir / "top10_worst_PINN.csv"
    worst.to_csv(worst_path, index=False)
    print(f"saved -> {worst_path}")


if __name__ == "__main__":
    main()