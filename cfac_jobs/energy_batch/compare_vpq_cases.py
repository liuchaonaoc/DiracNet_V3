#!/usr/bin/env python3
"""Compare V(r), P(r), Q(r) for a list of (Z, n) cases between cFAC and
PINN-ART.  Each case is rendered as a 3-panel figure (V, P, Q) on a
log-r axis.  All cases are also stacked into one big grid figure.

Inputs:
    --ckpt PATH        Path to a Stage-A msgpack checkpoint
                       (default: checkpoints/v3_stage_a_laguerre_basis_c_c2/stage_a_last.msgpack)
    --config PATH      Path to a Stage-A config YAML
                       (default: configs/v3_stage_a_laguerre_basis_c.yaml)
    --cases Z1 N1 Z2 N2 Z3 N3 ...   list of (Z, n) pairs to plot
                       (default: the 14-case sweep:
                        H/Fe {1,2,5,8}  +  Li/C/O {1,5})
    --out-dir PATH     Directory for output PNGs
                       (default: energy_batch/vpq_compare)

Outputs:
    vpq_compare/VPQ_Z<Z>_n<N>.png     one per case
    vpq_compare/VPQ_grid.png          all cases stacked (N × 3 panels)

Example:
    python compare_vpq_cases.py
    python compare_vpq_cases.py --ckpt /path/to/new_ckpt.msgpack \\
        --cases 1 1 1 8 8 1 26 1 26 8
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

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.training.checkpoint import load_params


DEFAULT_CASES = [
    (1, 1), (1, 2), (1, 5), (1, 8),
    (3, 1), (3, 5),
    (6, 1), (6, 5),
    (8, 1), (8, 5),
    (26, 1), (26, 2), (26, 5), (26, 8),
]

ELEM = {1: "H", 2: "He", 3: "Li", 6: "C", 8: "O", 26: "Fe"}


def parse_fac(z: int, n: int) -> dict | None:
    p = JOBS / f"cf_Z{z}_n{n}_PQ.dat"
    if not p.exists():
        return None
    data = np.loadtxt(p, comments="#")
    r = data[:, 1]
    vc_r = data[:, 2]
    u_r = data[:, 3]
    P = data[:, 4]
    Q = data[:, 5]
    V = vc_r / np.where(r > 0, r, 1.0) + u_r / np.where(r > 0, r, 1.0)
    return {"r": r, "V": V, "P": P, "Q": Q}


def load_pinn(model, params, ds, grid, n_csf_max: int, z: int, n: int) -> dict | None:
    """Look up the manifest row for (Z, n) and apply the PINN."""
    target = None
    for i in range(len(ds)):
        sh = ds.df.iloc[i]
        if int(sh["Z"]) == z and sh["level_config"] == f"{n}s1":
            target = i
            break
    if target is None:
        return None
    batch = collate_batches([ds[target]], n_csf_max=n_csf_max)
    out = model.apply(params, batch, grid, train=False, return_ci=False)
    V = np.asarray(out["V"][0])  # [N_orb, N_g] or [N_g]
    P = np.asarray(out["wavefunctions"]["P"][0])  # [N_orb, N_g]
    Q = np.asarray(out["wavefunctions"]["Q"][0])
    sh = np.asarray(batch["shell_table"])[0]  # [N_orb, 4]
    n_arr = sh[:, 0].astype(int)
    l_arr = sh[:, 1].astype(int)
    slot = None
    for a in range(P.shape[0]):
        if n_arr[a] == n and l_arr[a] == 0:
            slot = a
            break
    if slot is None:
        slot = 0
    return {
        "r": np.asarray(grid.r),
        "V": np.asarray(V).reshape(-1),
        "P": np.asarray(P[slot]).reshape(-1),
        "Q": np.asarray(Q[slot]).reshape(-1),
    }


def plot_row(ax_row, fac, pinn, label: str):
    rmax = max(fac["r"].max(), pinn["r"].max())
    rmin = 1e-2
    for ax, key, ylab, linthresh in zip(
        ax_row,
        ["V", "P", "Q"],
        ["V(r) [Hartree]", "P(r)", "Q(r)"],
        [1e-2, 1e-3, 1e-4],
    ):
        ax.plot(fac["r"], fac[key], 'k-', lw=1.0, label="cFAC")
        ax.plot(pinn["r"], pinn[key], 'r--', lw=1.0, label="PINN-ART")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xlim(rmin, rmax)
        ax.set_xlabel("r [a₀]")
        ax.set_ylabel(ylab)
        ax.set_title(f"{label}  {key}(r)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=PROJ / "checkpoints/v3_stage_a_laguerre_basis_c_c2/stage_a_last.msgpack")
    ap.add_argument("--config", type=Path,
                    default=PROJ / "configs/v3_stage_a_laguerre_basis_c.yaml")
    ap.add_argument("--cases", type=int, nargs="+",
                    help="Flat list: Z1 N1 Z2 N2 ...; default = 14-case sweep")
    ap.add_argument("--out-dir", type=Path,
                    default=JOBS / "vpq_compare")
    args = ap.parse_args()

    cases = DEFAULT_CASES
    if args.cases is not None:
        assert len(args.cases) % 2 == 0, "--cases must be Z1 N1 Z2 N2 ..."
        cases = list(zip(args.cases[::2], args.cases[1::2]))

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max),
        int(cfg.grid.n_grid), str(cfg.grid.scheme),
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    params = load_params(args.ckpt)
    ds = ManifestDataset(PROJ / cfg.dataset.manifest,
                         n_orb_max=int(cfg.model.n_orb_max),
                         n_csf_max=int(cfg.model.n_csf_max))

    n_ok = 0
    n_skip_fac = 0
    n_skip_pinn = 0
    for z, n in cases:
        elem = ELEM.get(z, f"Z={z}")
        label = f"{elem} {n}s"
        fac = parse_fac(z, n)
        if fac is None:
            print(f"[skip] no cFAC file for Z={z}, n={n}: run gen_cfac_batch.sh first")
            n_skip_fac += 1
            continue
        pinn = load_pinn(model, params, ds, grid, int(cfg.model.n_csf_max), z, n)
        if pinn is None:
            print(f"[skip] no manifest row for Z={z}, n={n}")
            n_skip_pinn += 1
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        plot_row(axes, fac, pinn, label)
        fig.tight_layout()
        out = out_dir / f"VPQ_Z{z}_n{n}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"saved -> {out}")
        n_ok += 1

    # Big grid: N rows × 3 cols
    fig, axes = plt.subplots(n_ok, 3, figsize=(15, 3.6 * n_ok))
    if n_ok == 1:
        axes = axes[None, :]
    row = 0
    for z, n in cases:
        elem = ELEM.get(z, f"Z={z}")
        label = f"{elem} {n}s"
        fac = parse_fac(z, n)
        pinn = load_pinn(model, params, ds, grid, int(cfg.model.n_csf_max), z, n)
        if fac is None or pinn is None:
            continue
        plot_row(axes[row], fac, pinn, label)
        row += 1
    fig.tight_layout()
    out = out_dir / "VPQ_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved -> {out}")

    print(f"[summary] ok={n_ok}  skip_fac={n_skip_fac}  skip_pinn={n_skip_pinn}")


if __name__ == "__main__":
    main()