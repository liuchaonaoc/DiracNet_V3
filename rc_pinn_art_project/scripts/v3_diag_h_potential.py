#!/usr/bin/env python3
"""Diagnose WHY the trained Round-2 model gets hydrogen's energy scale wrong.

Loads the Round-2 checkpoint, forwards the H 1s / 2s configs and reports:
  - predicted E_orb and the 1s->2s excitation (vs NIST 10.2 eV),
  - the learned network potential V_net(r) vs the exact -1/r, split into the
    core (small r, where binding energy is set) and the tail (large r, where
    the r^2-weighted SCF loss pulls).

    cd DiracNet_V3/rc_pinn_art_project && export PYTHONPATH=.
    python scripts/v3_diag_h_potential.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid

HARTREE_EV = 27.211386245988
CKPT = "checkpoints/v3_stage_a_z1_26_n10/stage_a_last.msgpack"
CFG = "configs/v3_stage_a_z1_26_n10.yaml"


def forward_config(ds, model, params, grid, Z, ion, cfg):
    hits = ds.df[
        (ds.df["Z"] == Z) & (ds.df["ion_charge"] == ion) & (ds.df["level_config"] == cfg)
    ].index
    if len(hits) == 0:
        return None
    batch = collate_batches([ds[int(hits[0])]], n_csf_max=int(getattr(model, "n_csf_max", 8)))
    out = model.apply(params, batch, grid, train=False)
    return batch, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max),
        int(cfg.grid.n_grid), str(cfg.grid.scheme),
    )
    r = np.asarray(grid.r, dtype=np.float64)
    ds = ManifestDataset(
        ROOT / cfg.dataset.manifest,
        n_orb_max=int(cfg.model.n_orb_max),
        n_csf_max=int(getattr(cfg.model, "n_csf_max", 8)),
    )
    key = jax.random.PRNGKey(int(cfg.seed))
    model, params0 = build_model_and_params(cfg, grid, key)
    params = load_params(ROOT / args.ckpt)
    print(f"  ckpt: {args.ckpt}")

    print("\n=== Hydrogen energy-scale diagnosis (trained Round-2 ckpt) ===")
    e_levels = {}
    for lc in ("1s1", "2s1"):
        res = forward_config(ds, model, params, grid, 1, 0, lc)
        if res is None:
            print(f"  [{lc}] not found in manifest")
            continue
        batch, out = res
        E_orb = np.asarray(out["E_orb"][0], dtype=np.float64)
        mask = np.asarray(batch["orb_mask"][0], dtype=bool)
        # active orbital = last unmasked slot for this level_config
        active = np.where(mask)[0][-1]
        e_levels[lc] = float(E_orb[active])
        V = np.asarray(out["V"][0], dtype=np.float64)
        bare = -1.0 / r
        core = r < 2.0
        tail = r > 10.0
        dev_core = np.sqrt(np.mean((V[core] - bare[core]) ** 2))
        dev_tail = np.sqrt(np.mean((V[tail] - bare[tail]) ** 2))
        print(f"\n  [{lc}] active orb slot={active}  E_orb={e_levels[lc]:+.5f} Ha")
        print(f"       RMS |V_net - (-1/r)|:  core(r<2)={dev_core:.4f}   tail(r>10)={dev_tail:.4f}")
        for rr in (0.05, 0.2, 0.5, 1.0, 5.0, 20.0):
            i = int(np.argmin(np.abs(r - rr)))
            print(f"         r={r[i]:7.3f}  V_net={V[i]:+10.4f}  -1/r={bare[i]:+10.4f}  diff={V[i]-bare[i]:+8.4f}")

    if "1s1" in e_levels and "2s1" in e_levels:
        exc = (e_levels["2s1"] - e_levels["1s1"]) * HARTREE_EV
        print(f"\n  predicted 1s->2s excitation = {exc:.4f} eV   (NIST 10.199 eV)")
        print("  exact hydrogenic E: 1s=-0.5, 2s=-0.125 Ha -> 10.20 eV\n")


if __name__ == "__main__":
    main()
