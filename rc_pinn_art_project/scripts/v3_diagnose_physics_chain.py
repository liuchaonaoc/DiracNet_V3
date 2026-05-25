#!/usr/bin/env python3
"""Single-sample physics chain diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.losses.pde_loss import dirac_pde_loss
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.hydrogenic import cosine_signed, hydrogenic_P_analytic
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_smoke.yaml")
    ap.add_argument("--row", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    manifest = ROOT / cfg.dataset.manifest
    if not manifest.exists():
        from scripts.v3_prepare_hydrogenic import main as prep

        prep()
    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max), int(cfg.grid.n_grid), str(cfg.grid.scheme)
    )
    key = jax.random.PRNGKey(0)
    model, params = build_model_and_params(cfg, grid, key)
    batch = collate_batches([ds[args.row]], n_csf_max=int(getattr(cfg.model, "n_csf_max", 8)))
    out = model.apply(params, batch, grid, train=False, return_ci=bool(getattr(cfg.ci, "enabled", False)))

    wf = out["wavefunctions"]
    pde = float(
        dirac_pde_loss(
            wf["P"], wf["Q"], wf["dPdr"], wf["dQdr"], out["V"],
            batch["kappa"], grid.r, grid, batch["orb_mask"], out["E_orb"],
        )
    )
    Z = int(jnp.asarray(batch["Z"][0]))
    P_ref = hydrogenic_P_analytic(grid.r, Z, 1, 0)
    cos = float(cosine_signed(wf["P"][0, 0], P_ref, grid))
    print(f"row={args.row} Z={Z} L_PDE={pde:.6e} |cos|={abs(cos):.4f} E_orb={float(out['E_orb'][0,0]):.6f}")
    if out.get("E_csf") is not None:
        print(f"E_csf[0]={float(out['E_csf'][0,0]):.6f}")


if __name__ == "__main__":
    main()
