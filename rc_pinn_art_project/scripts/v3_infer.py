#!/usr/bin/env python3
"""JIT inference with optional CI."""

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
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_smoke.yaml")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    manifest = ROOT / cfg.dataset.manifest
    if not manifest.exists():
        from scripts.v3_prepare_hydrogenic import main as prep

        prep()
    ds = ManifestDataset(manifest)
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max), int(cfg.grid.n_grid), str(cfg.grid.scheme)
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    batch = collate_batches([ds[0]], n_csf_max=int(cfg.model.n_csf_max))
    E_grid = jnp.logspace(-2, 3, 8)
    infer = jax.jit(lambda p, b: model.apply(p, b, grid, train=False, return_ci=True, E_grid=E_grid))
    out = infer(params, batch)
    print("E_orb", out["E_orb"][0, 0])
    if out["E_csf"] is not None:
        print("E_csf", out["E_csf"][0, :3])
    if out.get("cross_sections"):
        print("CE shape", out["cross_sections"]["CE"].shape)


if __name__ == "__main__":
    main()
