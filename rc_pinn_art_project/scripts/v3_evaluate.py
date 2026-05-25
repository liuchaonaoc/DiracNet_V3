#!/usr/bin/env python3
"""Evaluate model and write metrics.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from pinn_art.constants import HARTREE_TO_EV
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_smoke.yaml")
    ap.add_argument("--run", default="results/latest")
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    manifest = ROOT / cfg.dataset.manifest
    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max), int(cfg.grid.n_grid), str(cfg.grid.scheme)
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))

    mae_list = []
    n_nist = 0
    n_fallback = 0
    for i in range(min(10, len(ds))):
        batch = collate_batches([ds[i]], n_csf_max=int(cfg.model.n_csf_max))
        out = model.apply(params, batch, grid, train=False, return_ci=True)
        mask = batch["nist_mask"][:, 0]
        if bool(mask[0]):
            n_nist += 1
            if out["E_csf"] is not None:
                err = abs(float(out["E_csf"][0, 0] - batch["E_nist"][0, 0])) * HARTREE_TO_EV
                mae_list.append(err)
        else:
            n_fallback += 1

    out_dir = ROOT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "layer2": {
            "mae_nist_meV": float(sum(mae_list) / max(len(mae_list), 1)),
            "n_nist_matched": n_nist,
            "n_theory_fallback": n_fallback,
        }
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
