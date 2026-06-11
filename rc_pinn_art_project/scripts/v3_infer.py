#!/usr/bin/env python3
"""JIT inference with optional CI (Stage C)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from pinn_art.ci.racah_cache import RacahCache
from pinn_art.constants import hartree_to_meV
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params, merge_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_phase1_stage_c_z1_8.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--row", type=int, default=0, help="Manifest row index")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    manifest = ROOT / cfg.dataset.manifest
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest missing: {manifest}\n"
            "Run: PYTHONPATH=. python scripts/v3_prepare_nist_manifest.py"
        )

    ckpt_path = args.ckpt or getattr(getattr(cfg, "stage_c", None), "ckpt", None)
    if not ckpt_path:
        raise ValueError("Pass --ckpt or set stage_c.ckpt in config")

    racah_cache = RacahCache.load(ROOT / cfg.ci.racah_cache) if cfg.ci.enabled else None
    k_list = tuple(int(k) for k in cfg.ci.k_list)

    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max), int(cfg.grid.n_grid), str(cfg.grid.scheme)
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    params = merge_params(params, load_params(ROOT / ckpt_path))

    row = max(0, min(args.row, len(ds) - 1))
    batch = collate_batches(
        [ds[row]],
        n_csf_max=int(cfg.model.n_csf_max),
        racah_cache=racah_cache,
        k_list=k_list,
    )
    E_grid = jnp.logspace(-2, 3, 8)
    infer = jax.jit(
        lambda p, b: model.apply(p, b, grid, train=False, return_ci=True, E_grid=E_grid)
    )
    out = infer(params, batch)

    r = ds.df.iloc[row]
    print(f"row={row} Z={r['Z']} {r.get('element','')} level={r['level_config']}")
    print(f"  nist_inject={cfg.ci.nist_inject}  has_nist={r.get('has_nist_level', False)}")
    print(f"  E_orb_meV={float(hartree_to_meV(out['E_orb'][0, 0])):.4f}")
    if out.get("E_csf") is not None:
        print(f"  E_csf_meV={float(hartree_to_meV(out['E_csf'][0, 0])):.4f}")
        if out.get("diag_source") is not None:
            print(f"  diag_source={int(out['diag_source'][0, 0])} (1=NIST, 0=theory)")
    if out.get("cross_sections"):
        print("  CE shape", out["cross_sections"]["CE"].shape)


if __name__ == "__main__":
    main()
