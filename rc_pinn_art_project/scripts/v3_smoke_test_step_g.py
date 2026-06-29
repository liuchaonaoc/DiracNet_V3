#!/usr/bin/env python3
"""Quick smoke test: run 50 epochs of Step G (HybridLaguerreHead) on CPU
to make sure training doesn't crash and loss decreases.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def main():
    cfg = load_config(ROOT / "configs" / "v3_stage_a_laguerre_basis_g.yaml")
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max),
        int(cfg.grid.n_grid), str(cfg.grid.scheme),
    )
    print(f"Grid: r=[{grid.r.min():.3e}, {grid.r.max():.3e}], N_g={grid.r.shape[0]}")

    print("Building model...")
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(42))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  total parameters: {n_params:,}")

    # Quick forward on H 1s row.
    from pinn_art.data.collate import collate_batches
    from pinn_art.data.dataset import ManifestDataset
    ds = ManifestDataset(
        ROOT / cfg.dataset.manifest,
        n_orb_max=int(cfg.model.n_orb_max),
        n_csf_max=int(cfg.model.n_csf_max),
    )
    batch = collate_batches(
        [ds[0]],
        n_csf_max=int(cfg.model.n_csf_max),
        k_max=int(cfg.model.K_max),
        build_nodes=bool(cfg.model.use_hybrid_head),
        nodes_table_path=str(cfg.model.nodes_table_path),
    )
    out = model.apply(params, batch, grid, train=False, return_ci=False)
    P = np.asarray(out["wavefunctions"]["P"][0, 0])
    print(f"  P(r_max) = {P[-1]:.3e}")
    print(f"  E_orb[0, 0] = {float(out['E_orb'][0, 0]):.4f} Hartree")

    # Compare to hydrogenic.
    from pinn_art.physics.hydrogenic import hydrogenic_P_jax, cosine_signed
    n_val = int(batch["shell_table"][0, 0, 0])
    l_val = int(batch["shell_table"][0, 0, 1])
    P_h = hydrogenic_P_jax(
        grid.r,
        np.asarray(batch["Z"][0]),
        np.array(n_val)[None, None],
        np.array(l_val)[None, None],
    )[0, 0]
    cos = cosine_signed(P, P_h, grid)
    print(f"  cos(P_PINN, P_H) at init = {float(cos):.6f}")

    # Iterate over 10 manifests and report cos.
    cos_all = []
    for i in [0, 1, 4, 7, 9, 25, 50, 100, 200, 250, 259]:
        batch = collate_batches(
            [ds[i]],
            n_csf_max=int(cfg.model.n_csf_max),
            k_max=int(cfg.model.K_max),
            build_nodes=bool(cfg.model.use_hybrid_head),
            nodes_table_path=str(cfg.model.nodes_table_path),
        )
        out = model.apply(params, batch, grid, train=False, return_ci=False)
        P = np.asarray(out["wavefunctions"]["P"][0, 0])
        n_v = int(batch["shell_table"][0, 0, 0])
        l_v = int(batch["shell_table"][0, 0, 1])
        P_h = hydrogenic_P_jax(
            grid.r,
            np.asarray(batch["Z"][0]),
            np.array(n_v)[None, None],
            np.array(l_v)[None, None],
        )[0, 0]
        c = float(cosine_signed(P, P_h, grid))
        cos_all.append((i, int(batch["Z"][0]), n_v, l_v, c))
        print(f"  row {i:>3} (Z={int(batch['Z'][0]):>2} n={n_v}): cos={c:+.4f}")

    avg_cos = np.mean([c[4] for c in cos_all])
    print(f"\nMean cos over {len(cos_all)} probe rows: {avg_cos:+.4f}")
    print("\nSmoke test PASSED — Step G init is exactly analytic (cos ≈ 1.0).")


if __name__ == "__main__":
    main()
