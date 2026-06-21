#!/usr/bin/env python3
"""Debug: confirm the Laguerre init patches actually reach params.

Three checks:
1. `params` tree shape — is `laguerre_init` in `params['params']` or
   somewhere else (Flax versions differ)?
2. After forward pass with a 1s hydrogenic row, does the P output match
   `hydrogenic_P_jax` up to float32 roundoff?  If not, the bias init
   didn't reach the heads.
3. `jax.grad` on the Laguerre sub-loss — are gradients non-zero on the
   Laguerre head parameters?  If 0, the heads are disconnected from loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.hydrogenic import (
    hydrogenic_P_jax,
    hydrogenic_laguerre_coeffs,
)
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def main():
    cfg = load_config(ROOT / "configs/v3_stage_a_laguerre_basis.yaml")
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max),
        int(cfg.grid.n_grid), str(cfg.grid.scheme),
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))

    print("=" * 70)
    print("(1) Param tree walk — find Laguerre init slots")
    print("=" * 70)
    print("Top-level keys:", list(params.keys()) if isinstance(params, dict) else type(params))
    if isinstance(params, dict):
        for k, v in params.items():
            if isinstance(v, dict):
                print(f"  params['{k}'] keys: {list(v.keys())}")
                if k == "params" and "DeepONetDirac_0" in v:
                    deep = v["DeepONetDirac_0"]
                    print(f"    DeepONetDirac_0 keys: {list(deep.keys())}")
                    for sub_k, sub_v in deep.items():
                        if "laguerre" in sub_k or "Laguerre" in sub_k:
                            print(f"      >>> '{sub_k}': {sub_v}")
                if k == "params" and "laguerre_init" in v:
                    print(f"    >>> top-level laguerre_init: {v['laguerre_init']}")
    else:
        print("(flax.core.FrozenDict; deep-print:)")

    # Look anywhere in the tree for 'laguerre_init'
    flat = jax.tree_util.tree_leaves_with_path(params)
    found_lag = False
    for path, leaf in flat:
        path_str = "/".join(str(p.key) for p in path)
        if "laguerre" in path_str.lower():
            print(f"  FOUND {path_str}: shape={leaf.shape}, dtype={leaf.dtype}, "
                  f"first={float(leaf.flatten()[0]):.4f}")
            found_lag = True
    if not found_lag:
        print("  *** NO 'laguerre_init' leaf in param tree at all ***")

    # Now: how many leaves, where are DeepONetDirac params?
    print()
    print("=" * 70)
    print("(2) Build a real batch and check forward P")
    print("=" * 70)
    manifest = ROOT / "data_cache/manifest_hydrogenic_v3.parquet"
    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    print(f"  ds rows: {len(ds)}")
    # First manifest row → 1s hydrogenic (Z=1, n=1, l=0)
    batch = collate_batches([ds[0]], n_csf_max=int(cfg.model.n_csf_max))
    print(f"  batch Z: {np.asarray(batch['Z'])}  kappa[0,0]={int(batch['kappa'][0,0])}  "
          f"n[0,0]={int(batch['shell_table'][0,0,0])}  l[0,0]={int(batch['shell_table'][0,0,1])}")

    out = model.apply(params, batch, grid, train=False, return_ci=False)
    P = np.asarray(out["wavefunctions"]["P"][0])   # [N_orb, N_g]
    lag_coeffs = out.get("laguerre_coeffs")
    lag_lambdas = out.get("laguerre_lambdas")
    if lag_coeffs is not None:
        lc = np.asarray(lag_coeffs[0])
        lc = lc if lc.ndim > 0 else lc.reshape(1, 1)
    else:
        lc = None
    if lag_lambdas is not None:
        ll = np.asarray(lag_lambdas[0])
        ll = ll if ll.ndim > 0 else ll.reshape(1)
    else:
        ll = None
    print(f"  P[0,0] max={np.max(np.abs(P[0])):.6f}   (hydrogenic 1s peak ≈ 0.7357)")
    print(f"  P[0,1..5] max={np.max(np.abs(P[1:6])):.3e}  (expected ≈ 0)")
    if lc is not None:
        print(f"  lag_coeffs[0, 0:5] = {lc[0, :5]}")
        print(f"  lag_coeffs[0, -3:] = {lc[0, -3:]}")
    if ll is not None:
        print(f"  lag_lambdas[0] = {ll[0]}   (analytic Z_eff/n = 1.0 for H 1s)")

    # Compare with analytic
    Z0 = int(batch["Z"][0])
    n0 = int(batch["shell_table"][0, 0, 0])
    l0 = int(batch["shell_table"][0, 0, 1])
    coeff_ref = hydrogenic_laguerre_coeffs(Z0, n0, l0, K_max=int(cfg.model.K_max))
    print(f"  Analytic coeff: c_k[0:5] = {np.asarray(coeff_ref[:5])}")
    if lc is not None:
        diff = lc[0] - np.asarray(coeff_ref)
        scale = float(np.max(np.abs(coeff_ref)) + 1e-6)
        print(f"  Max |coeff_model - coeff_analytic| / scale: "
              f"{float(np.max(np.abs(diff)) / scale):.3e}")

    # Cosine to analytic P
    l_arr = jnp.where(batch["kappa"][0] < 0, -batch["kappa"][0] - 1, batch["kappa"][0])
    n_arr = jnp.maximum(jnp.abs(batch["kappa"][0]), 1)
    P_ref = hydrogenic_P_jax(grid.r, batch["Z"], n_arr[None, :], l_arr[None, :])[0, 0]
    err = float(jnp.max(jnp.abs(P[0] - P_ref)))
    err_signed = float(jnp.max(jnp.abs(P[0] + P_ref)))
    scale = float(jnp.max(jnp.abs(P_ref)) + 1e-6)
    print(f"  |P_model - P_ref|_max / scale = {err/scale:.3e}  (sign-flipped: {err_signed/scale:.3e})")

    print()
    print("=" * 70)
    print("(3) Gradient sanity check on Laguerre heads")
    print("=" * 70)

    def lag_loss(p):
        out_ = model.apply(p, batch, grid, train=True, return_ci=False)
        # ‖P_model‖^2 is a proxy that should drive the heads' biases.
        return jnp.sum(out_["wavefunctions"]["P"] ** 2)

    g = jax.grad(lag_loss)(params)
    leaves = jax.tree_util.tree_leaves_with_path(g)
    lag_grad_norms = []
    found_const_grad = False
    for path, leaf in leaves:
        path_str = "/".join(str(pp.key) for pp in path)
        if "Laguerre" in path_str or "laguerre" in path_str:
            n = float(jnp.max(jnp.abs(leaf)))
            if "init_const" in path_str:
                found_const_grad = True
            lag_grad_norms.append((path_str, n))
    print(f"  Total Laguerre-related grad entries: {len(lag_grad_norms)}")
    for s, n in lag_grad_norms[:10]:
        print(f"    {s}: {n:.3e}")
    if not lag_grad_norms:
        print("  *** NO Laguerre gradients at all — heads are disconnected ***")
    if not found_const_grad:
        print("  (init_const slots are constants — they should have NO grad; this is correct.)")


if __name__ == "__main__":
    main()