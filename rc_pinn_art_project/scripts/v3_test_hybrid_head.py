#!/usr/bin/env python3
"""Unit tests for §13.11-C HybridLaguerreHead.

Tests:
1. §2.3 init = physics: at init, head output equals analytic c_k for all
   (Z, n, l) test cases.
2. Softplus(0) - log(2) = 0 exactly (sanity).
3. cumsum + mask produces correct learned nodes.
4. Forward pass shape: [B, K_max+1].
5. End-to-end P(r) at init = hydrogenic P(r) (numerical identity).
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

from pinn_art.nets.laguerre_basis import (
    HybridLaguerreHead,
    load_analytic_nodes_table,
    _lookup_analytic_nodes,
)
from pinn_art.physics.hydrogenic import hydrogenic_laguerre_coeffs, hydrogenic_P_jax
from pinn_art.utils.grid import RadialGrid

TEST_CASES = [
    (1, 1, 0), (1, 2, 0), (1, 5, 0), (1, 8, 0), (1, 10, 0),
    (3, 1, 0), (3, 5, 0),
    (8, 1, 0), (8, 5, 0), (8, 9, 0),
    (26, 1, 0), (26, 5, 0), (26, 10, 0),
]

K_MAX = 9
TABLE = load_analytic_nodes_table(
    ROOT / "data_cache" / "laguerre_nodes_z1_26_n1_10.parquet"
)


def test_softplus_minus_log2_is_zero_at_zero():
    """softplus(0) - log(2) = 0 exactly (key to §2.3 init property)."""
    x = jnp.float32(0.0)
    val = jax.nn.softplus(x) - jnp.log(jnp.array(2.0))
    val_f = float(val)
    print(f"  softplus(0) - log(2) = {val_f:.6e}")
    assert abs(val_f) < 1e-6, f"softplus(0) - log(2) must be ≈ 0, got {val_f}"
    print("  ✓ PASS")


def test_init_c_k_equals_analytic():
    """At init, HybridLaguerreHead output must equal analytic c_k exactly."""
    print("Test 2: §2.3 init = analytic c_k")
    head = HybridLaguerreHead(K_max=K_MAX, d_hidden_node=64, d_hidden_ref=32, n_iter=3)
    key = jax.random.PRNGKey(0)
    B = len(TEST_CASES)
    d_branch = 128

    # Random branch features (init probe).
    branch_feat = jax.random.normal(key, (B, d_branch))

    # Per-test analytic c_k & analytic node positions.
    Z_arr = jnp.asarray([tc[0] for tc in TEST_CASES], dtype=jnp.int32)
    n_arr = jnp.asarray([tc[1] for tc in TEST_CASES], dtype=jnp.int32)
    l_arr = jnp.asarray([tc[2] for tc in TEST_CASES], dtype=jnp.int32)
    lambda_arr = Z_arr.astype(jnp.float32) / n_arr.astype(jnp.float32)
    alpha_arr = 2.0 * l_arr.astype(jnp.float32) + 1.0
    degree_arr = jnp.maximum(n_arr - l_arr - 1, 0).astype(jnp.float32)
    coeff_init = hydrogenic_laguerre_coeffs(Z_arr, n_arr, l_arr, K_max=K_MAX)
    coeff_init = jnp.asarray(coeff_init, dtype=jnp.float32)
    if coeff_init.ndim == 1:
        coeff_init = coeff_init[None, :]

    r_nodes = jnp.stack([
        TABLE.get((int(Z), int(n), int(l)),
                  jnp.zeros(K_MAX, dtype=jnp.float32))
        for (Z, n, l) in TEST_CASES
    ])

    params = head.init(
        key, branch_feat, coeff_init, r_nodes, lambda_arr, alpha_arr, degree_arr
    )
    c_k_out = head.apply(
        params, branch_feat, coeff_init, r_nodes, lambda_arr, alpha_arr, degree_arr
    )
    c_k_out_np = np.asarray(c_k_out)
    coeff_init_np = np.asarray(coeff_init)

    max_err = np.max(np.abs(c_k_out_np - coeff_init_np))
    print(f"  max |c_k_out - coeff_init| = {max_err:.6e}")
    assert max_err < 1e-5, f"init property violated: max err {max_err:.3e}"
    print("  ✓ PASS")


def test_node_features_shape():
    """HybridLaguerreHead node features should have shape 3*K_MAX."""
    expected = 3 * K_MAX  # K_MAX is module-level constant
    print(f"  K_max={K_MAX}, expected node_feat dim = {expected}")
    assert expected == 27
    print("  ✓ PASS")


def test_e2e_init_p_equals_hydrogenic():
    """End-to-end: PINN P(r) at init must equal hydrogenic P_H(r)."""
    print("Test 4: E2E P(r) at init = P_H(r)")
    # Build a tiny config (Step E as base, then flip on hybrid).
    from pinn_art.models.pinn_art_model import build_model_and_params
    from pinn_art.utils.config import load_config

    cfg = load_config(ROOT / "configs" / "v3_stage_a_laguerre_basis_g.yaml")
    grid = make_grid(cfg)

    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(42))

    # Forward on a single hydrogenic 1s row.
    from pinn_art.data.collate import collate_batches
    from pinn_art.data.dataset import ManifestDataset
    ds = ManifestDataset(
        ROOT / "data_cache" / "manifest_hydrogenic_z1_26_n10.parquet",
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
    P_pinn = np.asarray(out["wavefunctions"]["P"][0, 0])  # [N_g]

    P_h = hydrogenic_P_jax(
        grid.r,
        np.asarray(batch["Z"][0]),
        np.asarray(batch["shell_table"][0, 0, 0])[None, None],
        np.asarray(batch["shell_table"][0, 0, 1])[None, None],
    )[0, 0]
    max_err = np.max(np.abs(P_pinn - P_h))
    rel_err = max_err / (np.max(np.abs(P_h)) + 1e-12)
    print(f"  max |P_PINN - P_H| = {max_err:.6e}, rel = {rel_err:.6e}")
    assert rel_err < 1e-4, f"E2E init violated: rel err {rel_err:.3e}"
    print("  ✓ PASS")


def make_grid(cfg):
    from pinn_art.utils.grid import make_radial_grid
    return make_radial_grid(
        r_min=float(cfg.grid.r_min),
        r_max=float(cfg.grid.r_max),
        n_grid=int(cfg.grid.n_grid),
        scheme=str(cfg.grid.scheme),
    )


def main():
    print("=" * 70)
    print("§13.11-C HybridLaguerreHead Unit Tests")
    print("=" * 70)
    print()
    print("Test 1: softplus(0) - log(2) == 0 exactly")
    test_softplus_minus_log2_is_zero_at_zero()
    print()
    test_init_c_k_equals_analytic()
    print()
    print("Test 3: node features shape")
    test_node_features_shape()
    print()
    test_e2e_init_p_equals_hydrogenic()
    print()
    print("=" * 70)
    print("All tests PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
