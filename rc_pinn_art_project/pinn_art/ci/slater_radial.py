"""Slater radial integrals R^k."""

from __future__ import annotations

import jax.numpy as jnp

from ..utils.grid import RadialGrid


def compute_slater_R0(P_a, P_b, Q_a, Q_b, grid: RadialGrid, k: int = 0) -> jnp.ndarray:
    """R^0-like overlap density integral between orbitals a,b."""
    rho = P_a * P_b + Q_a * Q_b
    if k == 0:
        return grid.integrate(rho, axis=-1)
    r = grid.r
    weight = jnp.power(jnp.clip(r, 1e-8), float(k))
    return grid.integrate(rho * weight[None, :], axis=-1)


def compute_all_Rk_diagonal(P, Q, orb_mask, grid: RadialGrid, k_list=(0,)):
    """Diagonal-only Slater blocks [B, n_k, N_orb] for Phase-1 CI."""
    B, N, _ = P.shape
    out = []
    for k in k_list:
        rk = []
        for a in range(N):
            val = compute_slater_R0(P[:, a], P[:, a], Q[:, a], Q[:, a], grid, k=k)
            rk.append(val)
        out.append(jnp.stack(rk, axis=1))
    return jnp.stack(out, axis=1)
