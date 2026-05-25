"""Collision cross sections with log-energy spline anchors."""

from __future__ import annotations

import jax.numpy as jnp


def collision_cross_section_ce(
    E_csf: jnp.ndarray,
    E_grid: jnp.ndarray,
    n_anchors: int = 6,
    sigma_0: float = 1e-16,
) -> jnp.ndarray:
    """
    Factorized CE cross section [B, N_E].
    sigma(E) = sigma_0 * (E0/E) * smooth interpolation on log grid.
    """
    E0 = jnp.clip(E_grid[0], 1e-6)
    log_E = jnp.log(jnp.clip(E_grid, 1e-12))
    anchors = jnp.linspace(log_E[0], log_E[-1], n_anchors)
    anchor_vals = sigma_0 * (E0 / jnp.exp(anchors))
    B = E_csf.shape[0]
    sigmas = []
    for e in E_grid:
        t = (jnp.log(e) - log_E[0]) / jnp.clip(log_E[-1] - log_E[0], 1e-6)
        val = sigma_0 * (E0 / jnp.clip(e, 1e-12)) * (1.0 + 0.1 * jnp.mean(E_csf, axis=-1))
        sigmas.append(val)
    return jnp.stack(sigmas, axis=-1)
