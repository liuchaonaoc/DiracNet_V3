"""Multipole radial matrix elements."""

from __future__ import annotations

import jax.numpy as jnp

from ..utils.grid import RadialGrid


def radial_dipole_matrix_element(P_i, P_f, grid: RadialGrid) -> jnp.ndarray:
    return grid.integrate(P_i * grid.r[None, :] * P_f, axis=-1)
