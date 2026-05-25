"""Hydrogenic analytic references."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from scipy.special import genlaguerre

from ..utils.grid import RadialGrid


def hydrogenic_energy(Z: float | int, n: float | int) -> float:
    return -float(Z) ** 2 / (2.0 * float(n) ** 2)


def hydrogenic_P_analytic(r: jnp.ndarray, Z: float | int, n: int, l: int = 0) -> jnp.ndarray:
    if n <= l:
        raise ValueError("require n > l")
    rho = 2.0 * float(Z) * np.asarray(r) / float(n)
    fact = math.factorial(n - l - 1)
    fact_n = math.factorial(n + l)
    norm = math.sqrt((2.0 * float(Z) / float(n)) ** 3 * fact / (2.0 * n * fact_n))
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    R = norm * (rho ** l) * np.exp(-rho / 2.0) * L
    P = (rho / (2.0 * float(Z) / float(n))) * R
    return jnp.asarray(P, dtype=r.dtype if hasattr(r, "dtype") else jnp.float32)


def cosine_signed(P_model: jnp.ndarray, P_ref: jnp.ndarray, grid: RadialGrid) -> jnp.ndarray:
    num = grid.integrate(P_model * P_ref, axis=-1)
    den = jnp.sqrt(
        grid.integrate(P_model * P_model, axis=-1) * grid.integrate(P_ref * P_ref, axis=-1)
    )
    return num / jnp.clip(den, 1e-12)
