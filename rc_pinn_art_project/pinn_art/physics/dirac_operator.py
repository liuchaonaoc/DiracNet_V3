"""Radial Dirac operator H_D in atomic units."""

from __future__ import annotations

import jax.numpy as jnp

from ..constants import C_LIGHT
from ..utils.grid import RadialGrid


def dirac_apply(
    P: jnp.ndarray,
    Q: jnp.ndarray,
    dPdr: jnp.ndarray,
    dQdr: jnp.ndarray,
    V: jnp.ndarray,
    kappa: jnp.ndarray,
    r: jnp.ndarray,
    c: float = C_LIGHT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply H_D to (P,Q). Shapes: P,Q,dP,dQ [B,N_orb,N_g], V [B,N_g], kappa [B,N_orb]."""
    inv_r = 1.0 / r[None, None, :]
    kappa_over_r = kappa[..., None].astype(P.dtype) * inv_r
    V_b = V[:, None, :]
    LP = V_b * P + c * (-dQdr + kappa_over_r * Q)
    LQ = (V_b - 2.0 * c * c) * Q + c * (dPdr + kappa_over_r * P)
    return LP, LQ


def orbital_energy_from_dirac(
    P: jnp.ndarray,
    Q: jnp.ndarray,
    LP: jnp.ndarray,
    LQ: jnp.ndarray,
    grid: RadialGrid,
) -> jnp.ndarray:
    num = grid.integrate(P * LP + Q * LQ, axis=-1)
    den = grid.integrate(P * P + Q * Q, axis=-1)
    return num / jnp.clip(den, 1e-12)
