"""Cross-orbital orthogonality penalty."""

from __future__ import annotations

import jax.numpy as jnp

from ..utils.grid import RadialGrid


def orthonormality_loss(P, Q, grid: RadialGrid, orb_mask, eps: float = 1e-12):
    integrand = P[:, :, None, :] * P[:, None, :, :] + Q[:, :, None, :] * Q[:, None, :, :]
    S = grid.integrate(integrand, axis=-1)
    B, N, _ = S.shape
    # off-diagonal only
    mask_2d = orb_mask[:, :, None] * orb_mask[:, None, :]
    eye = jnp.eye(N, dtype=S.dtype)[None, :, :]
    off = (1.0 - eye) * mask_2d
    return jnp.sum((S * off) ** 2) / jnp.clip(jnp.sum(off), 1.0)
