"""Per-orbital normalization loss."""

from __future__ import annotations

import jax.numpy as jnp

from ..utils.grid import RadialGrid
from ..utils.numeric import masked_mean


def normalization_loss(P, Q, grid: RadialGrid, orb_mask):
    norm = grid.integrate(P * P + Q * Q, axis=-1)
    return masked_mean((norm - 1.0) ** 2, orb_mask)
