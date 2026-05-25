"""Phase-amplitude head for continuum (optional, disabled by default)."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class PhaseAmplitudeHead(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, trunk_hidden: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        A = nn.softplus(nn.Dense(1)(trunk_hidden))[..., 0]
        phi = nn.Dense(1)(trunk_hidden)[..., 0]
        return A, phi
