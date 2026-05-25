"""SIREN layers for trunk network."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn


class SirenDense(nn.Module):
    features: int
    omega_0: float = 30.0
    is_first: bool = False

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        if self.is_first:
            bound = 1.0 / fan_in
        else:
            bound = math.sqrt(6.0 / fan_in) / self.omega_0

        def kernel_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, minval=-bound, maxval=bound, dtype=dtype)

        x = nn.Dense(self.features, kernel_init=kernel_init)(x)
        return jnp.sin(self.omega_0 * x)


class SirenMLP(nn.Module):
    features: int
    hidden: int
    n_layers: int
    omega_0: float = 30.0

    @nn.compact
    def __call__(self, x):
        h = SirenDense(self.hidden, omega_0=self.omega_0, is_first=True)(x)
        for _ in range(self.n_layers - 2):
            h = SirenDense(self.hidden, omega_0=self.omega_0, is_first=False)(h)
        return nn.Dense(self.features, kernel_init=nn.initializers.zeros)(h)
