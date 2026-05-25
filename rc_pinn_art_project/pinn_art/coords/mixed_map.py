"""Mixed radial coordinate t(r) = c1*sqrt(r) + c2*log(r + eps)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class MixedRadialMap:
    c1: float = 1.0
    c2: float = 1.0
    r_eps: float = 1e-6

    def t(self, r: jnp.ndarray) -> jnp.ndarray:
        r_safe = jnp.maximum(r, self.r_eps)
        return self.c1 * jnp.sqrt(r_safe) + self.c2 * jnp.log(r_safe)

    def dt_dr(self, r: jnp.ndarray) -> jnp.ndarray:
        r_safe = jnp.maximum(r, self.r_eps)
        return self.c1 / (2.0 * jnp.sqrt(r_safe)) + self.c2 / r_safe
