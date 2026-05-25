"""Central potential V(r) = -Z/r + tanh(correction)."""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from .siren import SirenMLP


class PotentialHead(nn.Module):
    hidden: int = 64
    n_layers: int = 3
    omega_0: float = 30.0

    @nn.compact
    def __call__(self, t_grid: jnp.ndarray, Z: jnp.ndarray) -> jnp.ndarray:
        """t_grid [N_g], Z [B] -> V_corr [B, N_g]."""
        N_g = t_grid.shape[0]
        t_feat = jnp.broadcast_to(t_grid[None, :], (Z.shape[0], N_g))
        t_in = t_feat[..., None]
        corr = SirenMLP(features=1, hidden=self.hidden, n_layers=self.n_layers, omega_0=self.omega_0)(t_in)[..., 0]
        corr = nn.tanh(nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(corr[..., None]))[..., 0]
        r = jnp.exp(t_grid)
        r = jnp.linspace(1e-4, 50.0, N_g)
        return corr


def nuclear_plus_correction(
    r: jnp.ndarray,
    Z: jnp.ndarray,
    V_corr: jnp.ndarray,
) -> jnp.ndarray:
    V_nuc = -Z[:, None].astype(jnp.float32) / jnp.clip(r[None, :], 1e-8)
    return V_nuc + jnp.tanh(V_corr)
