"""Branch network input features from atomic batch."""

from __future__ import annotations

import jax.numpy as jnp


def branch_input_dim(n_orb_max: int) -> int:
    # embed_z(32) + omega(n_orb) + n,l,kappa (3*n_orb) + 4 scalars
    return 32 + n_orb_max + 3 * n_orb_max + 4


def _embed_z(Z: jnp.ndarray, dim: int = 32) -> jnp.ndarray:
    """Sinusoidal embedding of atomic number."""
    Zf = Z.astype(jnp.float32)
    freqs = jnp.arange(1, dim // 2 + 1, dtype=jnp.float32)
    angles = Zf[:, None] / (100.0 ** (freqs[None, :] / (dim // 2)))
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)[:, :dim]


def build_branch_features(
    batch: dict,
    *,
    n_orb_max: int,
) -> jnp.ndarray:
    """Build [B, D_in] branch features."""
    Z = batch["Z"].astype(jnp.float32)
    B = Z.shape[0]
    orb_mask = batch["orb_mask"][:, :n_orb_max].astype(jnp.float32)
    omega = batch["omega"][:, :n_orb_max] * orb_mask

    shells = batch.get("shell_table")
    if shells is not None:
        n_idx = shells[:, :n_orb_max, 0].astype(jnp.float32) / 30.0
        l_idx = shells[:, :n_orb_max, 1].astype(jnp.float32) / 7.0
    else:
        n_idx = jnp.zeros((B, n_orb_max), dtype=jnp.float32)
        l_idx = jnp.zeros((B, n_orb_max), dtype=jnp.float32)

    kappa = batch["kappa"][:, :n_orb_max].astype(jnp.float32)
    kappa_n = kappa / 4.0

    nele = batch.get("nele", jnp.ones(B, dtype=jnp.int32)).astype(jnp.float32)
    ion = batch.get("ion_charge", jnp.zeros(B, dtype=jnp.int32)).astype(jnp.float32)
    omega_sum = jnp.sum(omega, axis=-1, keepdims=True)
    n_max = jnp.max(n_idx * orb_mask + (1.0 - orb_mask) * (-1.0), axis=-1, keepdims=True)
    n_max = jnp.clip(n_max, 0.0, 1.0)

    scalars = jnp.concatenate(
        [nele[:, None] / 30.0, ion[:, None] / 30.0, omega_sum / 30.0, n_max],
        axis=-1,
    )
    z_emb = _embed_z(Z)
    return jnp.concatenate([z_emb, omega, n_idx, l_idx, kappa_n, scalars], axis=-1)
