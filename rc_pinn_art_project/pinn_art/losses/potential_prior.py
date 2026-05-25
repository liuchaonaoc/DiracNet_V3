"""Potential regularization."""

from __future__ import annotations

import jax.numpy as jnp


def potential_prior_loss(V, Z, r):
    V_nuc = -Z[:, None].astype(jnp.float32) / jnp.clip(r[None, :], 1e-8)
    delta = V - V_nuc
    return jnp.mean(delta ** 2)


def potential_smooth_loss(V, r):
    d2V = jnp.diff(V, n=2, axis=-1)
    return jnp.mean(d2V ** 2)
