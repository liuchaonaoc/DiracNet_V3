"""Potential regularization."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def potential_prior_loss(V, Z, r, V_anchor=None):
    """Pull the network potential toward an anchor.

    Default anchor is the bare nuclear ``-Z/r`` (Round-1 behavior). When
    ``V_anchor`` is provided (R1.4: the DFS screened potential ``V_dfs``), it is
    used instead and held fixed via ``stop_gradient`` so the density side does
    not chase the potential head within a step.
    """
    if V_anchor is None:
        anchor = -Z[:, None].astype(jnp.float32) / jnp.clip(r[None, :], 1e-8)
    else:
        anchor = jax.lax.stop_gradient(V_anchor)
    delta = V - anchor
    return jnp.mean(delta ** 2)


def potential_smooth_loss(V, r):
    d2V = jnp.diff(V, n=2, axis=-1)
    return jnp.mean(d2V ** 2)
