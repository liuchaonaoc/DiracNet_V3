"""Numeric helpers."""

from __future__ import annotations

import jax.numpy as jnp


def masked_mean(x: jnp.ndarray, mask: jnp.ndarray, axis=None, eps: float = 1e-12) -> jnp.ndarray:
    m = mask.astype(x.dtype)
    num = jnp.sum(x * m, axis=axis)
    den = jnp.sum(m, axis=axis).clip(min=eps)
    return num / den


def sanitize_array(x: jnp.ndarray, lo: float = -1e8, hi: float = 1e8) -> jnp.ndarray:
    x = jnp.where(jnp.isfinite(x), x, 0.0)
    return jnp.clip(x, lo, hi)
