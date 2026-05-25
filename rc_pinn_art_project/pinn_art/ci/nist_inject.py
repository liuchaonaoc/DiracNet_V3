"""NIST diagonal injection with theory fall-back."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def fill_h_diagonal_hybrid(
    H_theory: jnp.ndarray,
    E_nist: jnp.ndarray,
    nist_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    H_ii = where(nist_mask, stop_gradient(E_nist), H_ii^theory).
    Returns (H_out, diag_source) with diag_source 1=nist, 0=theory.
    """
    H_diag_theory = jnp.diagonal(H_theory, axis1=-2, axis2=-1)
    E_nist_const = jax.lax.stop_gradient(E_nist)
    H_ii = jnp.where(nist_mask, E_nist_const, H_diag_theory)
    M = H_theory.shape[-1]
    idx = jnp.arange(M)
    H_out = H_theory.at[:, idx, idx].set(H_ii)
    diag_source = nist_mask.astype(jnp.int32)
    return H_out, diag_source


def inject_nist_diagonal(H_theory, E_nist, nist_mask):
    H_out, _ = fill_h_diagonal_hybrid(H_theory, E_nist, nist_mask)
    return H_out
