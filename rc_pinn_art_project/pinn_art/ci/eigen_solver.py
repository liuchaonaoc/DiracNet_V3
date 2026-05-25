"""Differentiable symmetric eigh with degeneracy protection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..constants import HARTREE_TO_EV


def regularize_H(H: jnp.ndarray, eps_degen_ev: float = 1e-6) -> jnp.ndarray:
    eps_ha = eps_degen_ev / HARTREE_TO_EV
    M = H.shape[-1]
    eye = jnp.eye(M, dtype=H.dtype)
    H_sym = 0.5 * (H + jnp.swapaxes(H, -1, -2))
    H_sym = jnp.where(jnp.isfinite(H_sym), H_sym, 0.0)
    return H_sym + eps_ha * eye


def _eigh_forward(H: jnp.ndarray, eps_degen_ev: float):
    return jnp.linalg.eigh(regularize_H(H, eps_degen_ev))


@jax.custom_vjp
def safe_eigh(H: jnp.ndarray, eps_degen_ev: float = 1e-6):
    return _eigh_forward(H, eps_degen_ev)


def safe_eigh_fwd(H, eps_degen_ev):
    E, V = _eigh_forward(H, eps_degen_ev)
    return (E, V), (H, E, V, eps_degen_ev)


def safe_eigh_bwd(res, g):
    H, E, V, eps_degen_ev = res
    gE, gV = g
    gap_ha = 1e-4 / HARTREE_TO_EV
    M = H.shape[-1]
    gH = jnp.einsum("...i,ij->...ij", gE, jnp.eye(M, dtype=H.dtype))
    Ei = E[..., :, None]
    Ej = E[..., None, :]
    diff = Ei - Ej
    safe_diff = jnp.where(jnp.abs(diff) < gap_ha, jnp.sign(diff + 1e-20) * gap_ha, diff)
    F = gV / safe_diff
    F = jnp.where(jnp.eye(M, dtype=bool), 0.0, F)
    gH = gH + jnp.einsum("...ij,...kj->...ik", F, V)
    gH = 0.5 * (gH + jnp.swapaxes(gH, -1, -2))
    return (gH, None)


safe_eigh.defvjp(safe_eigh_fwd, safe_eigh_bwd)
