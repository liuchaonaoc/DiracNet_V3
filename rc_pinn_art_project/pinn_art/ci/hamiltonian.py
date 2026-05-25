"""Assemble CI Hamiltonian with optional Racah + Slater corrections."""

from __future__ import annotations

import jax.numpy as jnp


def assemble_hamiltonian_diagonal(
    E_orb: jnp.ndarray,
    csf_mask: jnp.ndarray,
    csf_to_orb: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Build H [B, M, M] with diagonal from orbital energies mapped to CSFs.
    If csf_to_orb is None, use first M_orb energies padded.
    """
    B, M = csf_mask.shape
    if csf_to_orb is None:
        M_orb = E_orb.shape[1]
        diag = jnp.zeros((B, M), dtype=E_orb.dtype)
        take = min(M, M_orb)
        diag = diag.at[:, :take].set(E_orb[:, :take])
    else:
        diag = jnp.sum(csf_to_orb * E_orb[:, None, :], axis=-1)
    H = jnp.zeros((B, M, M), dtype=E_orb.dtype)
    idx = jnp.arange(M)
    H = H.at[:, idx, idx].set(diag)
    eye = jnp.eye(M, dtype=H.dtype)
    H = jnp.where(csf_mask[:, :, None] & csf_mask[:, None, :], H, eye[None, :, :] * 0.0)
    return H


def assemble_hamiltonian(
    E_orb: jnp.ndarray,
    Rk: jnp.ndarray,
    C_ang: jnp.ndarray,
    csf_mask: jnp.ndarray,
    csf_to_orb: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Assemble H = H_one-body + sum_k C^k * R^k.

    E_orb:     [B, N_orb]
    Rk:        [B, n_k, N_orb]  (diagonal Slater integrals per orbital)
    C_ang:     [B, n_k, M, M]
    csf_mask:  [B, M]
    """
    H = assemble_hamiltonian_diagonal(E_orb, csf_mask, csf_to_orb)
    B, M = csf_mask.shape
    n_k = Rk.shape[1]
    N_orb = Rk.shape[2]

    for k_idx in range(n_k):
        Ck = C_ang[:, k_idx, :M, :M]
        Rk_k = Rk[:, k_idx, : min(N_orb, M)]
        for i in range(min(M, N_orb)):
            for j in range(min(M, N_orb)):
                if i == j:
                    continue
                contrib = Ck[:, i, j] * Rk_k[:, i] * 0.5 + Ck[:, i, j] * Rk_k[:, j] * 0.5
                H = H.at[:, i, j].add(contrib * csf_mask[:, i] * csf_mask[:, j])

    H = 0.5 * (H + jnp.swapaxes(H, -1, -2))
    return H
