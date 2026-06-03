"""Stage B — CI / Slater calibration losses."""

from __future__ import annotations

import jax.numpy as jnp

from ..constants import hartree_to_meV


def slater_R0_consistency_loss(
    Rk: jnp.ndarray,
    orb_mask: jnp.ndarray,
    *,
    target: float = 1.0,
) -> jnp.ndarray:
    """R^0(a,a) should match normalization (≈1 for unit-norm orbitals)."""
    R0 = Rk[:, 0, :]
    err = (R0 - target) ** 2
    w = orb_mask.astype(R0.dtype)
    return jnp.sum(err * w) / (jnp.sum(w) + 1e-12)


def hamiltonian_offdiag_loss(H: jnp.ndarray, csf_mask: jnp.ndarray) -> jnp.ndarray:
    """Penalize off-diagonal CI matrix elements (weak for multi-CSF later)."""
    B, M, _ = H.shape
    idx = jnp.arange(M)
    off = H.at[:, idx, idx].set(0.0)
    mask2 = csf_mask[:, :, None] & csf_mask[:, None, :]
    eye = jnp.eye(M, dtype=bool)[None, :, :]
    mask2 = mask2 & ~eye
    num = jnp.sum((off**2) * mask2)
    den = jnp.sum(mask2) + 1e-12
    return num / den


def e_csf_orbital_consistency_loss(
    E_csf: jnp.ndarray,
    E_orb: jnp.ndarray,
    csf_mask: jnp.ndarray,
    orb_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Leading CSF energy should track the active orbital energy (single-ref)."""
    E_lead = E_csf[:, 0]
    E_o = E_orb[:, 0]
    active = csf_mask[:, 0] & orb_mask[:, 0]
    err = (E_lead - E_o) ** 2
    return jnp.sum(err * active.astype(err.dtype)) / (jnp.sum(active) + 1e-12)


def synthetic_spectrum_loss(
    E_pred: jnp.ndarray,
    E_ref: jnp.ndarray,
) -> jnp.ndarray:
    """MSE on sorted eigenvalues (Hartree)."""
    return jnp.mean((jnp.sort(E_pred) - jnp.sort(E_ref)) ** 2)


def spectrum_error_meV(E_pred: jnp.ndarray, E_ref: jnp.ndarray) -> jnp.ndarray:
    """Max |ΔE| in meV for sorted spectra."""
    dE = hartree_to_meV(jnp.abs(jnp.sort(E_pred) - jnp.sort(E_ref)))
    return jnp.max(dE)
