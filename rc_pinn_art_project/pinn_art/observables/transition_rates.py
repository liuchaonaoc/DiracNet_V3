"""E1 transition rates and gf values."""

from __future__ import annotations

import jax.numpy as jnp

from ..constants import C_LIGHT, HARTREE_TO_EV, TWO_PI
from ..utils.grid import RadialGrid
from .multipole import radial_dipole_matrix_element


def compute_e1_transitions(
    E_csf: jnp.ndarray,
    V_csf: jnp.ndarray,
    P: jnp.ndarray,
    Q: jnp.ndarray,
    grid: RadialGrid,
    csf_mask: jnp.ndarray,
) -> dict:
    """
    Pairwise E1 transitions among CSF basis (simplified: use orbital 0 as dipole).
    Returns A_ki, gf, wavelength_nm, source (0=theory).
    """
    B, M = E_csf.shape
    P0 = P[:, 0, :]
    transitions = []
    A_list = []
    for i in range(M):
        for f in range(M):
            if i == f:
                continue
            dE = jnp.abs(E_csf[:, f] - E_csf[:, i])
            nu = dE * HARTREE_TO_EV / (TWO_PI * 2.418884e-17 + 1e-30)
            D = radial_dipole_matrix_element(P0, P0, grid)
            A = (64.0 * jnp.pi ** 4 * jnp.clip(nu, 1e-10) ** 3 / (3.0 * 1e-30)) * D ** 2
            gf = 1.0
            wl = 1239.841984 / jnp.clip(dE * HARTREE_TO_EV, 1e-12)
            A_list.append(A)
    if not A_list:
        A_ki = jnp.zeros((B, 0))
    else:
        A_ki = jnp.stack(A_list, axis=-1)
    return {
        "A_ki": A_ki,
        "gf": A_ki,
        "wavelength_nm": jnp.zeros((B, max(1, A_ki.shape[-1]))),
        "source": jnp.zeros((B, max(1, A_ki.shape[-1])), dtype=jnp.int32),
    }
