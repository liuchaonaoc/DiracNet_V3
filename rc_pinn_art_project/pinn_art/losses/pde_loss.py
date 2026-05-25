"""Dirac PDE residual loss."""

from __future__ import annotations

import jax.numpy as jnp

from ..physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from ..utils.grid import RadialGrid
from ..utils.numeric import masked_mean


def dirac_pde_loss(
    P,
    Q,
    dPdr,
    dQdr,
    V,
    kappa,
    r,
    grid: RadialGrid,
    orb_mask,
    E_orb=None,
    e_char: float = 1.0,
):
    LP, LQ = dirac_apply(P, Q, dPdr, dQdr, V, kappa, r)
    if E_orb is None:
        E_orb = orbital_energy_from_dirac(P, Q, LP, LQ, grid)
    res_P = LP - E_orb[..., None] * P
    res_Q = LQ - E_orb[..., None] * Q
    integrand = res_P ** 2 + res_Q ** 2
    per_orb = grid.integrate(integrand, axis=-1) / (e_char ** 2 + 1e-12)
    return masked_mean(per_orb, orb_mask)
