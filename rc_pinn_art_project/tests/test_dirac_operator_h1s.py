import jax.numpy as jnp

from pinn_art.physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from pinn_art.physics.hydrogenic import hydrogenic_energy, hydrogenic_P_analytic


def test_h1s_pde_small_with_coulomb_V(r_grid):
    Z = 1
    r = r_grid.r
    P = hydrogenic_P_analytic(r, Z, 1, 0)[None, None, :]
    Q = P * 1e-4
    dP = jnp.gradient(P, r, axis=-1)
    dQ = jnp.gradient(Q, r, axis=-1)
    V = (-Z / jnp.clip(r, 1e-8))[None, :]
    kappa = jnp.array([[-1]])
    LP, LQ = dirac_apply(P, Q, dP, dQ, V, kappa, r)
    E = orbital_energy_from_dirac(P, Q, LP, LQ, r_grid)
    res = LP - E[..., None] * P
    assert float(jnp.max(jnp.abs(res))) < 5.0
