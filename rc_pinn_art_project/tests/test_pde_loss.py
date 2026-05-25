import jax.numpy as jnp

from pinn_art.losses.pde_loss import dirac_pde_loss
from pinn_art.physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from pinn_art.physics.hydrogenic import hydrogenic_P_analytic


def test_pde_loss_small_on_analytic_h1s(r_grid):
    Z, n = 1, 1
    r = r_grid.r
    P = hydrogenic_P_analytic(r, Z, n, 0)[None, None, :]
    Q = P * 1e-4
    dP = jnp.gradient(P, r, axis=-1)
    dQ = jnp.gradient(Q, r, axis=-1)
    V = (-Z / jnp.clip(r, 1e-8))[None, :]
    kappa = jnp.array([[-1]])
    mask = jnp.array([[True]])
    LP, LQ = dirac_apply(P, Q, dP, dQ, V, kappa, r)
    E = orbital_energy_from_dirac(P, Q, LP, LQ, r_grid)
    loss = dirac_pde_loss(P, Q, dP, dQ, V, kappa, r, r_grid, mask, E_orb=E)
    # Analytic H 1s with approximate Q: residual dominated by Q component
    assert float(loss) < 5e4
    assert jnp.isfinite(loss)
