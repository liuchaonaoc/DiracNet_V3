import jax.numpy as jnp

from pinn_art.ci.slater_radial import compute_slater_R0
from pinn_art.physics.hydrogenic import hydrogenic_P_analytic


def test_slater_R0_positive(r_grid):
    r = r_grid.r
    P = hydrogenic_P_analytic(r, 1, 1, 0)
    Q = P * 1e-4
    R0 = float(compute_slater_R0(P[None, :], P[None, :], Q[None, :], Q[None, :], r_grid, k=0)[0])
    assert R0 > 0.0


def test_slater_R0_k2_finite(r_grid):
    r = r_grid.r
    P = hydrogenic_P_analytic(r, 1, 2, 0)
    Q = P * 1e-4
    R2 = compute_slater_R0(P[None, :], P[None, :], Q[None, :], Q[None, :], r_grid, k=2)[0]
    assert bool(jnp.isfinite(R2))
