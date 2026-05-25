"""hydrogenic_P_jax matches scipy implementation for all (n, l, Z) used in Stage A."""
import jax.numpy as jnp
import numpy as np
import pytest

from pinn_art.physics.hydrogenic import (
    hydrogenic_P_analytic,
    hydrogenic_P_jax,
    hydrogenic_dP_dr_jax,
)
from pinn_art.utils.grid import make_radial_grid


@pytest.mark.parametrize(
    "Z,n,l",
    [
        (1, 1, 0),
        (1, 2, 0),
        (1, 2, 1),
        (1, 3, 0),
        (1, 3, 1),
        (1, 3, 2),
        (1, 4, 0),
        (2, 1, 0),
        (4, 1, 0),
        (4, 2, 0),
        (4, 2, 1),
        (8, 1, 0),
        (8, 2, 0),
    ],
)
def test_hydrogenic_P_jax_matches_scipy(Z, n, l):
    grid = make_radial_grid(r_min=1e-4, r_max=80.0, n_grid=512, scheme="loglinear")
    r = grid.r
    ref = hydrogenic_P_analytic(r, Z, n, l)
    Zb = jnp.array([Z], dtype=jnp.int32)
    nb = jnp.array([[n] + [0] * 3], dtype=jnp.int32)
    lb = jnp.array([[l] + [0] * 3], dtype=jnp.int32)
    P = hydrogenic_P_jax(r, Zb, nb, lb)
    got = P[0, 0]
    # absolute tolerance loose due to log/lgamma roundoff; check shape & relative
    err = float(jnp.max(jnp.abs(got - ref)))
    scale = float(jnp.max(jnp.abs(ref))) + 1e-6
    assert err / scale < 5e-3, f"P mismatch Z={Z} n={n} l={l}: rel_err={err/scale:.3e}"


def test_hydrogenic_P_jax_norm_unity():
    grid = make_radial_grid(r_min=1e-4, r_max=100.0, n_grid=1024, scheme="loglinear")
    r = grid.r
    Z = jnp.array([1], dtype=jnp.int32)
    # n=1..4, l=0
    nb = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    lb = jnp.array([[0, 0, 0, 0]], dtype=jnp.int32)
    P = hydrogenic_P_jax(r, Z, nb, lb)  # [1, 4, N_g]
    int_p2 = jnp.sum(P * P * grid.dr[None, None, :], axis=-1)[0]
    assert all(abs(float(x) - 1.0) < 5e-2 for x in int_p2), int_p2


def test_hydrogenic_P_jax_invalid_returns_zero():
    grid = make_radial_grid(r_min=1e-4, r_max=20.0, n_grid=64)
    r = grid.r
    Z = jnp.array([1], dtype=jnp.int32)
    # n=0 (masked-out orbital) and l>=n
    nb = jnp.array([[0, 1]], dtype=jnp.int32)
    lb = jnp.array([[0, 1]], dtype=jnp.int32)
    P = hydrogenic_P_jax(r, Z, nb, lb)
    assert jnp.all(P[0, 0] == 0)
    assert jnp.all(P[0, 1] == 0)


@pytest.mark.parametrize("Z,n,l", [(1, 1, 0), (1, 2, 0), (1, 2, 1), (4, 2, 0), (4, 2, 1)])
def test_hydrogenic_dP_dr_matches_finite_diff(Z, n, l):
    grid = make_radial_grid(r_min=1e-4, r_max=40.0, n_grid=2048, scheme="loglinear")
    r = grid.r
    Zb = jnp.array([Z], dtype=jnp.int32)
    nb = jnp.array([[n]], dtype=jnp.int32)
    lb = jnp.array([[l]], dtype=jnp.int32)
    P = hydrogenic_P_jax(r, Zb, nb, lb)[0, 0]
    dP_analytic = hydrogenic_dP_dr_jax(r, Zb, nb, lb)[0, 0]
    dP_fd = jnp.gradient(P, r)
    mask = (r > 0.05) & (r < 25.0)
    err = float(jnp.max(jnp.abs(dP_analytic[mask] - dP_fd[mask])))
    scale = float(jnp.max(jnp.abs(dP_fd[mask])) + 1e-6)
    assert err / scale < 0.05, f"dP/dr mismatch Z={Z} n={n} l={l}: rel_err={err/scale:.3e}"
