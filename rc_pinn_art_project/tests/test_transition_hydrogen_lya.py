import jax.numpy as jnp

from pinn_art.observables.transition_rates import compute_e1_transitions
from pinn_art.utils.grid import make_radial_grid


def test_transition_output_keys():
    grid = make_radial_grid(n_grid=64, r_max=20.0)
    B, M, Ng = 1, 2, grid.n_grid
    E = jnp.array([[-1.0, -0.25]])
    V = jnp.eye(M)[None, :, :]
    P = jnp.exp(-grid.r)[None, None, :]
    Q = P * 0.01
    mask = jnp.array([[True, True]])
    out = compute_e1_transitions(E, V, P, Q, grid, mask)
    assert "A_ki" in out
    assert "gf" in out
    assert "source" in out
