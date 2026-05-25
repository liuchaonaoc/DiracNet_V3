import jax.numpy as jnp

from pinn_art.physics.orthogonalizer import lowdin_orthonormalize


def test_ortho_normalizes_orbitals(r_grid):
    B, N, Ng = 1, 2, r_grid.n_grid
    P = jnp.ones((B, N, Ng))
    Q = jnp.ones((B, N, Ng)) * 0.1
    mask = jnp.array([[True, True]])
    out = lowdin_orthonormalize(P, Q, r_grid, mask)
    n0 = float(r_grid.integrate(out["P"][:, 0] ** 2 + out["Q"][:, 0] ** 2, axis=-1)[0])
    n1 = float(r_grid.integrate(out["P"][:, 1] ** 2 + out["Q"][:, 1] ** 2, axis=-1)[0])
    assert abs(n0 - 1.0) < 1e-4
    assert abs(n1 - 1.0) < 1e-4


def test_ortho_reduces_overlap(r_grid, rng_key):
    import jax

    B, Ng = 1, r_grid.n_grid
    t = jnp.linspace(0.1, 1.0, Ng)
    k1, k2 = jax.random.split(rng_key)
    P1 = jnp.sin(t * 1.3)[None, None, :]
    P2 = jnp.cos(t * 0.7 + 0.1)[None, None, :]
    P = jnp.concatenate([P1, P2], axis=1)
    Q = P * 0.05
    mask = jnp.array([[True, True]])
    before = float(
        r_grid.integrate(
            P[:, 0] * P[:, 1] + Q[:, 0] * Q[:, 1],
            axis=-1,
        )[0]
    )
    out = lowdin_orthonormalize(P, Q, r_grid, mask, use_full_lowdin=True)
    after = float(
        r_grid.integrate(
            out["P"][:, 0] * out["P"][:, 1] + out["Q"][:, 0] * out["Q"][:, 1],
            axis=-1,
        )[0]
    )
    assert abs(after) < abs(before)
    assert abs(after) < 1e-3
