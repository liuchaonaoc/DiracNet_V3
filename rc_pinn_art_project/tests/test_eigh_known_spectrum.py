import jax.numpy as jnp

from pinn_art.ci.eigen_solver import safe_eigh


def test_eigh_recovers_diagonal():
    H = jnp.diag(jnp.array([-2.0, -1.0, -0.5]))
    E, V = safe_eigh(H[None, :, :])
    E = E[0]
    assert jnp.allclose(jnp.sort(E), jnp.array([-2.0, -1.0, -0.5]), atol=1e-5)
    assert float(jnp.max(jnp.abs(V[0].T @ V[0] - jnp.eye(3)))) < 1e-4


def test_eigh_2x2_offdiag():
    H = jnp.array([[[0.0, 0.1], [0.1, 0.0]]])
    E, _ = safe_eigh(H)
    assert jnp.allclose(jnp.sort(E[0]), jnp.array([-0.1, 0.1]), atol=1e-4)
