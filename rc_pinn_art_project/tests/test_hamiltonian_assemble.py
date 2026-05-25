import jax.numpy as jnp

from pinn_art.ci.hamiltonian import assemble_hamiltonian, assemble_hamiltonian_diagonal


def test_hamiltonian_diagonal_symmetric():
    E = jnp.array([[-0.5, -0.1, 0.0]])
    mask = jnp.array([[True, True, False]])
    H = assemble_hamiltonian_diagonal(E, mask)
    assert H.shape == (1, 3, 3)
    assert float(jnp.max(jnp.abs(H[0] - H[0].T))) < 1e-8
    assert abs(float(H[0, 0, 0]) + 0.5) < 1e-6


def test_hamiltonian_with_racah():
    E = jnp.array([[-0.5, -0.1]])
    Rk = jnp.array([[[0.8, 0.3]]])
    C = jnp.array([[[[1.0, 0.1], [0.1, 1.0]]]])
    mask = jnp.array([[True, True]])
    H = assemble_hamiltonian(E, Rk, C, mask)
    assert H.shape == (1, 2, 2)
    assert jnp.isfinite(H).all()
