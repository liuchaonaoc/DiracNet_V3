import jax.numpy as jnp

from pinn_art.ci.nist_inject import fill_h_diagonal_hybrid


def test_fallback_uses_theory_when_mask_false():
    H = jnp.array([[[1.0, 0.2], [0.2, 2.0]]])
    E_nist = jnp.array([[0.5, jnp.nan]])
    mask = jnp.array([[True, False]])
    H_out, src = fill_h_diagonal_hybrid(H, E_nist, mask)
    assert float(H_out[0, 1, 1]) == 2.0
    assert float(H_out[0, 0, 0]) == 0.5
    assert int(src[0, 1]) == 0


def test_nist_when_mask_true():
    H = jnp.array([[[1.0, 0.0], [0.0, 2.0]]])
    E_nist = jnp.array([[-0.5]])
    mask = jnp.array([[True, False]])
    H_out, _ = fill_h_diagonal_hybrid(H, E_nist, mask)
    assert float(H_out[0, 0, 0]) == -0.5
