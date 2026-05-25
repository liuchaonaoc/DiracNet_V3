import jax
import jax.numpy as jnp

from pinn_art.ci.nist_inject import fill_h_diagonal_hybrid


def test_fallback_uses_theory_when_mask_false():
    H = jnp.array([[[1.0, 0.2], [0.2, 2.0]]])
    E_nist = jnp.array([[0.5, 0.0]])
    mask = jnp.array([[True, False]])
    H_out, src = fill_h_diagonal_hybrid(H, E_nist, mask)
    assert float(H_out[0, 1, 1]) == 2.0
    assert float(H_out[0, 0, 0]) == 0.5
    assert int(src[0, 1]) == 0


def test_nist_stops_gradient_on_masked_diag():
    H_th = jnp.array([[[0.0, 0.1], [0.1, 0.0]]])

    def loss_fn(e_nist):
        H_out, _ = fill_h_diagonal_hybrid(H_th, e_nist, jnp.array([[True, False]]))
        return H_out[0, 0, 0]

    g = jax.grad(loss_fn)(jnp.array([[0.5, 0.0]]))
    assert float(g[0, 0]) == 0.0
