import jax
import jax.numpy as jnp

from pinn_art.ci.eigen_solver import safe_eigh


def test_eigh_gradient_finite():
    key = jax.random.PRNGKey(0)

    def loss(H):
        E, _ = safe_eigh(H)
        return jnp.sum(E)

    for _ in range(5):
        key, k = jax.random.split(key)
        H = jax.random.normal(k, (4, 4))
        H = 0.5 * (H + H.T)
        g = jax.grad(loss)(H)
        assert jnp.isfinite(g).all()
