import jax
import jax.numpy as jnp

from pinn_art.nets.siren import SirenMLP


def test_siren_forward_finite(rng_key):
    net = SirenMLP(features=4, hidden=16, n_layers=3, omega_0=15.0)
    x = jax.random.normal(rng_key, (2, 32, 5))
    params = net.init(rng_key, x)
    y = net.apply(params, x)
    assert y.shape == (2, 32, 4)
    assert jnp.isfinite(y).all()


def test_siren_zero_last_layer_init(rng_key):
    net = SirenMLP(features=1, hidden=8, n_layers=2, omega_0=10.0)
    x = jnp.zeros((1, 16, 3))
    params = net.init(rng_key, x)
    y = net.apply(params, x)
    assert float(jnp.max(jnp.abs(y))) < 1e-5
