import jax.numpy as jnp
import numpy as np

from pinn_art.coords.mixed_map import MixedRadialMap


def test_t_monotone():
    m = MixedRadialMap()
    r = jnp.linspace(1e-4, 10.0, 50)
    t = m.t(r)
    assert jnp.all(jnp.diff(t) > 0)


def test_dt_dr_positive():
    m = MixedRadialMap()
    r = jnp.linspace(1e-3, 10.0, 50)
    assert jnp.all(m.dt_dr(r) > 0)


def test_dt_dr_vs_fd_interior():
    """Analytic dt/dr vs FD on interior points (avoid r->0 singularity)."""
    m = MixedRadialMap()
    r = np.linspace(0.2, 5.0, 200)
    t = np.asarray(m.t(jnp.asarray(r)))
    dt_num = np.gradient(t, r)
    dt = np.asarray(m.dt_dr(jnp.asarray(r)))
    rel = np.abs(dt - dt_num) / np.clip(np.abs(dt), 1e-6, None)
    assert np.median(rel) < 0.02
