import jax
import jax.numpy as jnp

from pinn_art.training.gate_a import check_gate_a


def test_gate_fail_on_random(h1s_batch, r_grid, rng_key):
    B, Ng = 1, r_grid.n_grid
    key, k1, k2 = jax.random.split(rng_key, 3)
    P = jax.random.normal(k1, (B, 16, Ng))
    Q = jax.random.normal(k2, (B, 16, Ng)) * 0.01
    out = {
        "V": -jnp.ones((B, Ng)),
        "wavefunctions": {
            "P": P, "Q": Q,
            "dPdr": jnp.zeros_like(P),
            "dQdr": jnp.zeros_like(Q),
        },
        "E_orb": jnp.array([[0.0] * 16]),
    }
    rep = check_gate_a(
        out, h1s_batch, r_grid,
        cos_threshold=0.99,
        pde_threshold=1e-6,
        e_orb_meV_threshold=1.0,
    )
    assert rep.verdict == "FAIL"
