import jax.numpy as jnp

from pinn_art.losses.asymptotic_loss import asymptotic_tail_loss


def test_asymptotic_finite(h1s_batch, r_grid):
    P = jnp.exp(-0.5 * r_grid.r)[None, None, :].repeat(16, axis=1)
    Q = P * 0.01
    loss = asymptotic_tail_loss(P, Q, r_grid.r, h1s_batch["orb_mask"], Z=h1s_batch["Z"])
    assert jnp.isfinite(loss)
    assert float(loss) >= 0.0
