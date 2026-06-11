"""R1.2 — SCF consistency loss."""

import jax
import jax.numpy as jnp
import numpy as np

from pinn_art.losses.scf_consistency import scf_consistency_loss


def test_loss_zero_at_fixed_point():
    V = jnp.linspace(-5.0, -0.1, 64)[None, :]
    assert float(scf_consistency_loss(V, V)) < 1e-12


def test_loss_positive_off_fixed_point():
    V_net = jnp.linspace(-5.0, -0.1, 64)[None, :]
    V_dfs = V_net + 0.3
    assert float(scf_consistency_loss(V_net, V_dfs)) > 0.0


def test_stop_gradient_blocks_density_side():
    V_net = jnp.full((1, 32), -1.0)
    V_dfs = jnp.full((1, 32), -2.0)

    # gradient w.r.t. V_dfs must be zero (stop_gradient on target)
    g = jax.grad(lambda vd: scf_consistency_loss(V_net, vd))(V_dfs)
    assert float(jnp.max(jnp.abs(g))) < 1e-12

    # but it flows to V_net
    g_net = jax.grad(lambda vn: scf_consistency_loss(vn, V_dfs))(V_net)
    assert float(jnp.max(jnp.abs(g_net))) > 0.0


def test_no_stop_gradient_couples_both():
    V_net = jnp.full((1, 32), -1.0)
    V_dfs = jnp.full((1, 32), -2.0)
    g = jax.grad(lambda vd: scf_consistency_loss(V_net, vd, stop_grad=False))(V_dfs)
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_weighted_mean_matches_manual():
    rng = np.random.default_rng(0)
    V_net = jnp.asarray(rng.normal(size=(2, 16)).astype(np.float32))
    V_dfs = jnp.asarray(rng.normal(size=(2, 16)).astype(np.float32))
    w = jnp.asarray(np.abs(rng.normal(size=(16,))).astype(np.float32))
    got = float(scf_consistency_loss(V_net, V_dfs, weight=w))
    diff2 = np.array((V_net - V_dfs) ** 2)
    wn = np.array(w)[None, :]
    manual = float(np.sum(wn * diff2) / np.sum(np.broadcast_to(wn, diff2.shape)))
    assert abs(got - manual) < 1e-5
