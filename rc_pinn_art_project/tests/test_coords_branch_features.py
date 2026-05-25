import jax.numpy as jnp

from pinn_art.coords.shell_features import branch_input_dim, build_branch_features


def test_branch_input_dim():
    assert branch_input_dim(16) == 32 + 16 + 48 + 4


def test_branch_features_shape(h1s_batch):
    feat = build_branch_features(h1s_batch, n_orb_max=16)
    assert feat.shape == (1, branch_input_dim(16))
    assert jnp.isfinite(feat).all()
