import jax
import jax.numpy as jnp

from pinn_art.utils.grid import make_radial_grid

from pinn_art.coords.shell_features import build_branch_features
from pinn_art.coords.mixed_map import MixedRadialMap
from pinn_art.nets.deeponet import DeepONetDirac
from pinn_art.utils.grid import make_radial_grid


def test_deeponet_forward_jit(h1s_batch, rng_key):
    grid = make_radial_grid(n_grid=64, r_max=20.0)
    m = MixedRadialMap()
    t = m.t(grid.r)
    dt = m.dt_dr(grid.r)
    feat = build_branch_features(h1s_batch, n_orb_max=16)
    net = DeepONetDirac(d_in_branch=feat.shape[-1], d_branch=32, d_trunk=32, n_siren_layers=2, omega_0=10.0)
    params = net.init(
        rng_key, feat, t, grid.r, dt,
        h1s_batch["kappa"], h1s_batch["orb_mask"], h1s_batch["Z"],
    )

    @jax.jit
    def fwd(p):
        return net.apply(
            p, feat, t, grid.r, dt,
            h1s_batch["kappa"], h1s_batch["orb_mask"], h1s_batch["Z"],
        )

    out = fwd(params)
    assert jnp.isfinite(out["P"]).all()


def test_deeponet_forward_shapes(h1s_batch, rng_key):
    grid = make_radial_grid(n_grid=64, r_max=20.0)
    m = MixedRadialMap()
    t = m.t(grid.r)
    dt = m.dt_dr(grid.r)
    feat = build_branch_features(h1s_batch, n_orb_max=16)
    net = DeepONetDirac(d_in_branch=feat.shape[-1], d_branch=32, d_trunk=32, n_siren_layers=2, omega_0=10.0)
    params = net.init(
        rng_key, feat, t, grid.r, dt,
        h1s_batch["kappa"], h1s_batch["orb_mask"], h1s_batch["Z"],
    )
    out = net.apply(
        params, feat, t, grid.r, dt,
        h1s_batch["kappa"], h1s_batch["orb_mask"], h1s_batch["Z"],
    )
    assert out["V"].shape == (1, 64)
    assert out["P"].shape == (1, 16, 64)
    assert jnp.isfinite(out["P"]).all()
