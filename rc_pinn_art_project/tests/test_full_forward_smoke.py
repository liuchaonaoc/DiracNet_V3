import jax

from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def test_full_forward(h1s_batch, rng_key):
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "v3_smoke.yaml")
    grid = make_radial_grid(n_grid=64, r_max=20.0)
    model, params = build_model_and_params(cfg, grid, rng_key)
    out = model.apply(params, h1s_batch, grid, train=False, return_ci=True)
    assert out["E_orb"].shape == (1, 16)
    assert out["E_csf"] is not None
    assert jnp_isfinite(out["V"])


def jnp_isfinite(x):
    import jax.numpy as jnp

    return bool(jnp.isfinite(x).all())
