import jax
import jax.numpy as jnp

from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def test_infer_jit_shapes(h1s_batch, rng_key):
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "v3_smoke.yaml")
    grid = make_radial_grid(n_grid=64, r_max=20.0)
    model, params = build_model_and_params(cfg, grid, rng_key)
    E_grid = jnp.logspace(-1, 2, 4)

    @jax.jit
    def infer(p, b):
        return model.apply(p, b, grid, train=False, return_ci=True, E_grid=E_grid)

    out = infer(params, h1s_batch)
    assert out["V"].shape == (1, 64)
    assert out["wavefunctions"]["P"].shape[0] == 1
    assert out["cross_sections"]["CE"].shape == (1, 4)
