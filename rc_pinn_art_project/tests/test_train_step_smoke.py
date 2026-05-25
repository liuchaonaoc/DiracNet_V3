import jax

from pinn_art.losses.loss_schedule import stage_a_weights
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.stage_a_trainer import train_step
from pinn_art.training.train_state import create_train_state
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def test_train_step(h1s_batch, rng_key):
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "v3_smoke.yaml")
    grid = make_radial_grid(n_grid=64, r_max=20.0)
    model, params = build_model_and_params(cfg, grid, rng_key)
    state = create_train_state(model, params, cfg, 10)
    w = stage_a_weights(cfg)
    state2, m1 = train_step(state, h1s_batch, grid, w)
    state3, m2 = train_step(state2, h1s_batch, grid, w)
    assert float(m1["loss"]) < 1e6
    assert float(m2["loss"]) < float(m1["loss"]) * 2.0 + 1.0
