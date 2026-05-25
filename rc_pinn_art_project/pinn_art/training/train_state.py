"""Flax train state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import optax
from flax.training import train_state


class TrainState(train_state.TrainState):
    pass


def create_train_state(model, params, cfg, total_steps: int) -> TrainState:
    opt_cfg = getattr(cfg, "optimizer", cfg)
    lr = float(getattr(opt_cfg, "lr_trunk", 3e-4))
    wd = float(getattr(opt_cfg, "weight_decay", 1e-4))
    tx = optax.chain(
        optax.clip_by_global_norm(float(getattr(opt_cfg, "grad_clip", 1.0))),
        optax.adamw(lr, weight_decay=wd),
    )

    def apply_fn(params, batch, grid, **kw):
        return model.apply(params, batch, grid, **kw)

    return TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
