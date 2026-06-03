"""Stage B training — freeze DeepONet, train slater_log_scale (+ optional tiny PDE anchor)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from ..losses.asymptotic_loss import asymptotic_tail_loss
from ..losses.ci_loss import (
    e_csf_orbital_consistency_loss,
    hamiltonian_offdiag_loss,
    slater_R0_consistency_loss,
)
from ..losses.pde_loss import dirac_pde_loss
from ..utils.grid import RadialGrid


def _is_trainable_path(path) -> bool:
    joined = "/".join(str(p) for p in path)
    return "slater_log_scale" in joined


def slater_only_mask(params):
    return jax.tree_util.tree_map_with_path(
        lambda path, _: _is_trainable_path(path),
        params,
    )


def compute_stage_b_loss(out: dict, batch: dict, grid: RadialGrid, weights: dict) -> tuple[jnp.ndarray, dict]:
    wf = out["wavefunctions"]
    P, Q = wf["P"], wf["Q"]
    dPdr, dQdr = wf["dPdr"], wf["dQdr"]
    V = out["V"]
    kappa = batch["kappa"]
    orb_mask = batch["orb_mask"]
    csf_mask = batch["csf_mask"]
    r = grid.r
    Z = batch["Z"]

    l_slat = slater_R0_consistency_loss(out["Rk"], orb_mask)
    l_off = hamiltonian_offdiag_loss(out["H"], csf_mask)
    l_ecsf = e_csf_orbital_consistency_loss(out["E_csf"], out["E_orb"], csf_mask, orb_mask)
    l_pde = dirac_pde_loss(P, Q, dPdr, dQdr, V, kappa, r, grid, orb_mask, E_orb=out["E_orb"])

    total = (
        weights["slat"] * l_slat
        + weights["offdiag"] * l_off
        + weights["e_csf"] * l_ecsf
        + weights.get("pde", jnp.asarray(0.0)) * l_pde
    )
    metrics = {
        "loss": total,
        "slat": l_slat,
        "offdiag": l_off,
        "e_csf": l_ecsf,
        "pde": l_pde,
    }
    return total, metrics


@partial(jax.jit, static_argnames=("weights_key",))
def _train_step_jit(state, batch, grid: RadialGrid, weights_vals, weights_key):
    weights = dict(zip(weights_key, weights_vals))

    def loss_fn(params):
        out = state.apply_fn(params, batch, grid, train=True, return_ci=True)
        loss, metrics = compute_stage_b_loss(out, batch, grid, weights)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.tree.map(lambda g: jnp.nan_to_num(g), grads)
    mask = slater_only_mask(state.params)
    grads = jax.tree.map(lambda g, m: jnp.where(m, g, 0.0), grads, mask)
    new_state = state.apply_gradients(grads=grads)
    metrics = {**metrics, "loss": loss}
    return new_state, metrics


def train_step(state, batch, grid: RadialGrid, weights_dict):
    keys = tuple(sorted(weights_dict.keys()))
    vals = tuple(jnp.asarray(weights_dict[k], dtype=jnp.float32) for k in keys)
    return _train_step_jit(state, batch, grid, vals, keys)


def create_stage_b_train_state(model, params, cfg, total_steps: int = 10000):
    from .train_state import TrainState

    opt_cfg = getattr(cfg, "optimizer", cfg)
    stage_b_opt = getattr(cfg, "stage_b", None)
    lr = float(getattr(stage_b_opt, "lr", getattr(opt_cfg, "lr_trunk", 1e-3) * 0.5))
    wd = float(getattr(opt_cfg, "weight_decay", 1e-4))
    tx = optax.chain(
        optax.clip_by_global_norm(float(getattr(opt_cfg, "grad_clip", 1.0))),
        optax.adamw(lr, weight_decay=wd),
    )

    def apply_fn(params, batch, grid, **kw):
        return model.apply(params, batch, grid, **kw)

    return TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
