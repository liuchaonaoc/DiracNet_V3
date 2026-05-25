"""Stage A training step (JIT-compiled)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from ..losses.asymptotic_loss import asymptotic_tail_loss
from ..losses.loss_schedule import stage_a_weights
from ..losses.norm_loss import normalization_loss
from ..losses.ortho_loss import orthonormality_loss
from ..losses.pde_loss import dirac_pde_loss
from ..losses.potential_prior import potential_prior_loss, potential_smooth_loss
from ..utils.grid import RadialGrid


def compute_stage_a_loss(out: dict, batch: dict, grid: RadialGrid, weights: dict) -> tuple[jnp.ndarray, dict]:
    wf = out["wavefunctions"]
    P, Q = wf["P"], wf["Q"]
    dPdr, dQdr = wf["dPdr"], wf["dQdr"]
    V = out["V"]
    kappa = batch["kappa"]
    orb_mask = batch["orb_mask"]
    r = grid.r
    Z = batch["Z"]

    shell_table = batch.get("shell_table")
    n_principal = shell_table[:, : kappa.shape[1], 0] if shell_table is not None else None

    l_pde = dirac_pde_loss(P, Q, dPdr, dQdr, V, kappa, r, grid, orb_mask, E_orb=out["E_orb"])
    l_ortho = orthonormality_loss(P, Q, grid, orb_mask)
    l_asym = asymptotic_tail_loss(P, Q, r, orb_mask, Z=Z, n_principal=n_principal)
    l_norm = normalization_loss(P, Q, grid, orb_mask)
    l_vp = potential_prior_loss(V, Z, r)
    l_vs = potential_smooth_loss(V, r)

    total = (
        weights["pde"] * l_pde
        + weights["ortho"] * l_ortho
        + weights["asym"] * l_asym
        + weights["norm"] * l_norm
        + weights["v_prior"] * l_vp
        + weights["v_smooth"] * l_vs
    )
    metrics = {
        "loss": total,
        "pde": l_pde,
        "ortho": l_ortho,
        "asym": l_asym,
        "norm": l_norm,
        "v_prior": l_vp,
        "v_smooth": l_vs,
    }
    return total, metrics


@partial(jax.jit, static_argnames=("weights_key",))
def _train_step_jit(state, batch, grid: RadialGrid, weights_vals, weights_key):
    weights = dict(zip(weights_key, weights_vals))

    def loss_fn(params):
        out = state.apply_fn(params, batch, grid, train=True, return_ci=False)
        loss, metrics = compute_stage_a_loss(out, batch, grid, weights)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.tree.map(lambda g: jnp.nan_to_num(g), grads)
    new_state = state.apply_gradients(grads=grads)
    metrics = {**metrics, "loss": loss}
    return new_state, metrics


def train_step(state, batch, grid: RadialGrid, weights_dict):
    """JIT-wrapped training step.

    `weights_dict` is split into hashable keys (static) + jax-array values
    so we get a single compile across epochs even if the dict object identity
    changes.
    """
    keys = tuple(sorted(weights_dict.keys()))
    vals = tuple(jnp.asarray(weights_dict[k], dtype=jnp.float32) for k in keys)
    return _train_step_jit(state, batch, grid, vals, keys)
