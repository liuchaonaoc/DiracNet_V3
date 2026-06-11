"""Stage A training step (JIT-compiled)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from ..losses.asymptotic_loss import asymptotic_tail_loss
from ..losses.norm_loss import normalization_loss
from ..losses.ortho_loss import orthonormality_loss
from ..losses.pde_loss import dirac_pde_loss
from ..losses.potential_prior import potential_prior_loss, potential_smooth_loss
from ..losses.scf_consistency import scf_consistency_loss
from ..physics.dfs_potential import (
    build_dfs_potential,
    electron_density,
    zeff_anchor_potential,
)
from ..utils.grid import RadialGrid

# (enabled, alpha_x, latter_tail, anchor_vprior, fermi_amaldi, scf_weight_mode,
#  anchor_vprior_zeff)
# bare-hydrogenic Round-1 default
_NO_DFS = (False, 1.0, True, False, True, "density", False)


def _scf_radial_weight(mode: str, rho, r):
    """Radial weighting for the SCF consistency loss (P1).

    The old ``r**2`` weight de-emphasized the valence/core region and let
    ``V_net`` drift exactly where the binding energy is set (see
    EXCITATION_VS_NIST.md §A.2). ``density`` weights by the electron density
    rho(r) = sum_a omega_a (P_a^2+Q_a^2): it constrains V precisely where the
    electrons live (which sets E_orb) and vanishes at the r->0 cusp and the far
    tail, so neither dominates.
    """
    if mode == "uniform":
        return None
    if mode == "r":
        return r
    if mode == "r2":
        return r * r
    # default: density-weighted, with a small uniform floor so V stays
    # constrained even where the current density is ~0.
    rho_sg = jax.lax.stop_gradient(rho)
    floor = 1e-3 * jnp.max(rho_sg, axis=-1, keepdims=True)
    return rho_sg + floor


def compute_stage_a_loss(
    out: dict,
    batch: dict,
    grid: RadialGrid,
    weights: dict,
    dfs_cfg: tuple = _NO_DFS,
) -> tuple[jnp.ndarray, dict]:
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

    dfs_enabled, alpha_x, latter_tail, anchor_vprior, fermi_amaldi, scf_weight_mode, anchor_vprior_zeff = dfs_cfg
    V_dfs = None
    V_zeff_anchor = None
    scf_weight = None
    if dfs_enabled:
        omega = batch["omega"][:, : kappa.shape[1]]
        nele = batch["nele"]
        V_dfs = build_dfs_potential(
            P, Q, omega, Z, nele, grid,
            alpha_x=alpha_x, latter_tail=latter_tail, fermi_amaldi=fermi_amaldi,
        )
        rho = electron_density(P, Q, omega)
        scf_weight = _scf_radial_weight(scf_weight_mode, rho, r)
        if anchor_vprior_zeff and n_principal is not None:
            V_zeff_anchor = zeff_anchor_potential(
                Z, n_principal, omega, orb_mask, grid,
            )

    l_pde = dirac_pde_loss(P, Q, dPdr, dQdr, V, kappa, r, grid, orb_mask, E_orb=out["E_orb"])
    l_ortho = orthonormality_loss(P, Q, grid, orb_mask)
    l_asym = asymptotic_tail_loss(P, Q, r, orb_mask, Z=Z, n_principal=n_principal)
    l_norm = normalization_loss(P, Q, grid, orb_mask)
    # 路径 B: V_anchor 优先级 zeff > dfs > bare
    V_anchor_for_loss = V_zeff_anchor if V_zeff_anchor is not None else (V_dfs if anchor_vprior else None)
    l_vp = potential_prior_loss(V, Z, r, V_anchor=V_anchor_for_loss)
    l_vs = potential_smooth_loss(V, r)
    # P1: density-weighted SCF (was r^2, which let V_net drift in the valence
    # region where the binding energy is set — see EXCITATION_VS_NIST.md §A.2).
    l_scf = (
        scf_consistency_loss(V, V_dfs, weight=scf_weight)
        if V_dfs is not None
        else jnp.asarray(0.0, dtype=V.dtype)
    )

    total = (
        weights["pde"] * l_pde
        + weights["ortho"] * l_ortho
        + weights["asym"] * l_asym
        + weights["norm"] * l_norm
        + weights["v_prior"] * l_vp
        + weights["v_smooth"] * l_vs
        + weights.get("scf", 0.0) * l_scf
    )
    metrics = {
        "loss": total,
        "pde": l_pde,
        "ortho": l_ortho,
        "asym": l_asym,
        "norm": l_norm,
        "v_prior": l_vp,
        "v_smooth": l_vs,
        "scf": l_scf,
    }
    return total, metrics


@partial(jax.jit, static_argnames=("weights_key", "dfs_cfg"))
def _train_step_jit(state, batch, grid: RadialGrid, weights_vals, weights_key, dfs_cfg):
    weights = dict(zip(weights_key, weights_vals))

    def loss_fn(params):
        out = state.apply_fn(params, batch, grid, train=True, return_ci=False)
        loss, metrics = compute_stage_a_loss(out, batch, grid, weights, dfs_cfg)
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.tree.map(lambda g: jnp.nan_to_num(g), grads)
    new_state = state.apply_gradients(grads=grads)
    metrics = {**metrics, "loss": loss}
    return new_state, metrics


def train_step(state, batch, grid: RadialGrid, weights_dict, dfs_cfg: tuple = _NO_DFS):
    """JIT-wrapped training step.

    `weights_dict` is split into hashable keys (static) + jax-array values
    so we get a single compile across epochs even if the dict object identity
    changes. `dfs_cfg` is a hashable tuple (enabled, alpha_x, latter_tail,
    anchor_vprior, fermi_amaldi, scf_weight_mode) passed as a static argument.
    """
    keys = tuple(sorted(weights_dict.keys()))
    vals = tuple(jnp.asarray(weights_dict[k], dtype=jnp.float32) for k in keys)
    return _train_step_jit(state, batch, grid, vals, keys, tuple(dfs_cfg))
