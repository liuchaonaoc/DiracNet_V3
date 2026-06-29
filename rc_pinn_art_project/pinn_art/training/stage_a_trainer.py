"""Stage A training step (JIT-compiled)."""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
import optax

from ..losses.asymptotic_loss import asymptotic_tail_loss
from ..losses.coeff_loss import (
    coeff_anchor_loss,
    coeff_decay_loss,
    lambda_prior_loss,
)
from ..losses.norm_loss import normalization_loss
from ..losses.ortho_loss import orthonormality_loss
from ..losses.pde_loss import dirac_pde_loss
from ..losses.potential_prior import potential_prior_loss, potential_smooth_loss
from ..losses.q_corr_loss import q_residual
from ..losses.scf_consistency import scf_consistency_loss
from ..physics.dfs_potential import (
    build_dfs_potential,
    electron_density,
    zeff_anchor_potential,
)
from ..physics.slater_correction import slater_correction_potential
from ..utils.grid import RadialGrid

# (enabled, alpha_x, latter_tail, anchor_vprior, fermi_amaldi, scf_weight_mode,
#  anchor_vprior_zeff, path_a_enabled)
# bare-hydrogenic Round-1 default
_NO_DFS = (False, 1.0, True, False, True, "density", False, False)


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

    dfs_enabled, alpha_x, latter_tail, anchor_vprior, fermi_amaldi, scf_weight_mode, anchor_vprior_zeff, path_a_enabled = dfs_cfg
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

    # B' (Path A) inject: V_target = V_dfs + V_slater_corr (R^k→V_dfs 注入)
    # slater_log_scale 已在 model.__call__ 创建 (init=-3.0, exp=0.05)
    # Rk 来自 out["Rk"] (compute_all_Rk_diagonal 在 model 内已算)
    V_dfs_aug = V_dfs
    if path_a_enabled:
        Rk = out.get("Rk")
        slater_log_scale = out.get("slater_log_scale")
        if Rk is not None and slater_log_scale is not None:
            V_slater = slater_correction_potential(
                P, Q, omega, Rk, slater_log_scale, rho,
            )
            # B' fix v3: stop_gradient V_slater 防止其经 scf_loss 反传 P/Q 梯度
            # (E-prime baseline 训练时无此通道, 加上后破坏 H 1s 锚点)
            V_dfs_aug = V_dfs + jax.lax.stop_gradient(V_slater)

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
    # B' (Path A): SCF target = V_dfs_aug = V_dfs + V_slater_corr
    l_scf = (
        scf_consistency_loss(V, V_dfs_aug, weight=scf_weight)
        if V_dfs_aug is not None
        else jnp.asarray(0.0, dtype=V.dtype)
    )

    # --- Stage A Round 2: Laguerre-basis regularizers ---
    # These three terms are computed only when the Laguerre basis is active.
    # `out["laguerre_*"]` is set in `PinnArtModel.__call__` only when
    # `use_laguerre_basis=True`.  Otherwise the dict key is absent and the
    # value defaults to 0.
    K_max = 9  # default; matched at init in DeepONetDirac(K_max=9)
    lag_coeffs = out.get("laguerre_coeffs")
    lag_coeff_init = out.get("laguerre_coeff_init")
    lag_lambdas = out.get("laguerre_lambdas")
    lag_lambda_init = out.get("laguerre_lambda_init")
    lag_deltaQ = out.get("laguerre_deltaQ")
    if bool(os.environ.get("PINNART_DEBUG_LAG")) and jax.process_index() == 0:
        _lc = "None" if lag_coeffs is None else f"shape={lag_coeffs.shape}"
        _ll = "None" if lag_lambdas is None else f"shape={lag_lambdas.shape}"
        _ldq = "None" if lag_deltaQ is None else f"shape={lag_deltaQ.shape}"
        print(f"  [lag-dbg] coeff={_lc}  lambdas={_ll}  deltaQ={_ldq}", flush=True)
    l_coeff = (
        coeff_decay_loss(lag_coeffs, K_max) if lag_coeffs is not None
        else jnp.asarray(0.0, dtype=V.dtype)
    )
    # §13.C: anchor high-n coeffs back toward the analytic init (single-electron
    # P_H is exact; this counters training-induced degradation at n >= n_min).
    if lag_coeffs is not None and lag_coeff_init is not None and n_principal is not None:
        # NOTE: under jit the weights values are traced arrays, so keep
        # `n_min` as-is (the `>=` comparison handles a traced scalar fine).
        l_coeff_anchor = coeff_anchor_loss(
            lag_coeffs, lag_coeff_init, n_principal,
            orb_mask=orb_mask,
            n_min=weights.get("_coeff_anchor_n_min", 8.0),
        )
    else:
        l_coeff_anchor = jnp.asarray(0.0, dtype=V.dtype)
    if lag_lambdas is not None and lag_lambda_init is not None:
        # Per-batch λ_init is `Z_eff / n` for each row's (Z, n, l).
        # For inactive orbital slots both `lag_lambdas` and
        # `lag_lambda_init` are 1.0 (placeholder), so the loss vanishes
        # for those slots naturally — no extra mask required.
        l_lambda = lambda_prior_loss(lag_lambdas, lag_lambda_init)
    else:
        l_lambda = jnp.asarray(0.0, dtype=V.dtype)
    l_q = (
        q_residual(P, dPdr, V, kappa, grid,
                   deltaQ=lag_deltaQ,
                   perturb_scale_Q=jnp.asarray(
                       weights.get("_perturb_scale_Q", jnp.float32(0.05)),
                       dtype=jnp.float32,
                   ))
        if lag_deltaQ is not None
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
        + weights.get("coeff_decay", 0.0) * l_coeff
        + weights.get("coeff_anchor", 0.0) * l_coeff_anchor
        + weights.get("lambda_prior", 0.0) * l_lambda
        + weights.get("q_residual", 0.0) * l_q
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
        "coeff_decay": l_coeff,
        "coeff_anchor": l_coeff_anchor,
        "lambda_prior": l_lambda,
        "q_residual": l_q,
    }
    return total, metrics


@partial(jax.jit, static_argnames=("weights_key", "dfs_cfg"))
def _train_step_jit(state, batch, grid: RadialGrid, weights_vals, weights_key, dfs_cfg):
    weights = dict(zip(weights_key, weights_vals))
    # B''' fix: 跟随 dfs_cfg[7] (path_a_enabled) 决定 return_ci
    # - path_a_enabled=True  → return_ci=True (model 创建 slater_log_scale, V_slater 可注入)
    # - path_a_enabled=False → return_ci=False (baseline 模式, fast)
    return_ci_flag = bool(dfs_cfg[7])  # path_a_enabled 控制 return_ci

    def loss_fn(params):
        out = state.apply_fn(params, batch, grid, train=True, return_ci=return_ci_flag)
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
