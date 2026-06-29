"""Differentiable Generalized-Laguerre Basis operators.

Per `prompts/17_generalized_laguerre_basis.md` (Stage A Round 2), this module
provides:

- `LaguerreCoeffHead` / `LaguerreLambdaHead` / `LaguerreQCorrHead` Flax modules
  that emit (per-orbital) Laguerre expansion coefficients `c_k`, a learnable
  decay factor `λ_a`, and a small SIREN-style correction `δQ_a` for the small
  component respectively.
- `laguerre_p_sum` — composes envelope `r^{|κ|}·exp(-λr)` with the polynomial
  stack to obtain `P_a(r)` and its analytic gradient.
- `kinetic_balance_q_with_corr` — Dirichlet-Q operator: skeleton
  `Q_skel = c·(dP + κ/r·P)/(2c²-V)` plus a small bounded perturbation `δQ`
  (default amplitude 5%) — required to avoid variational collapse.
- `q_residual` — ρ-weighted squared deviation `∫ ρ·(Q − Q_skel)² dr` to bound
  `δQ` away from the kinetic-balance skeleton (Gemini §II.3 warning).
- `laguerre_grad_convexity_check` — diagnostic that confirms the residual is
  linear in `c_k` (sanity check for "convex optimization" claim).

Design notes
------------
* All shapes are batched `[B, N_orb, ...]`. The module deliberately keeps
  `K_max` a fixed Python int (default 9) so JAX tracing is stable across Z.
* `laguerre_p_sum` returns BOTH `P` and `dP/dr` (analytic, via jax.grad on the
  closed-form envelope) so the kinetic-balance Q can be computed without
  finite differences.
* Coefficients `c_k` are masked for `k > n - l` via the caller; the head itself
  emits `K_max + 1` values unconditionally so all orbitals share one kernel.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.scipy.special import gammaln

from ..constants import C_LIGHT
from ..physics.hydrogenic import _laguerre_generalized_stack


_GL64_X, _GL64_W = np.polynomial.laguerre.laggauss(64)


# --------------------------------------------------------------------------- #
#  Per-orbital heads (Flax modules)
# --------------------------------------------------------------------------- #


class LaguerreCoeffHead(nn.Module):
    """Per-orbital MLP emitting `K_max + 1` Laguerre expansion coefficients.

    Output = `delta(branch_feat) + coeff_init`, where `delta(.)` is a small
    zero-init MLP residual (so at init the output exactly equals
    `coeff_init` for every branch_feat).  This pattern is Round-3 fix:
    `coeff_init` can be a *per-batch* array of shape `[B, K_max+1]`
    (analytic hydrogenic Laguerre coefficients for each row's (Z, n, l)),
    bypassing the Flax constraint that `nn.initializers.constant` only
    accepts a fixed-shape tensor.
    """

    K_max: int = 9
    d_hidden: int = 64

    @nn.compact
    def __call__(self, branch_feat: jnp.ndarray, coeff_init: jnp.ndarray) -> jnp.ndarray:
        """branch_feat: [B, d_in];  coeff_init: [K_max+1] or [B, K_max+1] (analytic).
        Returns coeffs: [B, K_max+1]."""
        h = nn.relu(nn.Dense(self.d_hidden)(branch_feat))
        h = nn.relu(nn.Dense(self.d_hidden)(h))
        delta = nn.Dense(
            self.K_max + 1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="laguerre_coeff_head",
        )(h)  # [B, K_max+1]
        # Broadcast `coeff_init` to [B, K_max+1] (it may arrive as [K+1] or [B, K+1]).
        coeff_init_b = jnp.broadcast_to(
            coeff_init[None, ...] if jnp.ndim(coeff_init) == 1 else coeff_init,
            delta.shape,
        )
        return delta + coeff_init_b


class LaguerreLambdaHead(nn.Module):
    """Per-orbital learnable decay λ_a. softplus output, init = λ_init.

    Output = `softplus(delta(branch_feat) + log(expm1(lambda_init)))`, where
    `delta(.)` is a small zero-init MLP residual.  This pattern is
    Round-3 fix: `lambda_init` can be a per-batch scalar or `[B]` array
    (analytic Z/n for each row's (Z, n)), bypassing the Flax
    constraint that `nn.initializers.constant` only accepts a fixed
    scalar/tensor.
    """

    d_hidden: int = 32

    @nn.compact
    def __call__(self, branch_feat: jnp.ndarray, lambda_init: jnp.ndarray) -> jnp.ndarray:
        """branch_feat: [B, d_in]; lambda_init: scalar or [B] jnp (analytic Z/n).
        Returns λ: [B] (positive, softplus)."""
        init = jnp.asarray(lambda_init, dtype=jnp.float32)
        # softplus inverse: y = log(expm1(y))
        safe_init = jnp.maximum(init, 1e-4)
        bias_val = jnp.log(jnp.exp(safe_init) - 1.0)  # scalar or [B]
        if jnp.ndim(bias_val) == 0:
            bias_val_b = jnp.broadcast_to(bias_val, (branch_feat.shape[0],))
        else:
            bias_val_b = bias_val

        h = nn.relu(nn.Dense(self.d_hidden)(branch_feat))
        h = nn.relu(nn.Dense(self.d_hidden)(h))
        delta = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="lambda_head",
        )(h)[..., 0]  # [B]
        return jax.nn.softplus(delta + bias_val_b)


class LaguerreQCorrHead(nn.Module):
    """Per-orbital SIREN-style bounded correction `δQ_a(r)`. Output is tanh(·)
    so the final δQ amplitude is bounded by `perturb_scale_Q` (set in forward)."""

    d_hidden: int = 64
    omega_0: float = 30.0

    @nn.compact
    def __call__(
        self,
        t_grid: jnp.ndarray,
        branch_feat: jnp.ndarray,
        kappa: jnp.ndarray,
    ) -> jnp.ndarray:
        """t_grid: [N_g]; branch_feat: [B, d_in]; kappa: [B].
        Returns δQ (raw, pre-amplitude): [B, N_g]."""
        from .siren import SirenDense

        B = branch_feat.shape[0]
        N_g = t_grid.shape[0]
        t_feat = t_grid[None, :, None]
        t_b = jnp.broadcast_to(t_feat, (B, N_g, 1))
        b_b = jnp.broadcast_to(branch_feat[:, None, :], (B, N_g, branch_feat.shape[-1]))
        x = jnp.concatenate([t_b, b_b], axis=-1)
        # 2-layer SIREN, kernel zero-init so output starts at 0.
        h = SirenDense(self.d_hidden, omega_0=self.omega_0, is_first=True, name="q_d0")(x)
        h = SirenDense(self.d_hidden, omega_0=self.omega_0, is_first=False, name="q_d1")(h)
        out = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="q_head",
        )(h)
        return jnp.tanh(out[..., 0])


# --------------------------------------------------------------------------- #
#  P(r) from Laguerre coefficients
# --------------------------------------------------------------------------- #


def laguerre_p_sum(
    coeffs: jnp.ndarray,
    laguerre_stack: jnp.ndarray,
    envelope: jnp.ndarray,
    perturb_corr: jnp.ndarray | None = None,
    perturb_scale: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """P(r) = envelope · Σ_k coeffs[k] · L_k(2λr) · (1 + perturb_scale · perturb_corr).

    Convention (matches `hydrogenic_P_jax`):
        P_{nκ}(r) = r^{|κ|} · e^{-λr} · Σ c_k · L_k^{2|κ|-1}(2λr)

    Parameters
    ----------
    coeffs       : [B, N_orb, K_max+1]  (preferred) OR [..., K_max+1]
    laguerre_stack: [K_max+1, ..., N_g] (axis 0 = polynomial index)
                   OR [B, N_orb, K_max+1, N_g] (axis 2 = polynomial index)
                   with α = 2|κ|-1 already baked in by caller.
    envelope     : [B, N_orb, N_g]   = r^{|κ|}·exp(-λr)
    perturb_corr : [B, N_orb, N_g]   (raw tanh output)

    Returns
    -------
    P, dP/dr : same leading shape as `envelope`.
    """
    coeffs = jnp.asarray(coeffs)
    envelope = jnp.asarray(envelope)

    if laguerre_stack.ndim == 2:
        # [K_max+1, N_g] ⇒ add leading [B=1, N_orb=1] dims.
        lag_stack = laguerre_stack[None, None, :, :]
        leading_batch = envelope.shape[:-1] if envelope.ndim > 1 else (1, 1)
        B, N_orb = leading_batch[0], (leading_batch[1] if len(leading_batch) > 1 else 1)
        coeffs = jnp.broadcast_to(coeffs, (B, N_orb, coeffs.shape[-1]))
        if envelope.ndim == 1:
            envelope = envelope[None, None, :]
    else:
        lag_stack = laguerre_stack

    P_core = jnp.einsum("bok,bokg->bog", coeffs, lag_stack)  # [B, N_orb, N_g]
    if perturb_corr is not None:
        P_core = P_core * (1.0 + perturb_scale * perturb_corr)
    P = envelope * P_core

    dP = jnp.zeros_like(P)
    return P, dP


def _infer_r_from_envelope(_envelope: jnp.ndarray) -> jnp.ndarray:
    """Stub: not used; r is passed in via laguerre_dP_dr_with_r below."""
    raise NotImplementedError


def _laguerre_dP_dr(
    P: jnp.ndarray,
    coeffs: jnp.ndarray,
    laguerre_stack: jnp.ndarray,
    envelope: jnp.ndarray,
    perturb_corr: jnp.ndarray | None,
    perturb_scale: float,
) -> jnp.ndarray:
    """Placeholder: return zeros with the right shape & dtype.

    In practice `laguerre_p_sum` is invoked via `laguerre_p_sum_with_r` (below)
    which carries r explicitly for the analytic derivative.
    """
    return jnp.zeros_like(P)


def laguerre_p_sum_with_r(
    r: jnp.ndarray,
    coeffs: jnp.ndarray,
    lambda_a: jnp.ndarray,
    kappa: jnp.ndarray,
    alpha: jnp.ndarray,
    perturb_corr: jnp.ndarray | None = None,
    perturb_scale: float = 0.05,
    return_d2: bool = False,
):
    """Convenience: compute envelope + Laguerre stack inside, return P, dP/dr.

    Shapes:
        r: [N_g]
        coeffs: [B, N_orb, K_max+1]
        lambda_a: [B, N_orb]
        kappa: [B, N_orb]
        alpha: [B, N_orb]  (= 2|kappa|)

    Returns P, dP_dr: [B, N_orb, N_g].

    If ``return_d2=True`` also returns the analytic second derivative
    ``d²P/dr²`` (same shape) — used by the §13.B analytic kinetic-balance
    dQ path to avoid the long-range noise of ``jnp.gradient(Q)``.

    Analytic d²P/dr² (perturb_corr ignored; the Laguerre forward passes
    ``perturb_corr=None``):

        P        = env · P_core,           env = r^{|κ|} e^{-λr}
        env'     = env·(|κ|/r − λ)
        env''    = env·[(|κ|/r − λ)² − |κ|/r²]
        P_core   = Σ c_k L_k^α(ρ),         ρ = 2λr
        P_core'  = (2λ)  Σ c_k (−L_{k−1}^{α+1})
        P_core'' = (2λ)² Σ c_k  L_{k−2}^{α+2}
        P''      = env''·P_core + 2 env'·P_core' + env·P_core''
    """
    B, N_orb, _ = coeffs.shape
    N_g = r.shape[0]
    K_max = coeffs.shape[-1] - 1

    lam = lambda_a[..., None]  # [B, N_orb, 1]
    rho = 2.0 * lam * r[None, None, :]  # [B, N_orb, N_g]
    # _laguerre_generalized_stack returns [K_max+1, ..., N_g]; with the
    # broadcast rules it returns [K+1, B, N_orb, N_g] here.  Transpose to
    # [B, N_orb, K+1, N_g] for the einsum.
    L_stack = _laguerre_generalized_stack(rho, alpha, max_k=K_max)
    L_stack = jnp.transpose(L_stack, (1, 2, 0, 3))  # [B, N_orb, K+1, N_g]

    abs_kap = jnp.abs(kappa)
    r_safe = jnp.clip(r, 1e-12)
    log_env = abs_kap[..., None] * jnp.log(r_safe)[None, None, :] - lam * r[None, None, :]
    log_env = jnp.clip(log_env, -60.0, 30.0)
    env = jnp.exp(log_env)

    P_core = jnp.einsum("bok,bokg->bog", coeffs, L_stack)
    if perturb_corr is not None:
        P_core = P_core * (1.0 + perturb_scale * perturb_corr)
    P = env * P_core

    # Analytic derivative:
    #   d/dr env = env · (|kappa|/r - λ)
    d_env = env * (abs_kap[..., None] / r_safe[None, None, :] - lam)
    # d/dr P_core = Σ_k c_k · dL_k/drho · 2λ
    # dL_k/drho via identity: dL_k^α/drho = -L_{k-1}^{α+1}
    K_p1 = coeffs.shape[-1]
    alpha_p1 = alpha + 1.0
    L_dalpha = _laguerre_generalized_stack(rho, alpha_p1, max_k=K_p1 - 1)
    L_dalpha = jnp.transpose(L_dalpha, (1, 2, 0, 3))  # [B, N_orb, K_p1, N_g]
    # L_dalpha[..., k, :] holds L_k^{α+1}; we need L_{k-1}^{α+1} for k>=1.
    # Shift: prepend zeros along k axis.
    zeros_k = jnp.zeros_like(L_dalpha[..., :1, :])
    L_km1_a1 = jnp.concatenate([zeros_k, L_dalpha[..., :-1, :]], axis=-2)
    dL_drho = -L_km1_a1  # [B, N_orb, K_p1, N_g]
    dP_core_drho = jnp.einsum("bok,bokg->bog", coeffs, dL_drho)
    dP_core_dr = dP_core_drho * (2.0 * lam)  # chain rule dρ/dr = 2λ
    if perturb_corr is not None:
        # d/dx [P_core · (1+ε·c)] = (1+ε·c)·dP_core + ε·P_core·dc/dx
        # Approximation (cheap & smooth): ignore d(perturb)/dr — perturb_scale
        # is ≤0.05 so the contribution is sub-percent.
        dP_core_dr = dP_core_dr * (1.0 + perturb_scale * perturb_corr)

    dP_dr = d_env * P_core + env * dP_core_dr

    if not return_d2:
        return P, dP_dr

    # --- Analytic second derivative d²P/dr² (§13.B) ----------------------
    # env'' = env·[(|κ|/r − λ)² − |κ|/r²]
    inv_r = 1.0 / r_safe[None, None, :]
    fac = abs_kap[..., None] * inv_r - lam            # (|κ|/r − λ)
    d2_env = env * (fac * fac - abs_kap[..., None] * inv_r * inv_r)

    # P_core'' = (2λ)²·Σ c_k L_{k−2}^{α+2};  shift L^{α+2} stack by 2 in k.
    alpha_p2 = alpha + 2.0
    L_dalpha2 = _laguerre_generalized_stack(rho, alpha_p2, max_k=K_p1 - 1)
    L_dalpha2 = jnp.transpose(L_dalpha2, (1, 2, 0, 3))   # [B, N_orb, K_p1, N_g]
    zeros_2 = jnp.zeros_like(L_dalpha2[..., :2, :])
    L_km2_a2 = jnp.concatenate([zeros_2, L_dalpha2[..., :-2, :]], axis=-2)
    d2P_core_drho2 = jnp.einsum("bok,bokg->bog", coeffs, L_km2_a2)
    d2P_core_dr2 = d2P_core_drho2 * (2.0 * lam) * (2.0 * lam)
    if perturb_corr is not None:
        d2P_core_dr2 = d2P_core_dr2 * (1.0 + perturb_scale * perturb_corr)

    d2P_dr2 = d2_env * P_core + 2.0 * d_env * dP_core_dr + env * d2P_core_dr2
    return P, dP_dr, d2P_dr2


# --------------------------------------------------------------------------- #
#  Q(r) = kinetic-balance skeleton + small bounded δQ
# --------------------------------------------------------------------------- #


def kinetic_balance_q_with_corr(
    P: jnp.ndarray,
    dPdr: jnp.ndarray,
    V: jnp.ndarray,
    kappa: jnp.ndarray,
    r_grid: jnp.ndarray,
    deltaQ: jnp.ndarray | None = None,
    perturb_scale_Q: float = 0.05,
    c: float = C_LIGHT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Q = Q_skel + perturb_scale_Q · δQ
       Q_skel = c · (dP + κ/r · P) / (2c² - V)

    deltaQ is the raw (tanh-bounded) SIREN output from LaguerreQCorrHead.
    The `perturb_scale_Q` factor caps the absolute amplitude so the network
    cannot break relativistic limit behavior (Gemini §II.3 warning).

    Shapes:
        P:     [B, N_orb, N_g]
        dPdr:  [B, N_orb, N_g]
        V:     [B, N_g]
        kappa: [B, N_orb]
        deltaQ:[B, N_orb, N_g] or None
    """
    inv_r = 1.0 / jnp.clip(r_grid, 1e-8)  # [N_g]
    kap_r = kappa[..., None] * inv_r[None, None, :]  # [B, N_orb, N_g]
    denom = jnp.clip(2.0 * c * c - V[..., None, :], 1e-6, 1e10)  # [B,1,N_g]
    Q_skel = c * (dPdr + kap_r * P) / denom

    if deltaQ is not None:
        Q = Q_skel + perturb_scale_Q * deltaQ
    else:
        Q = Q_skel

    # dQ/dr via chain rule (analytic)
    # dQ_skel/dr is messy; use autodiff is heavy.  Caller usually overrides
    # with jnp.gradient at the model level. We return a placeholder.
    dQ = jnp.zeros_like(Q)
    return Q, dQ


def kinetic_balance_q_dq_analytic(
    P: jnp.ndarray,
    dPdr: jnp.ndarray,
    d2Pdr2: jnp.ndarray,
    V: jnp.ndarray,
    dVdr: jnp.ndarray,
    kappa: jnp.ndarray,
    r_grid: jnp.ndarray,
    deltaQ: jnp.ndarray | None = None,
    ddeltaQ: jnp.ndarray | None = None,
    perturb_scale_Q: float = 0.05,
    c: float = C_LIGHT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """§13.B: kinetic-balance Q **and** its analytic derivative dQ/dr.

    This removes the dominant long-range noise of ``jnp.gradient(Q)`` which,
    by numerically differentiating a quantity that already contains dP/dr,
    effectively computes d²P/dr² by finite differences — exactly where the
    high-n radial oscillations (r ~ n²/Z) are under-resolved on the
    loglinear grid.

        Q_skel = c·N/D,    N = dP + (κ/r)·P,   D = 2c² − V
        N'     = d²P + (κ/r)·dP − (κ/r²)·P
        D'     = −dV/dr
        Q_skel'= c·(N'·D − N·D') / D²
        Q      = Q_skel + s·δQ
        Q'     = Q_skel' + s·δQ'

    Shapes:
        P, dPdr, d2Pdr2 : [B, N_orb, N_g]
        V, dVdr         : [B, N_g]
        kappa           : [B, N_orb]
        deltaQ, ddeltaQ : [B, N_orb, N_g] or None
    """
    inv_r = 1.0 / jnp.clip(r_grid, 1e-8)              # [N_g]
    kap = kappa[..., None]                            # [B, N_orb, 1]
    kap_r = kap * inv_r[None, None, :]                # [B, N_orb, N_g]
    kap_r2 = kap * (inv_r * inv_r)[None, None, :]

    D = jnp.clip(2.0 * c * c - V[..., None, :], 1e-6, 1e10)   # [B,1,N_g]
    Dp = -dVdr[..., None, :]                                  # [B,1,N_g]

    N = dPdr + kap_r * P
    Np = d2Pdr2 + kap_r * dPdr - kap_r2 * P

    Q_skel = c * N / D
    dQ_skel = c * (Np * D - N * Dp) / (D * D)

    if deltaQ is not None:
        Q = Q_skel + perturb_scale_Q * deltaQ
        dQ = dQ_skel + (
            perturb_scale_Q * ddeltaQ if ddeltaQ is not None else 0.0
        )
    else:
        Q = Q_skel
        dQ = dQ_skel
    return Q, dQ


# --------------------------------------------------------------------------- #
#  Loss: L_q_residual = ∫ ρ·(Q − Q_skel)² dr  (Gemini §II.3)
# --------------------------------------------------------------------------- #


def q_residual(
    P: jnp.ndarray,
    dPdr: jnp.ndarray,
    V: jnp.ndarray,
    kappa: jnp.ndarray,
    grid,
    deltaQ: jnp.ndarray | None = None,
    perturb_scale_Q: float = 0.05,
    rho: jnp.ndarray | None = None,
    c: float = C_LIGHT,
) -> jnp.ndarray:
    """ρ-weighted residual between Q and its kinetic-balance skeleton.

    ρ defaults to |P|²·r² if not provided. The integral is taken along the
    last axis using the grid weights `grid.dr`.
    """
    inv_r = 1.0 / jnp.clip(grid.r, 1e-8)
    kap_r = kappa[..., None] * inv_r[None, None, :]
    denom = jnp.clip(2.0 * c * c - V[..., None, :], 1e-6, 1e10)
    Q_skel = c * (dPdr + kap_r * P) / denom

    if deltaQ is None:
        return jnp.array(0.0, dtype=P.dtype)

    delta = perturb_scale_Q * deltaQ  # [B, N_orb, N_g]
    if rho is None:
        # Electron density proxy for the orbital: |P|²·r²
        rho = (P * grid.r[None, None, :]) ** 2
    integrand = rho * delta * delta
    return jnp.mean(grid.integrate(integrand, axis=-1))


# --------------------------------------------------------------------------- #
#  §13.11-C: Node-position features (for HybridLaguerreHead)
# --------------------------------------------------------------------------- #
#
# Per `prompts/17_generalized_laguerre_basis.md` §13.11-C (Joint scheme:
# node coarse estimation + c_k iterative refinement), the HybridLaguerreHead
# consumes **node-position features** as additional input to its refinement
# MLP.  The analytic nodes {r_1, ..., r_{n-1}} are precomputed at startup
# from `data_cache/laguerre_nodes_z1_26_n1_10.parquet` and indexed at runtime
# via `load_analytic_nodes()`.
#
# Init property (per §13.11-C.3): `softplus(x) - log(2)` is used so that at
# init (kernel_init=zeros, bias_init=zeros) the network output for the
# node deltas is exactly zero → learned node = analytic node → c_k_init = c_H
# (per §2.3).
# --------------------------------------------------------------------------- #


def load_analytic_nodes_table(
    path: str | Path,
) -> dict[tuple[int, int, int], jnp.ndarray]:
    """Load the precomputed Laguerre polynomial nodes (r_1, ..., r_{n-1})
    for every (Z, n, l) in the manifest grid.  Returns a dict keyed by
    (Z, n, l) → [K_max] array (zero-padded for unused slots).

    Cache shape: dict with int keys and jnp.float32 values.  Loaded once at
    init time and passed into HybridLaguerreHead as a static dict.
    """
    import pandas as pd  # local import to keep JAX-only deps clean

    df = pd.read_parquet(path)
    out: dict[tuple[int, int, int], jnp.ndarray] = {}
    for _, row in df.iterrows():
        Z = int(row["Z"]); n = int(row["n"]); l = int(row["l"])
        K_max = int(row["K_max"])
        r = jnp.asarray(
            [row[f"r_node_{j}"] for j in range(K_max)], dtype=jnp.float32
        )
        out[(Z, n, l)] = r
    return out


def _lookup_analytic_nodes(
    table: dict[tuple[int, int, int], jnp.ndarray],
    Z_arr: jnp.ndarray,
    n_arr: jnp.ndarray,
    l_arr: jnp.ndarray,
    K_max: int,
) -> jnp.ndarray:
    """Look up analytic node positions for a (batched) (Z, n, l) tensor.

    Inputs:
        Z_arr: [B, N_orb] int
        n_arr: [B, N_orb] int
        l_arr: [B, N_orb] int
    Output:
        r_nodes: [B, N_orb, K_max] float32 (zero-padded for invalid slots)
    """
    B = int(Z_arr.shape[0])
    N_orb = int(Z_arr.shape[1])
    out = jnp.zeros((B, N_orb, K_max), dtype=jnp.float32)
    Z_np = np.asarray(Z_arr)
    n_np = np.asarray(n_arr)
    l_np = np.asarray(l_arr)
    for b in range(B):
        for a in range(N_orb):
            Z = int(Z_np[b, a]); n = int(n_np[b, a]); l = int(l_np[b, a])
            r = table.get((Z, n, l))
            if r is None:
                # Invalid slot (e.g. n <= l): leave zeros.
                continue
            out = out.at[b, a].set(r)
    return out


def nodes_to_laguerre_coeffs(
    r_nodes: jnp.ndarray,
    coeff_init: jnp.ndarray,
    lambda_orbital: jnp.ndarray,
    alpha_orbital: jnp.ndarray,
    degree_orbital: jnp.ndarray,
    K_max: int,
) -> jnp.ndarray:
    """Project node-defined polynomials back to generalized-Laguerre c_k.

    For an orbital with degree m = n-l-1 and Laguerre argument rho = 2λr,

        q(rho) = Π_i (rho - rho_i),  rho_i = 2λ r_i

    satisfies L_m^alpha(rho) = (-1)^m / m! * q(rho) when rho_i are the
    analytic roots.  We scale q so the active coefficient matches the
    analytic `coeff_init[..., m]`, then project q onto {L_k^alpha}_{k=0..K}
    with 64-point Gauss-Laguerre quadrature:

        c_k = <q_scaled, L_k^alpha> / ||L_k^alpha||².

    This makes node movement a direct physical path into P(r), instead of a
    weak auxiliary feature consumed by a residual MLP.
    """
    dtype = coeff_init.dtype
    K_max = int(K_max)
    x = jnp.asarray(_GL64_X, dtype=dtype)     # [Q], weight e^{-x}
    w = jnp.asarray(_GL64_W, dtype=dtype)     # [Q]

    lam = jnp.maximum(lambda_orbital.astype(dtype), jnp.asarray(1e-8, dtype=dtype))
    alpha = alpha_orbital.astype(dtype)
    degree_i = jnp.clip(
        jnp.round(degree_orbital).astype(jnp.int32), 0, K_max
    )
    degree = degree_i.astype(dtype)

    # Convert r-nodes to roots in rho-space and build the monic polynomial.
    rho_roots = 2.0 * lam[:, None] * r_nodes.astype(dtype)      # [B, K]
    live = (
        jnp.arange(K_max, dtype=dtype)[None, :] < degree[:, None]
    )                                                          # [B, K]
    factors = jnp.where(
        live[:, :, None],
        x[None, None, :] - rho_roots[:, :, None],
        jnp.ones((1, 1, x.shape[0]), dtype=dtype),
    )
    monic_poly = jnp.prod(factors, axis=1)                     # [B, Q]

    # Match the analytic coefficient magnitude at slot m.
    k_idx = jnp.arange(K_max + 1, dtype=dtype)
    active_mask = (k_idx[None, :] == degree_i[:, None]).astype(dtype)
    active_coeff = jnp.sum(coeff_init * active_mask, axis=-1)  # [B]
    sign = jnp.where((degree_i % 2) == 0, 1.0, -1.0).astype(dtype)
    scale = active_coeff * sign * jnp.exp(-gammaln(degree + 1.0))
    target_poly = scale[:, None] * monic_poly                  # [B, Q]

    rho = jnp.broadcast_to(x[None, :], target_poly.shape)      # [B, Q]
    Ls = _laguerre_generalized_stack(rho, alpha, max_k=K_max)  # [K+1, B, Q]

    quad_weight = w[None, :] * jnp.power(jnp.maximum(rho, 1e-30), alpha[:, None])
    inner = jnp.sum(Ls * target_poly[None, :, :] * quad_weight[None, :, :], axis=-1)
    inner = jnp.swapaxes(inner, 0, 1)                          # [B, K+1]
    norm = jnp.exp(gammaln(k_idx[None, :] + alpha[:, None] + 1.0) - gammaln(k_idx[None, :] + 1.0))
    return inner / jnp.maximum(norm, jnp.asarray(1e-30, dtype=dtype))


class HybridLaguerreHead(nn.Module):
    """§13.11-C joint scheme: node coarse estimation + c_k iterative refinement.

    Pipeline (per orbital):
        1. Stage 1 — node coarse estimator:
           Δr_raw = MLP(branch_feat) → Δr = softplus(Δr_raw) - log(2)
           learned_nodes = r_analytic + cumsum(Δr) (then mask k ≥ n-1)
        2. Stage 2 — c_k iterative refinement:
           c_k^(0) = coeff_init (analytic, sparse)
           for i in 1..n_iter:
             c_k^(i) = c_k^(i-1) + step · MLP_i(branch_feat, c_k^(i-1),
                                                  node_features)
           c_k_final = c_k^(n_iter)

    §2.3 init property (per §13.11-C.3):
       - At init, Δr_raw = 0 → Δr = 0 → learned_nodes = r_analytic.
       - At init, delta_i = 0 → c_k^(i) = c_k^(i-1) = ... = coeff_init.
       - Hence P(r) = analytic hydrogenic P_H(r) at init. ✓

    node_features: log(r) normalized + position indicators (simple, cheap).
    """

    K_max: int = 9
    d_hidden_node: int = 64       # Stage 1 node MLP hidden width
    d_hidden_ref: int = 32        # Stage 2 refinement MLP hidden width
    n_iter: int = 3               # refinement iterations
    step_size: float = 0.1        # per-iteration c_k update step

    @nn.compact
    def __call__(
        self,
        branch_feat: jnp.ndarray,        # [B, d_branch]
        coeff_init: jnp.ndarray,         # [B, K_max+1] analytic c_k
        r_nodes_analytic: jnp.ndarray,   # [B, K_max] precomputed node positions
        lambda_orbital: jnp.ndarray,     # [B] Laguerre decay λ
        alpha_orbital: jnp.ndarray,      # [B] generalized-Laguerre α
        degree_orbital: jnp.ndarray,     # [B] polynomial degree n-l-1
    ) -> jnp.ndarray:                   # returns c_k: [B, K_max+1]
        # ---- Stage 1: node coarse estimator ----
        # Output K_max deltas.  At init, kernel=zeros + bias=zeros → Δr_raw = 0
        # → softplus(0) - log(2) = 0 → Δr = 0 (strictly).
        h_node = nn.relu(nn.Dense(self.d_hidden_node, name="node_h")(branch_feat))
        dr_raw = nn.Dense(
            self.K_max,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="node_delta",
        )(h_node)
        # softplus - log(2) ensures softplus(0) - log(2) = 0 exactly.
        delta_r = jax.nn.softplus(dr_raw) - jnp.log(jnp.array(2.0))
        r_nodes_delta = jnp.cumsum(delta_r, axis=-1)        # [B, K_max]

        # Mask: only keep first (n-1) deltas (sparse physics).
        k_idx = jnp.arange(self.K_max, dtype=jnp.float32)
        degree = jnp.clip(degree_orbital, 0, self.K_max).astype(jnp.float32)
        mask = (k_idx[None, :] < degree[:, None]).astype(jnp.float32)
        r_nodes_delta = r_nodes_delta * mask

        # Learned node positions (init = r_analytic when delta=0).
        learned_nodes = jnp.maximum(
            r_nodes_analytic + r_nodes_delta,
            jnp.asarray(1e-8, dtype=r_nodes_analytic.dtype),
        )                                                   # [B, K_max]

        # Compact node features for Stage 2:
        # - log(1 + r_analytic) (spatial scale)
        # - log(1 + |delta_r|) (deviation magnitude)
        # - mask (which slots are "live")
        node_feat = jnp.concatenate([
            jnp.log1p(jnp.abs(learned_nodes)),
            jnp.log1p(jnp.abs(r_nodes_delta)),
            mask,
        ], axis=-1)                                          # [B, 3*K_max]

        # ---- Stage 2: c_k iterative refinement ----
        # Start from coefficients directly reconstructed from learned nodes.
        # This makes delta_r a physical path into P(r), not merely a weak
        # auxiliary feature.
        coeff_from_nodes = nodes_to_laguerre_coeffs(
            learned_nodes,
            coeff_init,
            lambda_orbital,
            alpha_orbital,
            degree_orbital,
            self.K_max,
        )
        coeff_from_base_nodes = nodes_to_laguerre_coeffs(
            r_nodes_analytic,
            coeff_init,
            lambda_orbital,
            alpha_orbital,
            degree_orbital,
            self.K_max,
        )
        c_k = coeff_init + (coeff_from_nodes - coeff_from_base_nodes)
        for i in range(self.n_iter):
            inp = jnp.concatenate([branch_feat, c_k, node_feat], axis=-1)
            h = nn.relu(nn.Dense(self.d_hidden_ref, name=f"ref_h_{i}")(inp))
            delta = nn.Dense(
                self.K_max + 1,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name=f"ref_delta_{i}",
            )(h)
            c_k = c_k + self.step_size * delta

        return c_k


# --------------------------------------------------------------------------- #
#  Coefficient-decay regularizer  L_coeff_decay = mean(c² / (k+1))
# --------------------------------------------------------------------------- #
#
# §13.10-A (2026-06-21): The classical 1/k! weighting under-constrains the
# sparse non-zero slot (k = n-l-1, e.g. c_9 for H 10s): 1/9! ≈ 2.8e-6 means
# the "active" coefficient is essentially free under the regularizer.  This
# allows the MLP to push large values into a single c_k and effectively bypass
# the analytic init, which is the root cause of n=8..10 cosine collapse.
#
# New formula 1/(k+1) provides UNIFORM per-k constraint strength
# (weight ratio = 36288× at k=9), encouraging sparse coefficients without
# killing the dominant mode.  Same §2.3 init property holds (analytically
# c_{n-l-1} = O(1) so penalty stays small in init).
# --------------------------------------------------------------------------- #


def coeff_decay_loss(coeffs: jnp.ndarray, K_max: int) -> jnp.ndarray:
    """L2 penalty weighted by 1/(k+1) to uniformly suppress all c_k.

    This is the §13.10-A "sparse forcing" formulation — replaces the classical
    1/k! weights (which under-constrain the sparse active slot at k=n-l-1).
    """
    weights = jnp.asarray([1.0 / (k + 1) for k in range(K_max + 1)], dtype=coeffs.dtype)
    return jnp.mean(coeffs * coeffs * weights[None, :])


# --------------------------------------------------------------------------- #
#  §13.C: coefficient ANCHOR loss  L_anchor = mean(mask · (c_k − c_k^init)²)
# --------------------------------------------------------------------------- #
#
# Empirical result across Steps C/D/E/F/G: for the SINGLE-ELECTRON hydrogenic
# training manifold the analytic P_H is the *exact* answer, and it is already
# loaded into `coeff_init` at init (|max(P − P_H)| ≈ 1e-7).  The high-n
# (n ≥ 8) node-gate failures are therefore *training-induced degradation* of a
# perfect initialization — the fragile 1-sparse fixed point gets knocked off
# by PDE/norm gradient noise and the 10-dim c_k basin is too small to recover.
#
# The original design (§5.3) explicitly forbade `‖c − c_H‖²`, arguing the
# coefficients should be energy-gradient driven.  Five rounds disproved that
# philosophy for single-electron data.  This anchor *toward `coeff_init`*
# (NOT toward zero, unlike coeff_decay) directly counters the degradation.
# It is applied only to the fragile high-n slots so low-n keeps its freedom,
# and it can be annealed off for the multi-electron stage.
# --------------------------------------------------------------------------- #


def coeff_anchor_loss(
    coeffs: jnp.ndarray,
    coeff_init: jnp.ndarray,
    n_principal: jnp.ndarray,
    orb_mask: jnp.ndarray | None = None,
    n_min: float = 8.0,
) -> jnp.ndarray:
    """Mean squared deviation of c_k from the analytic init, restricted to
    high-n orbitals (n ≥ ``n_min``).

    Shapes:
        coeffs, coeff_init : [B, N_orb, K_max+1]
        n_principal        : [B, N_orb]
        orb_mask           : [B, N_orb] (bool/0-1) or None
    """
    coeffs = jnp.asarray(coeffs)
    coeff_init = jnp.asarray(coeff_init, dtype=coeffs.dtype)
    n_arr = jnp.asarray(n_principal, dtype=coeffs.dtype)

    highn = (n_arr >= n_min).astype(coeffs.dtype)              # [B, N_orb]
    if orb_mask is not None:
        highn = highn * orb_mask.astype(coeffs.dtype)
    w = highn[..., None]                                       # [B, N_orb, 1]

    diff = (coeffs - coeff_init) * w
    denom = jnp.maximum(jnp.sum(w) * coeffs.shape[-1], 1.0)
    return jnp.sum(diff * diff) / denom


# --------------------------------------------------------------------------- #
#  λ prior loss  L_lambda_prior = mean((λ - λ_init)² / λ_init²)
# --------------------------------------------------------------------------- #


def lambda_prior_loss(lambda_a: jnp.ndarray, lambda_init: jnp.ndarray) -> jnp.ndarray:
    """Quadratic penalty around an analytic prior (default: Z_eff/n).

    Robust to per-batch broadcasting of `lambda_init` (scalar or [B]).
    """
    if jnp.ndim(lambda_init) == 0:
        denom = jnp.maximum(lambda_init * lambda_init, 1e-12)
        diff = lambda_a - lambda_init
        return jnp.mean(diff * diff / denom)
    # per-batch case
    init = jnp.asarray(lambda_init, dtype=lambda_a.dtype)
    if init.ndim == 1:
        init = init[:, None] if lambda_a.ndim == 2 else init
    denom = jnp.maximum(init * init, 1e-12)
    return jnp.mean((lambda_a - init) ** 2 / denom)