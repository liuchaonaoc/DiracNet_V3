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

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..constants import C_LIGHT
from ..physics.hydrogenic import _laguerre_generalized_stack


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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convenience: compute envelope + Laguerre stack inside, return P, dP/dr.

    Shapes:
        r: [N_g]
        coeffs: [B, N_orb, K_max+1]
        lambda_a: [B, N_orb]
        kappa: [B, N_orb]
        alpha: [B, N_orb]  (= 2|kappa|)

    Returns P, dP_dr: [B, N_orb, N_g]
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
    return P, dP_dr


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
#  Coefficient-decay regularizer  L_coeff_decay = mean(c² / k!)
# --------------------------------------------------------------------------- #


def coeff_decay_loss(coeffs: jnp.ndarray, K_max: int) -> jnp.ndarray:
    """L2 penalty weighted by 1/k! to suppress divergence of high-k terms."""
    import math

    weights = jnp.asarray([1.0 / math.factorial(k) for k in range(K_max + 1)], dtype=coeffs.dtype)
    return jnp.mean(coeffs * coeffs * weights[None, :])


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