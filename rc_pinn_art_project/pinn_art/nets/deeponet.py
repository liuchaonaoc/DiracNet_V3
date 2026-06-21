"""DeepONet: branch (discrete config) + SIREN trunk (continuous t).

Phase-3 ansatz — analytic hydrogenic skeleton + small SIREN perturbation:

    P_a(r) = P_H_{n_a, l_a}(r; Z) * (1 + eps * shape_a(r, branch, kappa, n, l))
    Q_a(r) = c (dP/dr + (kappa/r) P) / (2 c^2 - V)         <-- kinetic balance
    dQ_a   = d Q_a / dr  (autodiff via jnp.gradient)

where P_H is the JAX-native normalized hydrogenic radial function
(`hydrogenic_P_jax`); it already carries the correct n−l−1 interior nodes,
exponential tail exp(−Zr/n), and short-range r^{l+1} behavior.  The SIREN
"shape" head is initialized to 0 (final Dense kernel/bias = 0) so the model
starts at P_a = P_H, then learns at most an `eps`-bounded multiplicative
perturbation (default 0.2) to capture multi-electron screening/correlation.

Setting `use_hydrogenic_skeleton=False` reverts to the Phase-1/2 envelope-only
ansatz (kept for backward compat with old checkpoints).
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from ..constants import C_LIGHT
from ..physics.hydrogenic import (
    hydrogenic_P_jax,
    hydrogenic_dP_dr_jax,
    hydrogenic_laguerre_coeffs,
)
from .laguerre_basis import (
    LaguerreCoeffHead,
    LaguerreLambdaHead,
    LaguerreQCorrHead,
    kinetic_balance_q_with_corr,
    laguerre_p_sum_with_r,
)
from .siren import SirenDense


def _per_orbital_laguerre_init(
    Z: jnp.ndarray,
    n_principal: jnp.ndarray,
    l_orbital: jnp.ndarray,
    K_max: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-orbital analytic Laguerre coefficients and λ priors from per-batch (Z, n, l).

    Inputs are batched: Z [B], n_principal/l_orbital [B, N_orb].  Returns
    coeff_init [B, N_orb, K_max+1] and lambda_init [B, N_orb].  Slots with
    n <= l are masked to zeros (and λ = 1.0 as a placeholder).

    This replaces the old "compute once at model construction" approach
    (which fixed (n, l) per slot for the whole training run).  Computing
    inside the JIT region lets the same slot reuse different (n, l, Z)
    across batches — necessary because the training manifest uses slot 0
    to encode H 1s, H 2s, H 3s, ... across different rows.
    """
    Zf = Z.astype(jnp.float32)
    nb = n_principal.astype(jnp.int32)
    lb = l_orbital.astype(jnp.int32)
    valid = (nb > lb) & (nb >= 1)  # [B, N_orb]

    # For invalid slots feed (n=1, l=0) so the analytic helper returns a
    # benign non-NaN shape; we mask the result afterwards.
    nb_safe = jnp.where(valid, nb, 1)
    lb_safe = jnp.where(valid, lb, 0)
    Zf_safe = jnp.broadcast_to(Zf[:, None], nb_safe.shape)

    flat_n = nb_safe.reshape(-1)
    flat_l = lb_safe.reshape(-1)
    flat_Z = Zf_safe.reshape(-1)
    coeffs_flat = hydrogenic_laguerre_coeffs(flat_Z, flat_n, flat_l, K_max=K_max)
    if coeffs_flat.ndim == 1:
        coeffs_flat = coeffs_flat[None, :]
    target_shape = (nb_safe.size, K_max + 1)
    coeffs_flat = jnp.broadcast_to(coeffs_flat, target_shape)
    coeffs_all = coeffs_flat.reshape(*nb_safe.shape, K_max + 1)
    coeffs_all = coeffs_all * valid[..., None].astype(jnp.float32)

    # λ_init = Z_eff / n (using Z as a stand-in for Z_eff at init).
    lam_all = jnp.where(
        valid, Zf_safe / jnp.maximum(nb_safe.astype(jnp.float32), 1.0), 1.0
    )
    return coeffs_all, lam_all


class DeepONetDirac(nn.Module):
    d_branch: int = 128
    d_trunk: int = 128
    n_siren_layers: int = 4
    omega_0: float = 30.0
    n_orb_max: int = 16
    d_in_branch: int = 84
    use_hydrogenic_skeleton: bool = True
    perturb_eps: float = 0.2
    # --- Stage A Round 2 (Laguerre basis) toggles ---
    use_laguerre_basis: bool = False
    K_max: int = 9
    learn_lambda: bool = True
    perturb_scale_P: float = 0.05
    perturb_scale_Q: float = 0.05

    @nn.compact
    def __call__(
        self,
        branch_feat: jnp.ndarray,
        t_grid: jnp.ndarray,
        r_grid: jnp.ndarray,
        dt_dr: jnp.ndarray,
        kappa: jnp.ndarray,
        orb_mask: jnp.ndarray,
        Z: jnp.ndarray,
        n_principal: jnp.ndarray | None = None,
        l_orbital: jnp.ndarray | None = None,
        z_eff_orb: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        B = branch_feat.shape[0]
        N_g = t_grid.shape[0]
        N_orb = kappa.shape[1]

        b = nn.relu(nn.Dense(self.d_branch)(branch_feat))
        b = nn.Dense(self.d_branch)(b)

        t_feat = t_grid[None, :, None]
        t_broadcast = jnp.broadcast_to(t_feat, (B, N_g, 1))
        b_broadcast = jnp.broadcast_to(b[:, None, :], (B, N_g, self.d_branch))
        trunk_in = jnp.concatenate([t_broadcast, b_broadcast], axis=-1)

        V_raw = _siren_to_scalar(
            trunk_in,
            d_trunk=self.d_trunk,
            n_layers=self.n_siren_layers,
            omega_0=self.omega_0,
            final_bias=0.0,
            name_prefix="V",
        )
        V_corr = jnp.tanh(V_raw)
        V_nuc = -Z[:, None].astype(jnp.float32) / jnp.clip(r_grid[None, :], 1e-8)
        V = V_nuc + V_corr

        if n_principal is None:
            n_principal = jnp.maximum(jnp.abs(kappa), 1).astype(jnp.float32)
        # l from kappa if not provided:  l = |kappa+1/2| - 1/2  ⇒  for κ<0  l=-κ-1, κ>0 l=κ
        if l_orbital is None:
            l_orbital = jnp.where(kappa < 0, -kappa - 1, kappa).astype(jnp.int32)

        # Analytic hydrogenic skeleton (and its derivative) once for all orbitals.
        # Optional Z_eff warm-start (R1.3): per-orbital screened charge stabilizes
        # the n-l-1 node structure for many-electron (screened) systems.
        z_skel = Z if z_eff_orb is None else z_eff_orb
        if self.use_hydrogenic_skeleton:
            P_H = hydrogenic_P_jax(
                r_grid,
                z_skel,
                jnp.maximum(n_principal.astype(jnp.int32), 1),
                l_orbital.astype(jnp.int32),
            )  # [B, N_orb, N_g]
            dP_H = hydrogenic_dP_dr_jax(
                r_grid,
                z_skel,
                jnp.maximum(n_principal.astype(jnp.int32), 1),
                l_orbital.astype(jnp.int32),
            )

        n_principal = jnp.maximum(n_principal.astype(jnp.float32), 1.0)
        l_orbital_f = l_orbital.astype(jnp.float32)

        # ---- Stage A Round 2 (Round-3 fix v2): Laguerre-basis path ----
        # Per-batch analytic init (JIT-safe via gammaln-based
        # `hydrogenic_laguerre_coeffs`).  The same orbital slot can carry
        # different (n, l, Z) across manifest rows (H 1s in row 0, H 2s in
        # row 1, …); the analytic init now follows the per-batch (n, l, Z)
        # so the very first forward pass reproduces the analytic
        # hydrogenic P for whatever (Z, n, l) is in each row.
        lag_coeffs_out = None
        lag_lambdas_out = None
        lag_lambda_init_out = None
        lag_deltaQ_out = None
        if self.use_laguerre_basis:
            K_max = self.K_max
            N_orb = self.n_orb_max

            # Per-batch analytic init shapes: [B, N_orb, K_max+1] and [B, N_orb].
            coeff_init_per_orb, lam_init_per_orb = _per_orbital_laguerre_init(
                Z, n_principal, l_orbital_f.astype(jnp.int32), K_max
            )

            coeff_head = LaguerreCoeffHead(K_max=K_max, d_hidden=64)
            lambda_head = (
                LaguerreLambdaHead(d_hidden=32) if self.learn_lambda else None
            )
            q_corr_head = (
                LaguerreQCorrHead(d_hidden=64, omega_0=self.omega_0)
                if self.perturb_scale_Q > 0.0 else None
            )

            coeffs_all = jnp.zeros((B, N_orb, K_max + 1), dtype=jnp.float32)
            lambdas_all = jnp.zeros((B, N_orb), dtype=jnp.float32)
            deltaQ_all = jnp.zeros((B, N_orb, N_g), dtype=jnp.float32)
            for a_ in range(N_orb):
                # Per-batch init for slot a_  ⇒  [B, K_max+1] for coeffs
                # and [B] for lambda.  LaguerreCoeffHead /
                # LaguerreLambdaHead broadcast them internally.
                init_bias = coeff_init_per_orb[:, a_, :]  # [B, K+1]
                coeffs_all = coeffs_all.at[:, a_, :].set(
                    coeff_head(b, init_bias)
                )
                if lambda_head is not None:
                    lam_bias = lam_init_per_orb[:, a_]  # [B]
                    lambdas_all = lambdas_all.at[:, a_].set(
                        lambda_head(b, lam_bias)
                    )
                else:
                    lambdas_all = lambdas_all.at[:, a_].set(
                        lam_init_per_orb[:, a_]
                    )
                if q_corr_head is not None:
                    deltaQ_all = deltaQ_all.at[:, a_, :].set(
                        q_corr_head(t_grid, b, kappa[:, a_])
                    )

            # Mask out slots k > n - l (analytic orthogonality boundary).
            k_idx = jnp.arange(K_max + 1)[None, None, :]
            max_k_per_orb = jnp.maximum(
                n_principal - l_orbital_f - 1.0, 0.0
            )[..., None]
            coeff_mask = (k_idx <= max_k_per_orb).astype(jnp.float32)
            coeffs_all = coeffs_all * coeff_mask

            lag_coeffs_out = coeffs_all
            lag_lambdas_out = lambdas_all
            lag_lambda_init_out = lam_init_per_orb
            lag_deltaQ_out = deltaQ_all

        P_orbs = []
        Q_orbs = []
        dP_orbs = []
        dQ_orbs = []
        for a in range(N_orb):
            kap_a = kappa[:, a].astype(jnp.float32)
            n_a = n_principal[:, a]
            l_a = l_orbital_f[:, a]

            if self.use_laguerre_basis:
                # Laguerre path: P_a from (coeffs, λ_a), Q_a from kinetic
                # balance + δQ correction.
                coeffs_a = lag_coeffs_out[:, a, :]                # [B, K+1]
                lam_a = lag_lambdas_out[:, a]                     # [B]
                abs_kap_a = jnp.abs(kap_a)
                alpha_a = jnp.asarray(2 * abs_kap_a - 1.0, dtype=jnp.float32)
                coeffs_in = coeffs_a[:, None, :]
                P_a, dP_a = laguerre_p_sum_with_r(
                    r_grid,
                    coeffs_in,
                    lam_a[:, None],
                    kap_a[:, None],
                    alpha_a[:, None],
                    perturb_corr=None,
                    perturb_scale=self.perturb_scale_P,
                )
                P_a = P_a[:, 0, :]
                dP_a = dP_a[:, 0, :]
                Q_a, _ = kinetic_balance_q_with_corr(
                    P_a[:, None, :],
                    dP_a[:, None, :],
                    V,
                    kap_a[:, None],
                    r_grid,
                    deltaQ=lag_deltaQ_out[:, a, :][:, None, :],
                    perturb_scale_Q=self.perturb_scale_Q,
                )
                Q_a = Q_a[:, 0, :]
                dQ_a = jnp.gradient(Q_a, r_grid, axis=-1)
            else:
                kap_feat = jnp.broadcast_to((kap_a / 4.0)[:, None, None], (B, N_g, 1))
                n_feat = jnp.broadcast_to((n_a / 10.0)[:, None, None], (B, N_g, 1))
                l_feat = jnp.broadcast_to((l_a / 4.0)[:, None, None], (B, N_g, 1))
                orb_in = jnp.concatenate([trunk_in, kap_feat, n_feat, l_feat], axis=-1)

                # Initial perturbation MUST be zero so P_a = P_H at init.
                shape = _siren_to_scalar(
                    orb_in,
                    d_trunk=self.d_trunk,
                    n_layers=self.n_siren_layers,
                    omega_0=self.omega_0,
                    final_bias=0.0,
                    name_prefix=f"shape_{a}",
                )

                if self.use_hydrogenic_skeleton:
                    P_H_a = P_H[:, a, :]
                    dP_H_a = dP_H[:, a, :]
                    pert = 1.0 + self.perturb_eps * jnp.tanh(shape)
                    dpert_dr = self.perturb_eps * (1.0 - jnp.tanh(shape) ** 2) * jnp.gradient(
                        shape, r_grid, axis=-1
                    )
                    P_a = P_H_a * pert
                    dP_a = dP_H_a * pert + P_H_a * dpert_dr
                else:
                    env, denv = _envelope_with_derivative(r_grid, Z, kap_a, n_a)
                    P_a = env * (1.0 + shape)
                    dshape_dr = jnp.gradient(shape, r_grid, axis=-1)
                    dP_a = denv * (1.0 + shape) + env * dshape_dr

                Q_a = _kinetic_balance_q(P_a, dP_a, V, kap_a, r_grid)
                dQ_a = jnp.gradient(Q_a, r_grid, axis=-1)

            m = orb_mask[:, a, None].astype(P_a.dtype)
            P_orbs.append((P_a * m)[:, None, :])
            Q_orbs.append((Q_a * m)[:, None, :])
            dP_orbs.append((dP_a * m)[:, None, :])
            dQ_orbs.append((dQ_a * m)[:, None, :])

        return {
            "V": V,
            "P": jnp.concatenate(P_orbs, axis=1),
            "Q": jnp.concatenate(Q_orbs, axis=1),
            "dPdr": jnp.concatenate(dP_orbs, axis=1),
            "dQdr": jnp.concatenate(dQ_orbs, axis=1),
            # Stage A Round 2 extras (consumed by coeff_loss / q_corr_loss).
            "laguerre_coeffs": lag_coeffs_out,
            "laguerre_lambdas": lag_lambdas_out,
            "laguerre_lambda_init": lag_lambda_init_out,
            "laguerre_deltaQ": lag_deltaQ_out,
        }


def _siren_to_scalar(
    x: jnp.ndarray,
    *,
    d_trunk: int,
    n_layers: int,
    omega_0: float,
    final_bias: float,
    name_prefix: str,
) -> jnp.ndarray:
    """SIREN MLP returning a scalar per spatial point, init bias controls baseline."""
    h = SirenDense(d_trunk, omega_0=omega_0, is_first=True, name=f"{name_prefix}_d0")(x)
    for i in range(max(n_layers - 2, 0)):
        h = SirenDense(d_trunk, omega_0=omega_0, is_first=False, name=f"{name_prefix}_d{i + 1}")(h)
    out = nn.Dense(
        1,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.constant(final_bias),
        name=f"{name_prefix}_head",
    )(h)
    return out[..., 0]


def _envelope_with_derivative(
    r: jnp.ndarray, Z: jnp.ndarray, kappa: jnp.ndarray, n_principal: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """env = r^|kappa| * exp(-Z r / n);  denv/dr returned alongside."""
    abs_k = jnp.abs(kappa)
    n_clip = jnp.clip(n_principal, 1.0, 30.0)
    lam = Z.astype(jnp.float32) / n_clip
    r_c = jnp.clip(r, 1e-8)
    log_r = jnp.log(r_c)
    log_env = abs_k[:, None] * log_r[None, :] - lam[:, None] * r_c[None, :]
    env = jnp.exp(jnp.clip(log_env, -60.0, 30.0))
    denv = env * (abs_k[:, None] / r_c[None, :] - lam[:, None])
    return env, denv


def _kinetic_balance_q(
    P: jnp.ndarray, dPdr: jnp.ndarray, V: jnp.ndarray, kappa: jnp.ndarray, r_grid: jnp.ndarray,
    c: float = C_LIGHT,
) -> jnp.ndarray:
    """Q = c (dP + kappa/r P) / (2 c^2 - V)   (proper sign and scale).

    Reduces in NR limit to Q = (dP + kappa/r P)/(2c), and gives LP = E P + O(alpha^2).
    """
    inv_r = 1.0 / jnp.clip(r_grid, 1e-8)
    kap_r = kappa[:, None].astype(P.dtype) * inv_r[None, :]
    denom = jnp.clip(2.0 * c * c - V, 1e-6, 1e10)
    return c * (dPdr + kap_r * P) / denom
