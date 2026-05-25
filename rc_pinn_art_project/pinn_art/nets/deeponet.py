"""DeepONet: branch (discrete config) + SIREN trunk (continuous t).

Phase-1 ansatz for radial Dirac wavefunctions on hydrogen-like ions:

    P_a(r) = env(r; |kappa|, Z, n) * shape_a(t, branch)
    Q_a(r) = c (dP/dr + (kappa/r) P) / (2 c^2 - V)         <-- kinetic balance
    dQ_a   = d Q_a / dr  (autodiff via jnp.gradient)

where:
    env(r; |kappa|, Z, n) = r^{|kappa|} * exp(-Z r / n)
    shape(t)              = 1 + tanh(SIREN(t,branch,kappa))   # init to 1, range (0,2)
                          # but final Dense is zero/ones → shape_init = 1

The shape function is a free SIREN output so it can develop interior nodes
required for excited states (2s, 3s, …).  The envelope handles short-range
power law and long-range exponential decay; the SIREN can multiply by an
arbitrary polynomial-like factor.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from ..constants import C_LIGHT
from .siren import SirenDense


class DeepONetDirac(nn.Module):
    d_branch: int = 128
    d_trunk: int = 128
    n_siren_layers: int = 4
    omega_0: float = 30.0
    n_orb_max: int = 16
    d_in_branch: int = 84

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
        else:
            n_principal = jnp.maximum(n_principal.astype(jnp.float32), 1.0)

        P_orbs = []
        Q_orbs = []
        dP_orbs = []
        dQ_orbs = []
        for a in range(N_orb):
            kap_a = kappa[:, a].astype(jnp.float32)
            n_a = n_principal[:, a]
            kap_feat = jnp.broadcast_to((kap_a / 4.0)[:, None, None], (B, N_g, 1))
            n_feat = jnp.broadcast_to((n_a / 10.0)[:, None, None], (B, N_g, 1))
            orb_in = jnp.concatenate([trunk_in, kap_feat, n_feat], axis=-1)

            shape = _siren_to_scalar(
                orb_in,
                d_trunk=self.d_trunk,
                n_layers=self.n_siren_layers,
                omega_0=self.omega_0,
                final_bias=1.0,
                name_prefix=f"shape_{a}",
            )

            env, denv = _envelope_with_derivative(r_grid, Z, kap_a, n_a)

            P_a = env * shape
            dshape_dr = jnp.gradient(shape, r_grid, axis=-1)
            dP_a = denv * shape + env * dshape_dr

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
