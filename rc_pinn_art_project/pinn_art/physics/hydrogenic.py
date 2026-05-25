"""Hydrogenic analytic references.

Two implementations of P_{n,l}(r; Z):

- `hydrogenic_P_analytic`  : numpy/scipy, single (n, l, Z), for tests/eval.
- `hydrogenic_P_jax`       : JAX-native, batched over (n, l) per orbital, used as
                              the analytic skeleton inside the network ansatz so
                              the SIREN only learns a small perturbation.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from scipy.special import genlaguerre

from ..utils.grid import RadialGrid


def hydrogenic_energy(Z: float | int, n: float | int) -> float:
    return -float(Z) ** 2 / (2.0 * float(n) ** 2)


def hydrogenic_P_analytic(r: jnp.ndarray, Z: float | int, n: int, l: int = 0) -> jnp.ndarray:
    if n <= l:
        raise ValueError("require n > l")
    rho = 2.0 * float(Z) * np.asarray(r) / float(n)
    fact = math.factorial(n - l - 1)
    fact_n = math.factorial(n + l)
    norm = math.sqrt((2.0 * float(Z) / float(n)) ** 3 * fact / (2.0 * n * fact_n))
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    R = norm * (rho ** l) * np.exp(-rho / 2.0) * L
    P = (rho / (2.0 * float(Z) / float(n))) * R
    return jnp.asarray(P, dtype=r.dtype if hasattr(r, "dtype") else jnp.float32)


# --------------------------------------------------------------------------- #
#  JAX-native batched P_{n,l}(r; Z)
# --------------------------------------------------------------------------- #


def _laguerre_generalized_stack(
    rho: jnp.ndarray, alpha: jnp.ndarray, max_k: int
) -> jnp.ndarray:
    """L_k^alpha(rho) for k = 0..max_k.

    rho   : [..., N_g]
    alpha : [...] (broadcast to rho without the trailing N_g axis)
    return: [max_k+1, ..., N_g]   (L_0, L_1, ..., L_{max_k})
    """
    a = alpha[..., None]
    L0 = jnp.ones_like(rho)
    if max_k == 0:
        return jnp.stack([L0], axis=0)
    L1 = 1.0 + a - rho
    Ls = [L0, L1]
    L_prev, L_curr = L0, L1
    for j in range(1, max_k):
        jf = jnp.float32(j)
        L_next = ((2.0 * jf + 1.0 + a - rho) * L_curr - (jf + a) * L_prev) / (jf + 1.0)
        Ls.append(L_next)
        L_prev, L_curr = L_curr, L_next
    return jnp.stack(Ls, axis=0)


def hydrogenic_P_jax(
    r: jnp.ndarray,
    Z: jnp.ndarray,
    n: jnp.ndarray,
    l: jnp.ndarray,
    *,
    max_k: int = 6,
) -> jnp.ndarray:
    """Batched JAX-native P_{n,l}(r; Z).

    P_{n,l}(r; Z) = r * R_{n,l}(r; Z), normalized so ∫ P^2 dr = 1.

    r : [N_g]
    Z : [B]                 (broadcast to [B, N_orb])
    n : [B, N_orb] (int)    (n_principal; if n <= l the orbital is degenerate
                              and we return zeros — caller must mask)
    l : [B, N_orb] (int)
    return: [B, N_orb, N_g]
    """
    rf = r.astype(jnp.float32)
    Zf = Z.astype(jnp.float32)[:, None]
    n_f = jnp.maximum(n.astype(jnp.float32), 1.0)
    l_f = jnp.clip(l.astype(jnp.float32), 0.0, n_f - 1.0)
    alpha = 2.0 * l_f + 1.0
    k = jnp.clip(n_f - l_f - 1.0, 0.0, float(max_k)).astype(jnp.int32)

    lam = Zf / n_f  # [B, N_orb]
    rho = 2.0 * lam[..., None] * rf[None, None, :]  # [B, N_orb, N_g]

    # log of normalization N_{n,l,Z}
    #   N = sqrt((2Z/n)^3  *  (n-l-1)! / (2n (n+l)!))
    #   log N = 1.5 log(2Z/n) + 0.5 (lgamma(n-l) - log(2n) - lgamma(n+l+1))
    log_norm = (
        1.5 * jnp.log(2.0 * lam)
        + 0.5 * (gammaln(n_f - l_f) - jnp.log(2.0 * n_f) - gammaln(n_f + l_f + 1.0))
    )
    norm = jnp.exp(log_norm)[..., None]  # [B, N_orb, 1]

    Ls = _laguerre_generalized_stack(rho, alpha, max_k=max_k)  # [max_k+1, B, N_orb, N_g]
    L_k = jnp.take_along_axis(
        Ls, k[None, ..., None].astype(jnp.int32), axis=0
    )[0]  # [B, N_orb, N_g]

    rho_pow_l = jnp.exp(l_f[..., None] * jnp.log(jnp.clip(rho, 1e-30)))
    R = norm * rho_pow_l * jnp.exp(-0.5 * rho) * L_k  # [B, N_orb, N_g]
    P = rf[None, None, :] * R  # P = r * R

    # zero out invalid (n <= l) orbitals
    valid = (n >= (l + 1)).astype(P.dtype)[..., None]
    return P * valid


def hydrogenic_dP_dr_jax(
    r: jnp.ndarray,
    Z: jnp.ndarray,
    n: jnp.ndarray,
    l: jnp.ndarray,
    *,
    max_k: int = 6,
) -> jnp.ndarray:
    """Analytic derivative of `hydrogenic_P_jax` w.r.t. r.

    Computed as autodiff over r is awkward because r is a 1-D grid; we instead
    differentiate the closed form:
        d/dr [r R] = R + r dR/dr
    with
        dR/dr = R · (l/r − Z/n) + N (rho)^l exp(-rho/2) · dL_k^alpha/drho · (dρ/dr)
        dL_k^alpha/drho = -L_{k-1}^{alpha+1}(rho)  (Abramowitz/standard identity)
    """
    rf = r.astype(jnp.float32)
    Zf = Z.astype(jnp.float32)[:, None]
    n_f = jnp.maximum(n.astype(jnp.float32), 1.0)
    l_f = jnp.clip(l.astype(jnp.float32), 0.0, n_f - 1.0)
    alpha = 2.0 * l_f + 1.0
    k = jnp.clip(n_f - l_f - 1.0, 0.0, float(max_k)).astype(jnp.int32)

    lam = Zf / n_f
    rho = 2.0 * lam[..., None] * rf[None, None, :]
    drho_dr = 2.0 * lam[..., None]  # constant in r

    log_norm = (
        1.5 * jnp.log(2.0 * lam)
        + 0.5 * (gammaln(n_f - l_f) - jnp.log(2.0 * n_f) - gammaln(n_f + l_f + 1.0))
    )
    norm = jnp.exp(log_norm)[..., None]

    Ls = _laguerre_generalized_stack(rho, alpha, max_k=max_k)
    Ls_dalpha = _laguerre_generalized_stack(rho, alpha + 1.0, max_k=max_k)
    L_k = jnp.take_along_axis(Ls, k[None, ..., None].astype(jnp.int32), axis=0)[0]
    # dL_k^alpha/drho = -L_{k-1}^{alpha+1}; for k == 0 derivative is 0
    k_minus = jnp.clip(k - 1, 0, max_k)
    L_km1_a1 = jnp.take_along_axis(Ls_dalpha, k_minus[None, ..., None].astype(jnp.int32), axis=0)[0]
    dL_drho = jnp.where(k[..., None] > 0, -L_km1_a1, 0.0)

    rho_pow_l = jnp.exp(l_f[..., None] * jnp.log(jnp.clip(rho, 1e-30)))
    R = norm * rho_pow_l * jnp.exp(-0.5 * rho) * L_k
    # dR/drho = N rho^l exp(-rho/2) [ (l/rho - 1/2) L_k + dL_k/drho ]
    dR_drho = norm * rho_pow_l * jnp.exp(-0.5 * rho) * (
        (l_f[..., None] / jnp.clip(rho, 1e-30) - 0.5) * L_k + dL_drho
    )
    dR_dr = dR_drho * drho_dr
    dP_dr = R + rf[None, None, :] * dR_dr

    valid = (n >= (l + 1)).astype(dP_dr.dtype)[..., None]
    return dP_dr * valid


def cosine_signed(P_model: jnp.ndarray, P_ref: jnp.ndarray, grid: RadialGrid) -> jnp.ndarray:
    num = grid.integrate(P_model * P_ref, axis=-1)
    den = jnp.sqrt(
        grid.integrate(P_model * P_model, axis=-1) * grid.integrate(P_ref * P_ref, axis=-1)
    )
    return num / jnp.clip(den, 1e-12)
