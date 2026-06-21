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


def _factorial_jit(k):
    """JIT-safe factorial: exp(gammaln(k+1)). Works on Python ints and JAX tracers."""
    return jnp.exp(gammaln(jnp.asarray(k, dtype=jnp.float32) + 1.0))


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
    alpha : broadcasts against rho.shape[:-1]. Accepted forms:
            - Python int/float (treated as scalar)
            - 0-d jnp scalar
            - jnp array whose shape == rho.shape[:-1]  (most common case)
    return: [max_k+1, ..., N_g]   (L_0, L_1, ..., L_{max_k})
    """
    a = jnp.asarray(alpha)
    target_shape = rho.shape[:-1]
    if a.shape != target_shape:
        # Squeeze any leading size-1 dims first (allows callers to pass [1] or
        # 0-d scalars against a 1-d rho).
        a = jnp.squeeze(a, axis=tuple(i for i, s in enumerate(a.shape) if s == 1))
        if a.shape != target_shape:
            a = jnp.broadcast_to(a, target_shape)
    # Re-introduce trailing axis so `a - rho` always broadcasts cleanly,
    # irrespective of whether the caller passed a 1-d or batched alpha.
    a = a[..., None]
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
    Zf = Z.astype(jnp.float32)
    Zf = Zf[:, None] if Zf.ndim == 1 else Zf  # [B,1] (per-atom) or [B,N_orb] (per-orbital z_eff)
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
    Zf = Z.astype(jnp.float32)
    Zf = Zf[:, None] if Zf.ndim == 1 else Zf  # [B,1] or [B,N_orb]
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


# --------------------------------------------------------------------------- #
#  Analytic Laguerre expansion coefficients for the hydrogenic reference.
#  P_{n,l}(r; Z) = N · r^l · exp(-Z r / n) · L_{n-l-1}^{2l+1}(2 Z r / n)
#                = r^l · exp(-λ r) · Σ_k c_k^H · L_k^{2l+1}(2 λ r)
#
#  For hydrogen only k = n-l-1 contributes (all other c_k^H == 0); the others
#  are filled with zeros so the array has the same shape (K_max+1,) expected
#  by `laguerre_basis.laguerre_p_sum`.
# --------------------------------------------------------------------------- #


def hydrogenic_laguerre_coeffs(
    Z: float | int | jnp.ndarray,
    n: int | jnp.ndarray,
    l: int | jnp.ndarray,
    K_max: int = 9,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Return analytic hydrogenic Laguerre expansion coefficients (length K_max+1).

    For the hydrogen atom the normalized radial large component has only
    ONE non-zero coefficient at k = n - l - 1; all other slots are zero.

    Convention (matches `hydrogenic_P_jax` and `laguerre_p_sum`):
        P(r) = r^{|κ|} · exp(-λ r) · Σ_k c_k · L_k^{2|κ|-1}(2 λ r)

    The c_k below is chosen so that the reconstruction equals the
    normalized `hydrogenic_P_jax`. With |κ| = l + 1 the extra factor is
    (2Z/n)^l · (Z/n) absorbed into the single non-zero coefficient:

        c[n-l-1] = (2Z/n)^l · (Z/n) · sqrt((2 Z/n)^3 · (n-l-1)! / (2 n · (n+l)!))

    JIT-friendly: uses `gammaln` so the routine accepts traced `n`, `l`, `Z`
    (e.g. per-orbital arrays from a batched `shell_table`). Scalar ints are
    still accepted (and converted via `jnp.asarray` to a tracer).

    Parameters
    ----------
    Z : nuclear charge (scalar or [B] / [B, N_orb])
    n : principal quantum number (int or batched)
    l : orbital quantum number   (int or batched)
    K_max : truncation order (>= n - l - 1)
    dtype : output dtype

    Returns
    -------
    coeffs : [K_max+1] jnp.ndarray (scalar inputs) or
             broadcast([B, ...], K_max+1) (batched inputs)
    """
    Z_f = jnp.asarray(Z, dtype=dtype)
    n_f = jnp.asarray(n, dtype=dtype)
    l_f = jnp.asarray(l, dtype=dtype)

    k = n_f - l_f - 1.0  # polynomial index with the one non-zero coefficient
    # Guard: n > l ⇒ k >= 0
    k_safe = jnp.maximum(k, 0.0)

    # |κ| = l + 1 ⇒ α = 2|κ|-1 = 2l + 1.
    # norm = sqrt((2Z/n)^3 * k! / (2 n (n+l)!))
    log_norm = (
        1.5 * jnp.log(2.0 * Z_f / jnp.maximum(n_f, 1e-12))
        + 0.5 * (gammaln(k_safe + 1.0) - jnp.log(2.0 * jnp.maximum(n_f, 1e-12)) - gammaln(n_f + l_f + 1.0))
    )
    norm = jnp.exp(log_norm)
    # `extra = (2Z/n)^l` folds the `(2Zr/n)^l` pre-factor of R_{n,l} into the
    # single non-zero coefficient so the reconstruction matches `hydrogenic_P_jax`.
    extra = jnp.power(2.0 * Z_f / jnp.maximum(n_f, 1e-12), l_f)
    c_nl = norm * extra  # [broadcast shape]

    # Build the [K_max+1] array, scattering c_nl into slot k.
    K_max = int(K_max)
    K_idx = jnp.arange(K_max + 1, dtype=dtype)
    k_round = jnp.round(k).astype(jnp.int32)
    # If k is a tracer (batched), broadcast; if scalar int, just index.
    mask = (K_idx[None, ...] == k_round[..., None]).astype(dtype)  # broadcast mask
    coeffs_batched = c_nl[..., None] * mask  # [broadcast, K_max+1]
    coeffs_batched = jnp.where(
        jnp.broadcast_to(K_idx, coeffs_batched.shape) <= k_round[..., None],
        coeffs_batched,
        jnp.zeros_like(coeffs_batched),
    )

    # Squeeze leading size-1 dims so scalar inputs return [K_max+1] like before.
    out_shape = coeffs_batched.shape
    if out_shape[:-1] == (1,) or coeffs_batched.ndim > 1 and all(d == 1 for d in out_shape[:-1]):
        return coeffs_batched[0]
    return coeffs_batched
