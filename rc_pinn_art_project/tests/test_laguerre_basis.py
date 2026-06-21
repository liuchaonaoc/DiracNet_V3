"""Pure-operator unit tests for the Differentiable Generalized-Laguerre Basis
(per `prompts/17_generalized_laguerre_basis.md`).

These tests do NOT depend on the DeepONet / training loop; they only exercise
the building blocks so a regression in (a) Laguerre expansion matching the
hydrogen analytic reference, (b) node count, (c) orthogonality boundary, or
(d) Q-residual guard is caught immediately.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from pinn_art.constants import C_LIGHT
from pinn_art.nets.laguerre_basis import (
    kinetic_balance_q_with_corr,
    laguerre_p_sum,
    q_residual,
)
from pinn_art.physics.hydrogenic import (
    _laguerre_generalized_stack,
    hydrogenic_P_jax,
    hydrogenic_laguerre_coeffs,
)
from pinn_art.utils.grid import make_radial_grid


# --------------------------------------------------------------------------- #
#  test_hydrogenic_limit — network init (coeffs_hydrogenic, lambda=Z/n) ⇒ P = P_H
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "Z,n,l",
    [
        (1, 1, 0),
        (2, 2, 0),
        (3, 2, 0),
        (4, 2, 1),
        (8, 3, 1),
    ],
)
def test_hydrogenic_limit(Z, n, l):
    """With coeffs = analytic hydrogenic Laguerre coefficients and λ = Z/n,
    `laguerre_p_sum` must reproduce `hydrogenic_P_jax` to ~1e-3 relative error.
    """
    grid = make_radial_grid(r_min=1e-4, r_max=80.0, n_grid=512, scheme="loglinear")
    r = grid.r

    # Reference: JAX batched hydrogenic.
    Zb = jnp.array([Z], dtype=jnp.int32)
    nb = jnp.array([[n] + [0] * 3], dtype=jnp.int32)
    lb = jnp.array([[l] + [0] * 3], dtype=jnp.int32)
    P_ref = hydrogenic_P_jax(r, Zb, nb, lb)[0, 0]

    # Build from coefficients + λ.
    coeffs = hydrogenic_laguerre_coeffs(Z, n, l, K_max=9)
    K_max = coeffs.shape[0] - 1
    lam = jnp.array([Z / n], dtype=jnp.float32)
    # |κ| = l+1 ⇒ alpha = 2|κ|-1 = 2l+1
    abs_kappa = l + 1
    alpha = jnp.array([2 * abs_kappa - 1], dtype=jnp.float32)
    rho = 2.0 * lam * r  # [N_g]
    L_stack = _laguerre_generalized_stack(rho, alpha, max_k=K_max)  # [K_max+1, N_g]
    envelope = r ** abs_kappa * jnp.exp(-lam * r)
    P, _ = laguerre_p_sum(coeffs[None, :], L_stack, envelope[None, :])
    P_got = np.asarray(P[0, 0])

    err = float(jnp.max(jnp.abs(P_got - P_ref)))
    scale = float(jnp.max(jnp.abs(P_ref)) + 1e-6)
    assert err / scale < 1e-3, (
        f"hydrogenic limit fails Z={Z} n={n} l={l}: rel_err={err/scale:.3e}"
    )


# --------------------------------------------------------------------------- #
#  test_node_count — count_sign_changes(P) == n-l-1 for K_max >= n-l
# --------------------------------------------------------------------------- #


def count_nodes_radial(P_arr: np.ndarray, r_np: np.ndarray) -> int:
    """Robust node counter: number of zero-crossings of a radial wavefunction.

    Counts only transitions between two *non-zero* signs. Samples below the
    noise floor (|P| < 1e-3 * pmax) are treated as a "no-man's-land" and
    skipped, so the long FP-noise tail does not generate spurious crossings.
    """
    rmin_idx = int(np.searchsorted(r_np, 0.1))
    rmax_idx = int(np.searchsorted(r_np, 25.0))
    body = P_arr[rmin_idx:rmax_idx]
    if body.size < 2:
        return 0
    pmax = float(np.max(np.abs(body)))
    if pmax == 0.0:
        return 0
    sign_arr = np.sign(body)
    sign_arr = np.where(np.abs(body) > 1e-3 * pmax, sign_arr, 0)

    # Find pairs of consecutive non-zero entries with opposite signs.
    # Iterate via scipy-style: collapse to a string of non-zero signs.
    nonzero = sign_arr[sign_arr != 0]
    if nonzero.size < 2:
        return 0
    # Count adjacent opposite-sign pairs in the *non-zero-only* sequence.
    diffs = np.diff(nonzero)
    return int(np.sum(diffs != 0))


@pytest.mark.parametrize(
    "Z,n,l",
    [
        (1, 1, 0),  # 0 nodes
        (3, 2, 0),  # 1 node
        (4, 2, 1),  # 0 nodes
        (8, 3, 0),  # 2 nodes
        (8, 3, 2),  # 0 nodes
    ],
)
def test_node_count(Z, n, l):
    """At init (coeffs_hydrogenic, λ=Z/n), the Laguerre expansion must
    reproduce the exact n-l-1 sign changes of the hydrogenic P."""
    grid = make_radial_grid(r_min=1e-4, r_max=80.0, n_grid=4096, scheme="loglinear")
    r = grid.r

    coeffs = hydrogenic_laguerre_coeffs(Z, n, l, K_max=9)
    lam = jnp.array([Z / n], dtype=jnp.float32)
    abs_kappa = l + 1
    alpha = jnp.array([2 * abs_kappa - 1], dtype=jnp.float32)
    rho = 2.0 * lam * r
    L_stack = _laguerre_generalized_stack(rho, alpha, max_k=9)  # [10, N_g]
    envelope = r ** abs_kappa * jnp.exp(-lam * r)
    P, _ = laguerre_p_sum(coeffs[None, :], L_stack, envelope[None, :])
    P_arr = np.asarray(P[0, 0])
    r_np = np.asarray(r)

    sign_changes = count_nodes_radial(P_arr, r_np)
    expected = n - l - 1
    assert sign_changes == expected, (
        f"node count Z={Z} n={n} l={l}: got {sign_changes}, expected {expected}"
    )


# --------------------------------------------------------------------------- #
#  test_basis_orthogonality — ⟨ρ^α L_k^α L_j^α⟩ = δ_kj · N_k
# --------------------------------------------------------------------------- #


def test_basis_orthogonality_same_lambda_same_alpha():
    """When two Laguerre-basis functions share λ and α, the inner product with
    the natural weight ρ^α exp(-ρ) must be diagonal."""
    N_g = 4096
    rho = jnp.linspace(0.0, 60.0, N_g)
    alpha = jnp.array([3.0], dtype=jnp.float32)
    K = 3
    L_stack = _laguerre_generalized_stack(rho, alpha, max_k=K)  # [K+1, N_g]

    # Weight and density: w(ρ) = ρ^α exp(-ρ).
    w = (rho ** float(alpha[0])) * jnp.exp(-rho)  # [N_g]
    # Outer product weighted: [K+1, K+1]
    weighted = L_stack[:, None, :] * L_stack[None, :, :] * w[None, None, :]
    # Trapezoid rule on ρ.
    drho = (rho[-1] - rho[0]) / (N_g - 1)
    inner = jnp.sum(weighted, axis=-1) * drho

    # Theoretical normalization: Γ(k+α+1)/k!
    k_arr = jnp.arange(K + 1, dtype=jnp.float32)
    expected_diag = jnp.exp(jax_gammaln(k_arr + 4.0) - jax_gammaln(k_arr + 1.0))

    diag_vals = inner[jnp.eye(K + 1, dtype=bool)]
    diag_err = float(jnp.max(jnp.abs(diag_vals - expected_diag) / expected_diag))
    off = inner * (1.0 - jnp.eye(K + 1, dtype=inner.dtype))
    off_err = float(jnp.max(jnp.abs(off)) / float(jnp.max(jnp.abs(expected_diag))))
    assert diag_err < 1e-2, f"diag rel err {diag_err:.3e}"
    assert off_err < 1e-3, f"off-diag rel err {off_err:.3e}"


def jax_gammaln(x):
    from jax.scipy.special import gammaln

    return gammaln(x)


# --------------------------------------------------------------------------- #
#  test_orthogonality_breaks — when λ differs, different orbitals are NOT
#  naturally orthogonal (Gemini §2.6 warning).
# --------------------------------------------------------------------------- #


def test_orthogonality_breaks_for_different_alpha():
    """When two Laguerre-basis functions share λ but have DIFFERENT α (i.e.
    different orbital angular momentum l), the inner product with the natural
    weight ρ^α exp(-ρ) is NOT diagonal — the basis functions for different
    l live in different weighted L² spaces.

    Concretely: 2s (l=0 ⇒ α=1) and 2p (l=1 ⇒ α=3) built with the same λ=Z/n
    are NOT orthogonal under any single ρ^α·exp(-ρ) measure. We verify this
    by building each orbital in its OWN natural measure and computing the
    simple ∫P_2s P_2p dr inner product — which is non-zero."""
    grid = make_radial_grid(r_min=1e-4, r_max=80.0, n_grid=4096, scheme="loglinear")
    r = grid.r

    Z_2s, n_2s, l_2s = 3.0, 2, 0   # α = 2|κ|-1 = 1
    Z_2p, n_2p, l_2p = 3.0, 2, 1   # α = 2|κ|-1 = 3

    coeffs_2s = hydrogenic_laguerre_coeffs(Z_2s, n_2s, l_2s, K_max=9)
    coeffs_2p = hydrogenic_laguerre_coeffs(Z_2p, n_2p, l_2p, K_max=9)

    lam_2s = jnp.array([Z_2s / n_2s], dtype=jnp.float32)
    lam_2p = jnp.array([Z_2p / n_2p], dtype=jnp.float32)
    abs_kap_2s = l_2s + 1
    abs_kap_2p = l_2p + 1
    alpha_2s = jnp.array([2 * abs_kap_2s - 1], dtype=jnp.float32)
    alpha_2p = jnp.array([2 * abs_kap_2p - 1], dtype=jnp.float32)

    L_2s = _laguerre_generalized_stack(2.0 * lam_2s * r, alpha_2s, max_k=9)
    L_2p = _laguerre_generalized_stack(2.0 * lam_2p * r, alpha_2p, max_k=9)
    env_2s = r ** abs_kap_2s * jnp.exp(-lam_2s * r)
    env_2p = r ** abs_kap_2p * jnp.exp(-lam_2p * r)

    P_2s = jnp.einsum("k,k...->...", coeffs_2s, L_2s) * env_2s
    P_2p = jnp.einsum("k,k...->...", coeffs_2p, L_2p) * env_2p

    overlap = float(grid.integrate(P_2s * P_2p, axis=-1))
    norm_2s = float(grid.integrate(P_2s * P_2s, axis=-1))
    norm_2p = float(grid.integrate(P_2p * P_2p, axis=-1))
    cos = overlap / math.sqrt(norm_2s * norm_2p + 1e-12)

    # 2s and 2p with same λ are NOT orthogonal (different α ⇒ different
    # weighted measure); their plain ∫P_s P_p dr is non-zero.
    assert abs(cos) > 1e-3, f"2s/2p overlap suspiciously small: cos={cos:.3e}"


# --------------------------------------------------------------------------- #
#  test_normalization — ρ-weighted integral of (Q − Q_skel)² small when δQ=0
# --------------------------------------------------------------------------- #


def test_q_residual_zero_when_no_correction(r_grid):
    """If deltaQ is None, L_q_residual must be exactly 0."""
    r = r_grid.r
    P = jnp.exp(-r ** 2)
    dP = -2.0 * r * P
    V = -1.0 / jnp.clip(r, 1e-8)
    kappa = jnp.array([-1.0])
    res = float(q_residual(P[None, None, :], dP[None, None, :], V[None, :],
                           kappa, r_grid, deltaQ=None))
    assert abs(res) < 1e-12


def test_q_residual_grows_with_correction(r_grid):
    """Adding a constant δQ must increase the residual linearly with its
    squared magnitude (sanity check that the loss is well-defined)."""
    r = r_grid.r
    P = jnp.exp(-r ** 2)
    dP = -2.0 * r * P
    V = -1.0 / jnp.clip(r, 1e-8)
    kappa = jnp.array([-1.0])

    delta_small = 0.01 * jnp.tanh(r)
    delta_large = 0.10 * jnp.tanh(r)
    res_small = float(q_residual(P[None, None, :], dP[None, None, :], V[None, :],
                                 kappa, r_grid, deltaQ=delta_small[None, None, :]))
    res_large = float(q_residual(P[None, None, :], dP[None, None, :], V[None, :],
                                 kappa, r_grid, deltaQ=delta_large[None, None, :]))
    assert res_large > res_small * 50.0, (res_small, res_large)


def test_kinetic_balance_q_with_corr_recovers_skeleton():
    """`kinetic_balance_q_with_corr` with deltaQ=None must equal pure skeleton."""
    r = jnp.linspace(0.05, 20.0, 256)
    P = jnp.exp(-r ** 2)
    dP = -2.0 * r * P
    V = -3.0 / jnp.clip(r, 1e-8)
    kappa = jnp.array([-1.0])
    Q, _ = kinetic_balance_q_with_corr(P[None, None, :], dP[None, None, :], V[None, :],
                                       kappa, r, deltaQ=None, perturb_scale_Q=0.05)
    # Reference via direct formula
    Q_ref = C_LIGHT * (dP + (-1.0) / r * P) / jnp.clip(2 * C_LIGHT ** 2 - V, 1e-6, 1e10)
    err = float(jnp.max(jnp.abs(Q[0, 0] - Q_ref)))
    scale = float(jnp.max(jnp.abs(Q_ref)) + 1e-6)
    assert err / scale < 1e-4, err / scale


# --------------------------------------------------------------------------- #
#  test_lambda_range — λ stays near its prior when L_lambda_prior dominates
# --------------------------------------------------------------------------- #


def test_lambda_prior_loss_quadratic():
    """`L_lambda_prior = mean((λ - λ_init)^2 / λ_init^2)` must be quadratic and
    vanish when λ == λ_init."""
    from pinn_art.losses.coeff_loss import lambda_prior_loss

    lam_init = jnp.array([1.0, 2.0, 0.5])
    lam_eq = lam_init
    lam_off = lam_init * 1.5
    loss_eq = float(lambda_prior_loss(lam_eq, lam_init))
    loss_off = float(lambda_prior_loss(lam_off, lam_init))
    assert loss_eq < 1e-12
    # (1.5x prior)^2 = 0.25 deviation; mean ≈ 0.25
    assert 0.20 < loss_off < 0.30, loss_off