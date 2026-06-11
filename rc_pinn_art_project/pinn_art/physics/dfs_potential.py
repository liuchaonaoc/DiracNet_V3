"""Self-consistent Dirac-Fock-Slater (DHFS) local central potential (R1.1).

Builds a fully differentiable screened local potential from the current
orbitals (no SCF iteration loop — the fixed point is reached by a consistency
loss during training, see ``losses/scf_consistency.py``):

    rho(r)  = sum_a omega_a (P_a^2 + Q_a^2)              radial density, INT rho dr = N
    V_H(r)  = (1/r) INT_0^r rho dr' + INT_r^inf rho/r' dr'   Hartree (direct screening)
    V_x(r)  = -(3/2) alpha_x (3/pi * rho_3d)^(1/3)       Slater X-alpha exchange
    V_dfs   = -Z/r + V_H + V_x                           (then Latter tail)

The Latter tail correction clamps the potential so the ion asymptotes to
``-(Z - N + 1)/r`` (the +1 removes the spurious local self-interaction in the
outer region), i.e. ``V <- min(V, -(Z-N+1)/r)``.

Conventions
-----------
``P_a = r R_a`` so ``rho`` above is the *radial* density (1/length) with
``INT rho dr = N``; the 3-D density entering the Slater term is
``rho_3d = rho / (4 pi r^2)``. With that substitution the Slater coefficient
``-(3/2) alpha_x (3/pi)^(1/3)`` equals the textbook ``-3 alpha_x (3/(8 pi))^(1/3)``.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..utils.grid import RadialGrid

_FOUR_PI = 4.0 * jnp.pi


def slater_effective_charge(
    Z: jnp.ndarray,
    n_principal: jnp.ndarray,
    omega: jnp.ndarray,
    orb_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-orbital screened nuclear charge from Slater's rules (warm-start, R1.3).

    Screening of orbital ``a`` by an electron in orbital ``b``:
      n_b == n_a       -> 0.35   (same group; self-screening removed below)
      n_b == n_a - 1   -> 0.85
      n_b <= n_a - 2   -> 1.00

    Z, ... shapes: Z [B]; n_principal, omega, orb_mask [B, N_orb].
    return: z_eff [B, N_orb], clipped to [1, Z].
    """
    n = n_principal.astype(jnp.float32)
    om = omega.astype(jnp.float32)
    if orb_mask is not None:
        om = om * orb_mask.astype(om.dtype)
    n_a = n[:, :, None]          # [B, N_orb_a, 1]
    n_b = n[:, None, :]          # [B, 1, N_orb_b]
    same = jnp.abs(n_a - n_b) < 0.5
    one_below = jnp.abs(n_a - n_b - 1.0) < 0.5
    deep = (n_b <= n_a - 1.5)
    w = jnp.where(same, 0.35, jnp.where(one_below, 0.85, jnp.where(deep, 1.0, 0.0)))
    sigma = jnp.einsum("nab,nb->na", w, om)  # sum over orb_b of w_ab * omega_b
    sigma = sigma - 0.35  # remove the orbital's self-contribution (same group, weight 0.35)
    z_eff = Z.astype(jnp.float32)[:, None] - jnp.clip(sigma, 0.0)
    return jnp.clip(z_eff, 1.0, Z.astype(jnp.float32)[:, None])


def electron_density(
    P: jnp.ndarray,
    Q: jnp.ndarray,
    omega: jnp.ndarray,
) -> jnp.ndarray:
    """Radial density ``rho(r) = sum_a omega_a (P_a^2 + Q_a^2)``.

    P, Q   : [B, N_orb, N_g]
    omega  : [B, N_orb]   (occupation numbers; 0 for inactive orbitals)
    return : [B, N_g]     (INT rho dr = N for normalized orbitals)
    """
    dens_a = P * P + Q * Q  # [B, N_orb, N_g]
    return jnp.einsum("bo,bog->bg", omega.astype(dens_a.dtype), dens_a)


def hartree_potential(rho: jnp.ndarray, grid: RadialGrid) -> jnp.ndarray:
    """Direct (Hartree) screening potential from the radial density.

    V_H(r) = (1/r) INT_0^r rho dr' + INT_r^inf rho/r' dr'   (cumulative, differentiable)
    """
    r = grid.r[None, :]
    w = grid.dr[None, :].astype(rho.dtype)
    r_safe = jnp.clip(r, 1e-8)

    inner_incl = jnp.cumsum(rho * w, axis=-1)              # INT_0^r rho dr' (<= r)
    outer_terms = (rho / r_safe) * w                       # rho/r' dr'
    outer_total = jnp.sum(outer_terms, axis=-1, keepdims=True)
    outer_excl = outer_total - jnp.cumsum(outer_terms, axis=-1)  # INT_{>r} rho/r' dr'
    return inner_incl / r_safe + outer_excl


def slater_exchange_potential(
    rho: jnp.ndarray,
    grid: RadialGrid,
    alpha_x: float = 1.0,
) -> jnp.ndarray:
    """Slater X-alpha local exchange ``-(3/2) alpha_x (3/pi rho_3d)^(1/3)``."""
    r = grid.r[None, :]
    r_safe = jnp.clip(r, 1e-8)
    rho_3d = jnp.clip(rho / (_FOUR_PI * r_safe * r_safe), 0.0)
    coeff = 1.5 * alpha_x * (3.0 / jnp.pi) ** (1.0 / 3.0)
    return -coeff * jnp.power(jnp.clip(rho_3d, 1e-30), 1.0 / 3.0)


def latter_tail_correction(
    V: jnp.ndarray,
    Z: jnp.ndarray,
    nele: jnp.ndarray,
    grid: RadialGrid,
) -> jnp.ndarray:
    """Clamp to the physical ionic asymptote ``-(Z - N + 1)/r`` (Latter 1955)."""
    r = jnp.clip(grid.r[None, :], 1e-8)
    z_tail = (Z.astype(V.dtype) - nele.astype(V.dtype) + 1.0)[:, None]
    v_tail = -z_tail / r
    return jnp.minimum(V, v_tail)


def fermi_amaldi_factor(nele: jnp.ndarray) -> jnp.ndarray:
    """Self-interaction-correction prefactor ``(N - 1) / N`` (Fermi-Amaldi).

    The Hartree (and the crude Slater exchange) built from the *total* density
    ``rho`` includes each electron screening itself. The Fermi-Amaldi factor
    removes that spurious self-interaction on average: for ``N = 1`` it is 0, so
    ``V_H + V_x`` vanishes and the potential collapses to the bare ``-Z/r`` (the
    exact single-electron limit, e.g. hydrogen). ``N >= 1`` is assumed.

    nele   : [B]
    return : [B, 1]  (broadcasts over the radial axis)
    """
    n = jnp.clip(nele.astype(jnp.float32), 1.0)
    return ((n - 1.0) / n)[:, None]


def build_dfs_potential(
    P: jnp.ndarray,
    Q: jnp.ndarray,
    omega: jnp.ndarray,
    Z: jnp.ndarray,
    nele: jnp.ndarray,
    grid: RadialGrid,
    *,
    alpha_x: float = 1.0,
    latter_tail: bool = True,
    fermi_amaldi: bool = True,
    return_components: bool = False,
):
    """Differentiable DHFS local central potential ``V_dfs(r)``.

    P, Q   : [B, N_orb, N_g]
    omega  : [B, N_orb]
    Z, nele: [B]
    fermi_amaldi : apply the ``(N-1)/N`` self-interaction correction to the
        electron-electron screening ``V_H + V_x`` (P0 fix; recovers ``-Z/r``
        exactly for one-electron systems).
    return : [B, N_g]  (or dict of components if ``return_components``)
    """
    r = jnp.clip(grid.r[None, :], 1e-8)
    rho = electron_density(P, Q, omega)
    V_nuc = -Z.astype(rho.dtype)[:, None] / r
    V_H = hartree_potential(rho, grid)
    V_x = slater_exchange_potential(rho, grid, alpha_x=alpha_x)
    # Fermi-Amaldi removes the average self-interaction in the e-e screening.
    sic = fermi_amaldi_factor(nele).astype(V_H.dtype) if fermi_amaldi else 1.0
    V_ee = sic * (V_H + V_x)
    V_dfs = V_nuc + V_ee
    if latter_tail:
        V_dfs = latter_tail_correction(V_dfs, Z, nele, grid)

    if return_components:
        return {
            "V_dfs": V_dfs,
            "rho": rho,
            "V_nuc": V_nuc,
            "V_H": V_H,
            "V_x": V_x,
            "sic": sic,
            "V_ee": V_ee,
        }
    return V_dfs


def zeff_anchor_potential(
    Z: jnp.ndarray,
    n_principal: jnp.ndarray,
    omega: jnp.ndarray,
    orb_mask: jnp.ndarray,
    grid: RadialGrid,
) -> jnp.ndarray:
    """Slater-screened effective-charge anchor potential (路径 B, R1.5).

    Builds a *radial* anchor ``V_zeff(r) = -Z_anchor / r`` where ``Z_anchor``
    is the occupation-weighted average of the per-orbital Slater-screened
    effective charges from :func:`slater_effective_charge`. The intuition is
    that Slater's rules (0.35 / 0.85 / 1.00) give a very cheap but very
    well-conditioned estimate of the **average** screening each valence
    electron feels; using it as an *anchor* (not the target) lets the DeepONet
    refine away from this baseline where the Dirac-DE model demands it.

    Parameters
    ----------
    Z           : [B]               nuclear charge(s)
    n_principal : [B, N_orb]        principal quantum number per orbital
    omega       : [B, N_orb]        occupation numbers per orbital
    orb_mask    : [B, N_orb]        1 for active orbital slots
    grid        : RadialGrid        r-grid; V_zeff is defined on it

    Returns
    -------
    V_zeff : [B, N_g]   ``-Z_anchor / r``  (broadcast over r)
    """
    z_eff = slater_effective_charge(Z, n_principal, omega, orb_mask)  # [B, N_orb]
    om = omega.astype(z_eff.dtype)
    if orb_mask is not None:
        om = om * orb_mask.astype(om.dtype)
    n_occ = jnp.clip(jnp.sum(om, axis=-1, keepdims=True), 1.0)  # [B, 1]
    z_anchor = jnp.sum(z_eff * om, axis=-1, keepdims=True) / n_occ  # [B, 1]
    z_anchor = jnp.clip(z_anchor, 1.0, Z.astype(z_anchor.dtype)[:, None])
    r = jnp.clip(grid.r[None, :], 1e-8)
    return -z_anchor / r  # [B, N_g]
