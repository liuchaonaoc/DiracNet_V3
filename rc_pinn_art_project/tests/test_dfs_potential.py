"""R1.1 — DFS local central potential physics."""

import jax.numpy as jnp
import numpy as np

from pinn_art.physics.dfs_potential import (
    build_dfs_potential,
    electron_density,
    fermi_amaldi_factor,
    hartree_potential,
    slater_effective_charge,
)
from pinn_art.physics.hydrogenic import hydrogenic_P_jax
from pinn_art.utils.grid import make_radial_grid


def _h_1s():
    """Hydrogen 1s, single electron (N=1)."""
    grid = make_radial_grid(1e-4, 60.0, 512, "loglinear")
    Z = jnp.array([1])
    n = jnp.array([[1]])
    l = jnp.array([[0]])
    P = hydrogenic_P_jax(grid.r, Z, n, l)
    Q = jnp.zeros_like(P)
    omega = jnp.array([[1.0]])
    return grid, Z, P, Q, omega


def _he_1s2():
    """Helium 1s^2 hydrogenic density on a fine grid."""
    grid = make_radial_grid(1e-4, 60.0, 512, "loglinear")
    Z = jnp.array([2])
    n = jnp.array([[1]])
    l = jnp.array([[0]])
    P = hydrogenic_P_jax(grid.r, Z, n, l)
    Q = jnp.zeros_like(P)
    omega = jnp.array([[2.0]])
    return grid, Z, P, Q, omega


def test_density_integrates_to_N():
    grid, Z, P, Q, omega = _he_1s2()
    rho = electron_density(P, Q, omega)
    N = float(grid.integrate(rho)[0])
    assert abs(N - 2.0) < 1e-2


def test_dfs_asymptote_neutral_he():
    # He is neutral closed shell -> Latter tail enforces V -> -(Z-N+1)/r = -1/r
    grid, Z, P, Q, omega = _he_1s2()
    nele = jnp.array([2])
    V = np.array(build_dfs_potential(P, Q, omega, Z, nele, grid, latter_tail=True)[0])
    r = np.array(grid.r)
    for rr in (15.0, 30.0, 45.0):
        i = int(np.argmin(np.abs(r - rr)))
        expected = -1.0 / r[i]
        assert abs(V[i] - expected) < 0.05 * abs(expected) + 1e-4


def test_dfs_raw_neutral_screens_to_zero():
    # Without the Latter tail OR self-interaction correction, a neutral atom
    # screens the full nuclear charge to ~0 at large r.
    grid, Z, P, Q, omega = _he_1s2()
    nele = jnp.array([2])
    V = np.array(
        build_dfs_potential(
            P, Q, omega, Z, nele, grid, latter_tail=False, fermi_amaldi=False
        )[0]
    )
    r = np.array(grid.r)
    i = int(np.argmin(np.abs(r - 45.0)))
    assert abs(V[i]) < 1e-3


# --- P0 fix: Fermi-Amaldi self-interaction correction --------------------------


def test_fermi_amaldi_factor_values():
    fac = np.array(fermi_amaldi_factor(jnp.array([1, 2, 10]))[:, 0])
    assert abs(fac[0] - 0.0) < 1e-6      # N=1 -> screening fully removed
    assert abs(fac[1] - 0.5) < 1e-6
    assert abs(fac[2] - 0.9) < 1e-6


def test_hydrogen_recovers_bare_minus_one_over_r():
    # The whole point of the P0 fix: a one-electron system must collapse to the
    # bare -Z/r everywhere (no self-screening), so H 1s/2s/... energies are exact.
    grid, Z, P, Q, omega = _h_1s()
    nele = jnp.array([1])
    comp = build_dfs_potential(
        P, Q, omega, Z, nele, grid, latter_tail=True, fermi_amaldi=True,
        return_components=True,
    )
    V = np.array(comp["V_dfs"][0])
    r = np.array(grid.r)
    bare = -1.0 / r
    # self-interaction screening must vanish identically for N=1
    assert float(np.max(np.abs(comp["V_ee"][0]))) < 1e-6
    rel = np.abs(V - bare) / (np.abs(bare) + 1e-8)
    assert np.max(rel[5:-5]) < 1e-4


def test_fermi_amaldi_halves_screening_for_helium():
    # For He (N=2) the correction scales V_H + V_x by (N-1)/N = 1/2.
    grid, Z, P, Q, omega = _he_1s2()
    nele = jnp.array([2])
    full = build_dfs_potential(
        P, Q, omega, Z, nele, grid, latter_tail=False, fermi_amaldi=False,
        return_components=True,
    )
    corr = build_dfs_potential(
        P, Q, omega, Z, nele, grid, latter_tail=False, fermi_amaldi=True,
        return_components=True,
    )
    v_ee_full = np.array(full["V_ee"][0])
    v_ee_corr = np.array(corr["V_ee"][0])
    ratio = v_ee_corr[10:-10] / (v_ee_full[10:-10] + 1e-12)
    assert np.allclose(ratio, 0.5, atol=1e-3)


def test_hartree_matches_direct_double_sum():
    # Analytic cumulative V_H vs the direct definition INT rho(r')/r_> dr'.
    grid, Z, P, Q, omega = _he_1s2()
    rho = electron_density(P, Q, omega)
    V_H = np.array(hartree_potential(rho, grid)[0])
    r = np.array(grid.r)
    w = np.array(grid.dr)
    rho_np = np.array(rho[0])
    r_gt = np.maximum.outer(r, r)  # r_> [i,j] = max(r_i, r_j)
    V_H_direct = np.einsum("ij,j->i", 1.0 / r_gt, rho_np * w)
    rel = np.abs(V_H - V_H_direct) / (np.abs(V_H_direct) + 1e-6)
    # interior agreement (endpoints carry trapezoid edge error)
    assert np.median(rel[10:-10]) < 1e-2


def test_slater_effective_charge_bounds():
    Z = jnp.array([11])  # Na: 1s2 2s2 2p6 3s1
    n = jnp.array([[1, 2, 2, 3]])
    omega = jnp.array([[2.0, 2.0, 6.0, 1.0]])
    mask = jnp.array([[True, True, True, True]])
    zeff = np.array(slater_effective_charge(Z, n, omega, mask)[0])
    assert np.all(zeff >= 1.0) and np.all(zeff <= 11.0)
    # 1s sees nearly the full nucleus; valence 3s is strongly screened
    assert zeff[0] > zeff[-1]
    assert zeff[0] > 9.0
    assert zeff[-1] < 4.0
