#!/usr/bin/env python3
"""P0 verification — Fermi-Amaldi self-interaction correction (no retraining).

Round 2 broke the energy scale *even for hydrogen* (H 2s predicted ~2.2 eV vs
NIST 10.2 eV) because ``build_dfs_potential`` screened the single electron with
its own density (self-interaction) with no correction. This script isolates the
**energy target** from the (separately broken) network training: it solves the
radial Schrodinger eigenproblem for hydrogen on a fine grid using

  (a) the bare nuclear  -1/r,
  (b) the *corrected*   V_dfs (Fermi-Amaldi ON),
  (c) the *uncorrected* V_dfs (Fermi-Amaldi OFF, the Round-2 behaviour),

and reports E(1s), E(2s) and the 1s->2s excitation vs NIST (10.199 eV).

If the P0 fix is correct, (a) and (b) must agree (V_ee vanishes for N=1) and
land near 10.2 eV, while (c) is strongly compressed.

    cd DiracNet_V3/rc_pinn_art_project && export PYTHONPATH=.
    python scripts/v3_verify_p0_fermi_amaldi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax.numpy as jnp
import numpy as np

from pinn_art.physics.dfs_potential import build_dfs_potential
from pinn_art.physics.hydrogenic import hydrogenic_P_jax
from pinn_art.utils.grid import make_radial_grid

HARTREE_EV = 27.211386245988
NIST_H_2S_EV = 10.19881  # NIST ASD: 1s -> 2s excitation of neutral H


def solve_l0_levels(r: np.ndarray, V: np.ndarray, n_states: int = 2) -> np.ndarray:
    """Lowest ``n_states`` l=0 bound energies (Hartree) of -1/2 u'' + V u = E u.

    Finite-difference on a uniform grid with u(0)=u(r_max)=0 (u = r R).
    """
    h = r[1] - r[0]
    n = r.shape[0]
    main = 1.0 / h**2 + V
    off = -0.5 / h**2 * np.ones(n - 1)
    H = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    evals = np.linalg.eigvalsh(H)
    evals = evals[evals < 0.0]
    return evals[:n_states]


def hydrogen_density_on(grid) -> jnp.ndarray:
    """Normalized H 1s radial density rho = |P|^2 (omega=1) on ``grid``."""
    Z = jnp.array([1])
    n = jnp.array([[1]])
    l = jnp.array([[0]])
    P = hydrogenic_P_jax(grid.r, Z, n, l)  # [1, 1, N_g]
    Q = jnp.zeros_like(P)
    # renormalize so INT P^2 dr = 1 on this grid
    norm = jnp.sqrt(grid.integrate(P[:, 0] ** 2))
    P = P / norm[:, None, None]
    return P, Q


def main() -> None:
    # Fine uniform grid for an accurate finite-difference eigensolver.
    r_min, r_max, n_grid = 1.0e-3, 60.0, 6000
    grid = make_radial_grid(r_min, r_max, n_grid, "linear")
    r = np.asarray(grid.r, dtype=np.float64)

    Z = jnp.array([1])
    nele = jnp.array([1])
    omega = jnp.array([[1.0]])
    P, Q = hydrogen_density_on(grid)

    V_bare = -1.0 / r
    V_corr = np.asarray(
        build_dfs_potential(P, Q, omega, Z, nele, grid, fermi_amaldi=True)[0],
        dtype=np.float64,
    )
    V_unco = np.asarray(
        build_dfs_potential(P, Q, omega, Z, nele, grid, fermi_amaldi=False)[0],
        dtype=np.float64,
    )

    rows = []
    for name, V in (
        ("bare  -1/r (reference)", V_bare),
        ("V_dfs  Fermi-Amaldi ON  (P0 fix)", V_corr),
        ("V_dfs  Fermi-Amaldi OFF (Round-2)", V_unco),
    ):
        e = solve_l0_levels(r, V, n_states=2)
        e1s, e2s = float(e[0]), float(e[1])
        exc_ev = (e2s - e1s) * HARTREE_EV
        err = exc_ev - NIST_H_2S_EV
        rows.append((name, e1s, e2s, exc_ev, err))

    print("\n=== P0 verification: hydrogen 1s -> 2s excitation ===")
    print(f"  grid: linear r in [{r_min}, {r_max}], n={n_grid}  (FD eigensolver)")
    print(f"  NIST target: {NIST_H_2S_EV:.4f} eV\n")
    hdr = f"{'potential':38s} {'E_1s (Ha)':>11s} {'E_2s (Ha)':>11s} {'exc (eV)':>10s} {'err vs NIST (eV)':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for name, e1s, e2s, exc, err in rows:
        print(f"{name:38s} {e1s:11.5f} {e2s:11.5f} {exc:10.4f} {err:16.4f}")

    # Sanity: the corrected potential must match bare -1/r for a single electron.
    max_v_ee = float(np.max(np.abs(V_corr - V_bare)))
    print(f"\n  max |V_corr - (-1/r)| = {max_v_ee:.3e}  (should be ~0: N=1 self-interaction removed)")
    print(f"  uncorrected screening shifts exc by "
          f"{rows[2][3] - rows[1][3]:+.3f} eV vs the fix\n")


if __name__ == "__main__":
    main()
