#!/usr/bin/env python3
"""Visualize reference V(r), P(r), Q(r) for Z=3, l=0 (Li s orbitals).

NIST / experiment supply *energies*, not radial amplitudes.  This script
builds numerical references in PINN-ART conventions (atomic units, P = r R):

  1. Hydrogenic analytic P with Slater Z_eff; Q from Dirac kinetic balance
  2. DFS self-consistent V(r) from those orbitals (Hartree + Slater Xα + …)
  3. Optional fixed-point: recompute Q in V_dfs until potential stabilizes

Outputs PNG figures under ``logs/reference_z3_l0/``.

    cd DiracNet_V3/rc_pinn_art_project && export PYTHONPATH=.
    python scripts/v3_visualize_z3_l0_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pinn_art.constants import C_LIGHT, HARTREE_TO_EV, hartree_to_meV
from pinn_art.physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from pinn_art.physics.dfs_potential import build_dfs_potential, slater_effective_charge
from pinn_art.physics.hydrogenic import hydrogenic_P_analytic
from pinn_art.utils.grid import make_radial_grid

OUT_DIR = ROOT / "logs" / "reference_z3_l0"
C = float(C_LIGHT)

# Li neutral ground 1s²2s¹
LI_ORBITALS = [
    {"label": "1s", "n": 1, "l": 0, "kappa": -1, "omega": 2.0},
    {"label": "2s", "n": 2, "l": 0, "kappa": -1, "omega": 1.0},
]


def kinetic_balance_q(P: np.ndarray, dPdr: np.ndarray, V: np.ndarray, kappa: int, r: np.ndarray) -> np.ndarray:
    inv_r = 1.0 / np.maximum(r, 1e-8)
    denom = np.clip(2.0 * C * C - V, 1e-6, 1e10)
    return C * (dPdr + kappa * inv_r * P) / denom


def hydrogenic_p(r: np.ndarray, Z: float, n: int, l: int = 0) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(hydrogenic_P_analytic(jnp.asarray(r), Z, n, l), dtype=np.float64)
    dPdr = np.gradient(P, r)
    return P, dPdr


def normalize(P: np.ndarray, Q: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray]:
    nrm = float(grid.integrate(jnp.asarray(P * P + Q * Q)))
    s = np.sqrt(max(nrm, 1e-30))
    return P / s, Q / s


def orbital_energy(P: np.ndarray, Q: np.ndarray, V: np.ndarray, kappa: int, grid) -> float:
    r = np.asarray(grid.r)
    dP = np.gradient(P, r)
    dQ = np.gradient(Q, r)
    P_b = jnp.asarray(P)[None, None, :]
    Q_b = jnp.asarray(Q)[None, None, :]
    LP, LQ = dirac_apply(
        P_b, Q_b, jnp.asarray(dP)[None, None, :], jnp.asarray(dQ)[None, None, :],
        jnp.asarray(V)[None, :], jnp.array([[kappa]]), grid.r,
    )
    return float(orbital_energy_from_dirac(P_b, Q_b, LP, LQ, grid)[0, 0])


def build_reference(
    grid,
    *,
    Z: float = 3.0,
    nele: float = 3.0,
    max_iter: int = 30,
    tol: float = 1e-5,
) -> dict:
    """Hydrogenic P + kinetic-balance Q in iterated DFS potential."""
    r = np.asarray(grid.r, dtype=np.float64)

    n_arr = jnp.array([[float(o["n"]) for o in LI_ORBITALS]])
    l_arr = jnp.array([[0, 0]])
    om_arr = jnp.array([[o["omega"] for o in LI_ORBITALS]])
    mask = jnp.array([[True, True]])
    z_eff = np.asarray(slater_effective_charge(jnp.array([Z]), n_arr, om_arr, mask)[0])

    # Bare-Z hydrogenic (single-electron limit) for dashed comparison
    bare = []
    for i, orb in enumerate(LI_ORBITALS):
        P, dP = hydrogenic_p(r, Z, orb["n"], orb["l"])
        V_bare = -Z / np.maximum(r, 1e-8)
        Q = kinetic_balance_q(P, dP, V_bare, orb["kappa"], r)
        P, Q = normalize(P, Q, grid)
        bare.append({"label": orb["label"], "P": P, "Q": Q, "Z": Z})

    # Slater-screened hydrogenic warm start
    orbitals = []
    for i, orb in enumerate(LI_ORBITALS):
        P, dP = hydrogenic_p(r, float(z_eff[i]), orb["n"], orb["l"])
        V0 = -Z / np.maximum(r, 1e-8)
        Q = kinetic_balance_q(P, dP, V0, orb["kappa"], r)
        P, Q = normalize(P, Q, grid)
        orbitals.append({**orb, "P": P, "Q": Q, "z_eff": float(z_eff[i])})

    V = -Z / np.maximum(r, 1e-8)
    history = []

    for it in range(max_iter):
        P_st = jnp.asarray([o["P"] for o in orbitals])[None, ...]
        Q_st = jnp.asarray([o["Q"] for o in orbitals])[None, ...]
        om_st = jnp.asarray([[o["omega"] for o in orbitals]])
        V_new = np.asarray(
            build_dfs_potential(
                P_st, Q_st, om_st,
                jnp.asarray([Z]), jnp.asarray([nele]),
                grid, fermi_amaldi=True,
            ),
            dtype=np.float64,
        )[0]

        new_orbs = []
        energies = []
        for i, orb in enumerate(orbitals):
            P, dP = hydrogenic_p(r, orb["z_eff"], orb["n"], orb["l"])
            Q = kinetic_balance_q(P, dP, V_new, orb["kappa"], r)
            P, Q = normalize(P, Q, grid)
            E = orbital_energy(P, Q, V_new, orb["kappa"], grid)
            energies.append(E)
            new_orbs.append({**orb, "P": P, "Q": Q, "E_Ha": E})

        dv = float(np.max(np.abs(V_new - V)))
        history.append({"iter": it, "dV_max": dv, "E": energies})
        V, orbitals = V_new, new_orbs
        if dv < tol:
            break

    return {
        "r": r,
        "V": V,
        "V_nuc": -Z / np.maximum(r, 1e-8),
        "orbitals": orbitals,
        "bare_hydrogenic": bare,
        "z_eff": z_eff,
        "history": history,
    }


def plot_reference(ref: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r = ref["r"]
    V = ref["V"]
    V_nuc = ref["V_nuc"]

    # --- V(r): log and linear core ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax in axes:
        ax.plot(r, V_nuc, "k--", lw=1.0, alpha=0.45, label=r"$-Z/r$")
        ax.plot(r, V, "C0-", lw=2.0, label=r"$V_\mathrm{DFS}$")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(r"$r$ (Bohr)")
        ax.set_ylabel(r"$V(r)$ (Hartree)")
        ax.legend(fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_title(r"Full range (log $r$)")
    axes[1].set_xlim(0, 12)
    axes[1].set_ylim(-10, 1)
    axes[1].set_title(r"Valence region ($r \lesssim 12$)")
    fig.suptitle(r"Li ($Z=3$): DFS central potential from 1s²2s¹ density", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "V_z3_dfs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- P, Q: DFS vs bare-Z vs Z_eff hydrogenic ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {"1s": "C0", "2s": "C1"}
    bare_map = {b["label"]: b for b in ref["bare_hydrogenic"]}

    for orb in ref["orbitals"]:
        lab = orb["label"]
        c = colors[lab]
        row = 0 if lab == "1s" else 1
        P_bare, Q_bare = bare_map[lab]["P"], bare_map[lab]["Q"]
        P_zeff, dP = hydrogenic_p(r, orb["z_eff"], orb["n"], 0)
        Q_zeff = kinetic_balance_q(P_zeff, dP, V, orb["kappa"], r)

        axes[row, 0].semilogy(r, np.abs(P_bare) + 1e-30, "--", color="gray", alpha=0.6, label=rf"H($Z={3}$)")
        axes[row, 0].semilogy(r, np.abs(P_zeff) + 1e-30, ":", color=c, alpha=0.7, label=rf"H($Z_{{eff}}$={orb['z_eff']:.2f})")
        axes[row, 0].semilogy(r, np.abs(orb["P"]) + 1e-30, "-", color=c, lw=2, label=rf"DFS {lab}")
        axes[row, 0].set_title(rf"${lab}$: $|P(r)|$")
        axes[row, 0].set_xlabel(r"$r$")
        axes[row, 0].legend(fontsize=7)

        axes[row, 1].semilogy(r, np.abs(Q_bare) + 1e-30, "--", color="gray", alpha=0.6, label=rf"H($Z={3}$)")
        axes[row, 1].semilogy(r, np.abs(Q_zeff) + 1e-30, ":", color=c, alpha=0.7, label=rf"H($Z_{{eff}}$)")
        axes[row, 1].semilogy(r, np.abs(orb["Q"]) + 1e-30, "-", color=c, lw=2, label=rf"DFS {lab}")
        axes[row, 1].set_title(rf"${lab}$: $|Q(r)|$")
        axes[row, 1].set_xlabel(r"$r$")
        axes[row, 1].legend(fontsize=7)

    fig.suptitle(
        r"Li $l=0$: large $P=rR$ and small $Q$ (DFS = hydrogenic $P$ + kinetic balance $Q$ in $V_\mathrm{DFS}$)",
        y=1.01, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "PQ_z3_l0_reference.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Core linear scale (where DeepONet must resolve nodes) ---
    mask = r <= 8.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(r[mask], V[mask], "C0-", lw=2)
    axes[0].plot(r[mask], V_nuc[mask], "k--", alpha=0.35)
    axes[0].set_title(r"$V(r)$")
    for orb in ref["orbitals"]:
        axes[1].plot(r[mask], orb["P"][mask], lw=2, label=orb["label"])
        axes[2].plot(r[mask], orb["Q"][mask], lw=2, label=orb["label"])
    axes[1].set_title(r"$P(r)$ — note 2s radial node")
    axes[2].set_title(r"$Q(r)$")
    for ax in axes:
        ax.set_xlabel(r"$r$")
    axes[1].legend()
    axes[2].legend()
    fig.suptitle(r"Core / valence region ($r \leq 8$ Bohr)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "core_region_z3_l0.png", dpi=160)
    plt.close(fig)

    # --- Electron density ---
    rho = sum(o["omega"] * (o["P"] ** 2 + o["Q"] ** 2) for o in ref["orbitals"])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.semilogy(r, rho + 1e-35, "C2-", lw=2)
    ax.set_xlabel(r"$r$ (Bohr)")
    ax.set_ylabel(r"$\rho(r)=\sum_a \omega_a(P_a^2+Q_a^2)$")
    ax.set_title(r"Li radial density (sets Hartree screening in $V_\mathrm{DFS}$)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "density_z3.png", dpi=160)
    plt.close(fig)


def print_summary(ref: dict, grid) -> None:
    print("\n=== Z=3, l=0 reference shapes (Li 1s²2s¹) ===\n")
    print("Sources:")
    print("  • V(r)  — DFS: -Z/r + Hartree + Slater Xα (Fermi-Amaldi SIC, Latter tail)")
    print("  • P(r)  — analytic hydrogenic R_{n,l}(r; Z_eff) with P = r·R")
    print("  • Q(r)  — Dirac kinetic balance in V_dfs (same as DeepONet)")
    print("  • Energies — Dirac variational formula used in PINN-ART loss\n")

    print(f"{'orb':4s} {'Z_eff':>6s} {'E_orb (Ha)':>12s} {'E_orb (eV)':>11s} {'nodes':>6s} {'max|P|':>8s}")
    print("-" * 54)
    for orb in ref["orbitals"]:
        P = orb["P"]
        nodes = int(np.sum(P[: int(0.9 * len(P))][1:] * P[: int(0.9 * len(P))][:-1] < 0))
        print(
            f"{orb['label']:4s} {orb['z_eff']:6.3f} {orb['E_Ha']:12.6f} "
            f"{orb['E_Ha'] * HARTREE_TO_EV:11.2f} {nodes:6d} {np.max(np.abs(P)):8.4f}"
        )

    print(f"\nSCF-style iterations: {len(ref['history'])}, final max|ΔV| = {ref['history'][-1]['dV_max']:.2e}")
    r = ref["r"]
    V = ref["V"]
    print(f"V_dfs(r=0.1) = {V[np.argmin(np.abs(r-0.1))]:.3f} Ha  (screened vs -30 Ha bare)")
    print(f"V_dfs(r=5)   = {V[np.argmin(np.abs(r-5.0))]:.3f} Ha  (→ -1/r ion tail)")
    print(f"\nFigures: {OUT_DIR}/\n")


def main() -> None:
    grid = make_radial_grid(1.0e-4, 80.0, 4000, "loglinear")
    ref = build_reference(grid)
    plot_reference(ref)
    print_summary(ref, grid)


if __name__ == "__main__":
    main()
