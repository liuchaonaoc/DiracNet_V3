#!/usr/bin/env python3
"""Diagnose why Stage A loss is not decreasing.

Checks:
  1. Exact H 1s solution: feed it into dirac_apply → residual should be ~0
  2. Model forward: report scale of each loss term and weighted contributions
  3. Verify dQdr is actually computed (vs hard-coded 0)
  4. Verify n quantum number is encoded into envelope (vs only kappa)
  5. Verify Lowdin pre-orthogonalization kills ortho_loss
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from pinn_art.constants import C_LIGHT
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.losses.asymptotic_loss import asymptotic_tail_loss
from pinn_art.losses.norm_loss import normalization_loss
from pinn_art.losses.ortho_loss import orthonormality_loss
from pinn_art.losses.pde_loss import dirac_pde_loss
from pinn_art.losses.potential_prior import (
    potential_prior_loss,
    potential_smooth_loss,
)
from pinn_art.losses.loss_schedule import stage_a_weights
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def section(t: str) -> None:
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


def hydrogen_1s_exact(r: jnp.ndarray, Z: float = 1.0):
    """P_1s = 2 Z^(3/2) r exp(-Z r),  Q from correct Dirac kinetic balance."""
    P = 2.0 * (Z ** 1.5) * r * jnp.exp(-Z * r)
    dPdr = 2.0 * (Z ** 1.5) * jnp.exp(-Z * r) * (1.0 - Z * r)
    c = C_LIGHT
    kap_over_r = -1.0 / jnp.clip(r, 1e-8)  # kappa = -1
    V = -Z / jnp.clip(r, 1e-8)
    # Correct sign: (E + 2c^2 - V) Q = c (dP + kappa/r P)  →  denom = 2c^2 - V (positive)
    denom = jnp.clip(2.0 * c * c - V, 1e-6, 1e10)
    Q = (c * (dPdr + kap_over_r * P)) / denom
    dQdr = jnp.gradient(Q, r)
    return P, Q, dPdr, dQdr, V


def main() -> None:
    cfg = load_config(ROOT / "configs" / "v3_phase1_stage_a_z1_8.yaml")
    grid = make_radial_grid(
        float(cfg.grid.r_min),
        float(cfg.grid.r_max),
        int(cfg.grid.n_grid),
        str(cfg.grid.scheme),
    )
    r = grid.r

    # ----------------------------------------------------------------------
    section("[1] Exact H 1s -> dirac_apply residual (sanity)")
    P, Q, dPdr, dQdr, V = hydrogen_1s_exact(r, Z=1.0)
    norm_sq = float(grid.integrate(P * P + Q * Q))
    print(f"  ⟨P²+Q²⟩ = {norm_sq:.4f}  (should be ≈ 1.0)")

    # broadcast for dirac_apply: [B,N_orb,N_g]
    P_b = P[None, None, :]
    Q_b = Q[None, None, :]
    dP_b = dPdr[None, None, :]
    dQ_b = dQdr[None, None, :]
    kappa = jnp.array([[-1]])
    V_b = V[None, :]
    LP, LQ = dirac_apply(P_b, Q_b, dP_b, dQ_b, V_b, kappa, r)
    E = orbital_energy_from_dirac(P_b, Q_b, LP, LQ, grid)
    print(f"  E_dirac  = {float(E[0,0]):.6f} Ha  (expected ≈ -0.500)")

    res_P = LP[0, 0] - float(E[0, 0]) * P
    res_Q = LQ[0, 0] - float(E[0, 0]) * Q
    pde_exact = float(grid.integrate(res_P ** 2 + res_Q ** 2))
    print(f"  PDE residual ‖(H-E)ψ‖² = {pde_exact:.3e}  (should be small)")

    # Now zero out dQdr (== current network) to see how much it hurts
    LP_bad, LQ_bad = dirac_apply(P_b, Q_b, dP_b, jnp.zeros_like(dQ_b), V_b, kappa, r)
    E_bad = orbital_energy_from_dirac(P_b, Q_b, LP_bad, LQ_bad, grid)
    res_P_bad = LP_bad[0, 0] - float(E_bad[0, 0]) * P
    res_Q_bad = LQ_bad[0, 0] - float(E_bad[0, 0]) * Q
    pde_zerodQ = float(grid.integrate(res_P_bad ** 2 + res_Q_bad ** 2))
    print(f"  PDE residual with dQ/dr := 0   = {pde_zerodQ:.3e}  ← network's actual behavior")
    print(f"  E (with dQ/dr := 0)            = {float(E_bad[0,0]):.6f} Ha")

    # ----------------------------------------------------------------------
    section("[2] Model forward on H 1s, all loss terms")
    ds = ManifestDataset(
        ROOT / cfg.dataset.manifest, n_orb_max=int(cfg.model.n_orb_max),
        n_csf_max=int(cfg.model.n_csf_max),
    )
    batch = collate_batches([ds[0]], n_csf_max=int(cfg.model.n_csf_max))
    key = jax.random.PRNGKey(int(cfg.seed))
    model, params = build_model_and_params(cfg, grid, key)
    out = model.apply(params, batch, grid, train=False, return_ci=False)
    P_n = out["wavefunctions"]["P"]
    Q_n = out["wavefunctions"]["Q"]
    dP_n = out["wavefunctions"]["dPdr"]
    dQ_n = out["wavefunctions"]["dQdr"]
    V_n = out["V"]

    print(f"  P shape={P_n.shape}, ‖P‖∞={float(jnp.abs(P_n).max()):.3e}")
    print(f"  Q shape={Q_n.shape}, ‖Q‖∞={float(jnp.abs(Q_n).max()):.3e}")
    print(f"  dQ/dr ‖.‖∞ = {float(jnp.abs(dQ_n).max()):.3e}  "
          f"(if 0.000e+00 → dQdr hard-coded to 0)")
    print(f"  V  range = [{float(V_n.min()):.3f}, {float(V_n.max()):.3f}]")

    w = stage_a_weights(cfg)
    Z_arr = batch["Z"]
    orb_mask = batch["orb_mask"][:, : int(cfg.model.n_orb_max)]
    l_pde = float(dirac_pde_loss(
        P_n, Q_n, dP_n, dQ_n, V_n, batch["kappa"][:, : int(cfg.model.n_orb_max)],
        r, grid, orb_mask, E_orb=out["E_orb"],
    ))
    l_ortho = float(orthonormality_loss(P_n, Q_n, grid, orb_mask))
    l_asym = float(asymptotic_tail_loss(P_n, Q_n, r, orb_mask, Z=Z_arr))
    l_norm = float(normalization_loss(P_n, Q_n, grid, orb_mask))
    l_vp = float(potential_prior_loss(V_n, Z_arr, r))
    l_vs = float(potential_smooth_loss(V_n, r))

    table = [
        ("pde",      l_pde,   w["pde"]),
        ("ortho",    l_ortho, w["ortho"]),
        ("asym",     l_asym,  w["asym"]),
        ("norm",     l_norm,  w["norm"]),
        ("v_prior",  l_vp,    w["v_prior"]),
        ("v_smooth", l_vs,    w["v_smooth"]),
    ]
    total = sum(raw * wt for _, raw, wt in table)
    print()
    print(f"  {'term':<10} {'raw':>14} {'weight':>10} {'weighted':>14} {'%total':>8}")
    print(f"  {'-'*10:<10} {'-'*14:>14} {'-'*10:>10} {'-'*14:>14} {'-'*8:>8}")
    for name, raw, wt in table:
        weighted = raw * wt
        pct = 100.0 * weighted / max(total, 1e-12)
        print(f"  {name:<10} {raw:>14.4e} {wt:>10.2e} {weighted:>14.4e} {pct:>7.1f}%")
    print(f"  {'TOTAL':<10} {'':>14} {'':>10} {total:>14.4e}")

    # ----------------------------------------------------------------------
    section("[3] Network response: how much can it adjust?")
    # 1s and 2s are both kappa=-1, only n_idx differs in branch_feat
    batch_1s = collate_batches([ds[0]], n_csf_max=int(cfg.model.n_csf_max))   # n=1
    batch_6s = collate_batches([ds[5]], n_csf_max=int(cfg.model.n_csf_max))   # n=6
    P1 = model.apply(params, batch_1s, grid, train=False, return_ci=False)["wavefunctions"]["P"][0, 0]
    P6 = model.apply(params, batch_6s, grid, train=False, return_ci=False)["wavefunctions"]["P"][0, 0]
    rel = float(jnp.linalg.norm(P1 - P6) / jnp.linalg.norm(P1))
    print(f"  ‖P(H,1s) - P(H,6s)‖ / ‖P(H,1s)‖ = {rel:.4f}")
    print(f"  → if ≈ 0, envelope is the same for 1s..6s (n quantum number ignored)")
    # Count internal nodes
    def n_nodes(P):
        sign_changes = int(jnp.sum(jnp.diff(jnp.sign(P + 1e-30)) != 0))
        return sign_changes
    print(f"  nodes in P(1s)={n_nodes(P1)}  P(6s)={n_nodes(P6)}  "
          f"(true H: 1s has 0, 6s has 5)")

    # ----------------------------------------------------------------------
    section("[4] Diagnosis summary")
    issues = []
    if float(jnp.abs(dQ_n).max()) < 1e-10:
        issues.append("dQ/dr is hard-coded to 0 in deeponet.py → kills LP residual")
    if rel < 0.01:
        issues.append("P(1s) ≈ P(6s): network does NOT encode principal n")
    if l_ortho < 1e-10 and orb_mask.sum() <= 1:
        issues.append("ortho_loss ≡ 0 because only 1 orbital per sample (Z=1..8 hydrogenic)")
    if l_norm < 1e-10:
        issues.append("norm_loss ≡ 0 because Lowdin normalizes P,Q every forward")
    if l_vs * w["v_smooth"] > 1000:
        issues.append("v_smooth dominates loss — re-scale or lower weight")
    if pde_zerodQ > 100 * pde_exact and pde_exact < 1:
        issues.append(f"Setting dQ/dr=0 inflates PDE residual by ~{pde_zerodQ/max(pde_exact,1e-12):.0e}×")

    if not issues:
        print("  No obvious structural issues found.")
    else:
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")

    print()
    print("Recommended fixes (see chat).")


if __name__ == "__main__":
    main()
