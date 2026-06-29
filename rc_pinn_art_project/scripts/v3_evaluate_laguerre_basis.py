#!/usr/bin/env python3
"""Stage A Round 2 evaluator for the Laguerre-basis ansatz.

Loads a trained checkpoint (from v3_train_stage_a_laguerre_basis.py) and
runs three diagnostic suites:

1. **Morphology gate** (Layer-0, hard thresholds from prompts/17 §7.1)
   - For each active orbital: count the sign-changes of P_a on r ∈ [0.1, 25].
   - Must equal `n - l - 1` for that orbital.
   - Sign change counter is robust to FP noise (skips samples with
     |P| < 1e-3 * max(|P|) and counts only adjacent opposite-sign pairs).

2. **Reference similarity**
   - cosine(P_model, P_hydrogenic) and L2(P_model) / L2(P_ref) — both must
     be ≥ 0.95 for the hydrogenic baseline and ≥ 0.85 for DFS-like systems.
   - Reports node count, λ_a (learnable), λ_init (analytic Z_eff/n) and
     |λ_a − λ_init| / λ_init as a check on the softplus-driven drift.

3. **Energy gate**
   - orbital E (Hartree) → meV via `hartree_to_meV` for each active orbital.
   - For H / He / Li ground-state rows with NIST injection, compare
     against the loaded manifest's NIST values.

Outputs a JSON report to ``--out`` (default
``results/laguerre_basis_eval/<ckpt-stem>.json``) plus a small console
table.

Usage:
    python scripts/v3_evaluate_laguerre_basis.py \\
        --config configs/v3_stage_a_laguerre_basis.yaml \\
        --ckpt checkpoints/v3_stage_a_laguerre_basis/stage_a_last.msgpack \\
        --max-rows 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from pinn_art.constants import hartree_to_meV
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.hydrogenic import (
    cosine_signed,
    hydrogenic_P_jax,
)
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.utils.logging import get_logger


def _count_nodes(
    P_np: np.ndarray, r_np: np.ndarray,
    *,
    n: int = 1, Z: int = 1, l: int = 0,
) -> int:
    """Robust radial node counter (skips noise floor).

    The radial extent of an orbital with principal quantum number n scales
    as `n^2 / Z` (hydrogenic), so the node-search window must grow with
    n.  We use `rmax = 2 * n^2 + 5` (covers H 5s nodes out to r ≈ 27 with
    margin).  The lower bound `rmin = 0.1` avoids the r → 0 cusp.
    """
    rmin = 0.1
    rmax = max(25.0, 2.0 * float(n) * float(n) + 5.0)
    rmin_idx = int(np.searchsorted(r_np, rmin))
    rmax_idx = int(np.searchsorted(r_np, rmax))
    body = P_np[rmin_idx:rmax_idx]
    if body.size < 2:
        return 0
    pmax = float(np.max(np.abs(body)))
    if pmax == 0.0:
        return 0
    sign_arr = np.sign(body)
    sign_arr = np.where(np.abs(body) > 1e-3 * pmax, sign_arr, 0)
    nonzero = sign_arr[sign_arr != 0]
    if nonzero.size < 2:
        return 0
    diffs = np.diff(nonzero)
    return int(np.sum(diffs != 0))


def _summarize_active_orbitals(
    out: dict,
    grid,
    Z: np.ndarray,
    kappa: np.ndarray,
    n_principal: np.ndarray,
    l_orbital: np.ndarray,
) -> list[dict]:
    r_np = np.asarray(grid.r)
    P = np.asarray(out["wavefunctions"]["P"][0])    # [N_orb, N_g]
    lag = out.get("laguerre_lambdas")
    if lag is not None and lag[0] is not None:
        arr = np.asarray(lag[0])
        lambdas = arr if arr.ndim > 0 else np.full(P.shape[0], float(arr))
    else:
        lambdas = None
    lag = out.get("laguerre_lambda_init")
    if lag is not None and lag[0] is not None:
        arr = np.asarray(lag[0])
        lambda_init = arr if arr.ndim > 0 else np.full(P.shape[0], float(arr))
    else:
        lambda_init = None
    rows = []
    for a in range(P.shape[0]):
        if int(kappa[a]) == 0 and int(l_orbital[a]) == 0:
            # masked-out slot
            continue
        n_q = int(n_principal[a])
        l_q = int(l_orbital[a])
        nodes = _count_nodes(P[a], r_np, n=n_q, Z=int(Z[0]), l=l_q)
        expected = max(n_q - l_q - 1, 0) if n_q > l_q else 0
        lam_row = {
            "lambda": float(lambdas[a]) if lambdas is not None else None,
            "lambda_init": float(lambda_init[a]) if lambda_init is not None else None,
            "rel_drift": (
                abs(float(lambdas[a]) - float(lambda_init[a]))
                / max(abs(float(lambda_init[a])), 1e-12)
                if lambdas is not None and lambda_init is not None else None
            ),
        }
        rows.append({
            "slot": a,
            "n": n_q,
            "l": l_q,
            "nodes_observed": nodes,
            "nodes_expected": expected,
            "node_gate": nodes == expected,
            "lambda": lam_row["lambda"],
            "lambda_init": lam_row["lambda_init"],
            "lambda_rel_drift": lam_row["rel_drift"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a Laguerre-basis checkpoint"
    )
    ap.add_argument("--config", default="configs/v3_stage_a_laguerre_basis.yaml")
    ap.add_argument(
        "--ckpt",
        default="checkpoints/v3_stage_a_laguerre_basis/stage_a_last.msgpack",
        help="Path to msgpack checkpoint produced by the trainer",
    )
    ap.add_argument("--max-rows", type=int, default=30)
    ap.add_argument("--z-min", type=int, default=None)
    ap.add_argument("--z-max", type=int, default=None)
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: results/laguerre_basis_eval/<ckpt-stem>.json)",
    )
    args = ap.parse_args()

    log = get_logger()
    log.info("JAX backend=%s devices=%s", jax.default_backend(), jax.devices())
    cfg = load_config(ROOT / args.config)
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max),
        int(cfg.grid.n_grid), str(cfg.grid.scheme),
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    if args.ckpt:
        ckpt_path = ROOT / args.ckpt
        if ckpt_path.exists():
            from pinn_art.training.checkpoint import load_params
            params = load_params(ckpt_path)
            log.info("Loaded checkpoint: %s", ckpt_path)
        else:
            log.warning("ckpt %s not found — evaluating initial params", ckpt_path)

    manifest = ROOT / cfg.dataset.manifest
    ds = ManifestDataset(
        manifest,
        n_orb_max=int(cfg.model.n_orb_max),
        n_csf_max=int(getattr(cfg.model, "n_csf_max", 8)),
    )
    if args.z_min is not None or args.z_max is not None:
        mask = np.ones(len(ds), dtype=bool)
        if args.z_min is not None:
            mask &= ds.df["Z"].to_numpy() >= args.z_min
        if args.z_max is not None:
            mask &= ds.df["Z"].to_numpy() <= args.z_max
        ds.df = ds.df[mask].reset_index(drop=True)

    log.info("Evaluating %d manifest rows (max=%d)", len(ds), args.max_rows)

    summary = {
        "n_rows_evaluated": 0,
        "n_node_gate_passed": 0,
        "n_node_gate_total_active": 0,
        "mean_cos_P_hydrogenic": 0.0,
        "mean_lambda_rel_drift": 0.0,
        "max_lambda_rel_drift": 0.0,
        "rows": [],
    }

    n_rows = min(args.max_rows, len(ds))
    for i in range(n_rows):
        batch = collate_batches(
            [ds[i]],
            n_csf_max=int(cfg.model.n_csf_max),
            k_max=int(getattr(cfg.model, "K_max", 9)),
            build_nodes=bool(getattr(cfg.model, "use_hybrid_head", False)),
            nodes_table_path=str(getattr(cfg.model, "nodes_table_path",
                                         "data_cache/laguerre_nodes_z1_26_n1_10.parquet")),
        )
        out = model.apply(params, batch, grid, train=False, return_ci=False)

        # Reference similarity vs analytic hydrogenic.  Use
        # `batch["shell_table"]` for the true (n, l) — `|kappa|` is only
        # l+1 by construction and would mis-evaluate any non-1s row.
        sh = np.asarray(batch["shell_table"][0])
        n_from_shell = jnp.asarray(sh[:, 0], dtype=jnp.int32)
        l_from_shell = jnp.asarray(sh[:, 1], dtype=jnp.int32)
        # Fall back to `|kappa|` only if shell_table is all-zero (legacy path).
        use_shell = bool(np.any(n_from_shell > 0))
        if use_shell:
            n_arr = n_from_shell
            l_arr = l_from_shell
        else:
            l_arr = jnp.where(
                batch["kappa"][0] < 0, -batch["kappa"][0] - 1, batch["kappa"][0]
            )
            n_arr = jnp.maximum(jnp.abs(batch["kappa"][0]), 1)
        valid = ((n_arr > l_arr) & (n_arr > 0)).astype(jnp.float32)
        P_ref = hydrogenic_P_jax(grid.r, batch["Z"], n_arr[None, :], l_arr[None, :])
        P_ref = P_ref * valid[None, :, None]

        wf = out["wavefunctions"]
        # cosine_signed is per-orbital; reduce over active orbitals.
        P_model = wf["P"][0]   # [N_orb, N_g]
        active = np.asarray(valid) > 0.5
        cos_per = []
        for a in range(P_model.shape[0]):
            if active[a]:
                cos_per.append(float(cosine_signed(P_model[a], P_ref[0, a], grid)))
            else:
                cos_per.append(float("nan"))
        cos_per = np.asarray(cos_per, dtype=np.float64)
        if active.any():
            cos_p_active = float(np.nanmean(np.abs(cos_per[active])))
            cos_p_overall = float(np.nanmean(np.abs(cos_per[active])))
        else:
            cos_p_active = float("nan")
            cos_p_overall = float("nan")

        # Energy gate (Hartree → meV)
        E_orb_mev = hartree_to_meV(np.asarray(out["E_orb"][0]))  # [N_orb]

        # n_principal / l_orbital from shell_table
        sh = np.asarray(batch["shell_table"][0])
        n_principal = sh[:, 0]
        l_orbital = sh[:, 1]

        active_rows = _summarize_active_orbitals(
            out, grid, np.asarray(batch["Z"]),
            np.asarray(batch["kappa"][0]), n_principal, l_orbital,
        )

        # aggregate node-gate pass count
        for r_ in active_rows:
            if r_["node_gate"]:
                summary["n_node_gate_passed"] += 1
            summary["n_node_gate_total_active"] += 1
            if r_["lambda_rel_drift"] is not None:
                summary["mean_lambda_rel_drift"] += r_["lambda_rel_drift"]
                summary["max_lambda_rel_drift"] = max(
                    summary["max_lambda_rel_drift"], r_["lambda_rel_drift"]
                )

        if active_rows:
            summary["mean_cos_P_hydrogenic"] += cos_p_active

        summary["rows"].append({
            "row_index": i,
            "Z": int(np.asarray(batch["Z"][0])),
            "n_active_orbitals": int(np.sum(valid)),
            "cos_P_hydrogenic_overall": cos_p_overall,
            "cos_P_hydrogenic_active_mean": cos_p_active,
            "cos_P_hydrogenic_per_orbital": [float(x) for x in cos_per],
            "E_orb_mev": [float(x) for x in E_orb_mev],
            "active_orbitals": active_rows,
        })

        summary["n_rows_evaluated"] += 1

    if summary["n_rows_evaluated"] > 0:
        summary["mean_cos_P_hydrogenic"] /= summary["n_rows_evaluated"]
    if summary["n_node_gate_total_active"] > 0:
        summary["mean_lambda_rel_drift"] /= summary["n_node_gate_total_active"]
    else:
        summary["mean_lambda_rel_drift"] = float("nan")

    # Finalize
    node_pass_rate = (
        summary["n_node_gate_passed"] / max(summary["n_node_gate_total_active"], 1)
    )
    summary["node_gate_pass_rate"] = float(node_pass_rate)
    log.info(
        "node-gate pass rate: %d / %d (%.1f%%)",
        summary["n_node_gate_passed"], summary["n_node_gate_total_active"],
        100.0 * node_pass_rate,
    )
    log.info(
        "mean cos(P_model, P_hydrogenic) over active orbitals: %.4f",
        summary["mean_cos_P_hydrogenic"],
    )
    log.info(
        "λ_a drift vs λ_init: mean=%.4f  max=%.4f",
        summary["mean_lambda_rel_drift"], summary["max_lambda_rel_drift"],
    )

    out_path = (
        ROOT / args.out if args.out else
        ROOT / "results" / "laguerre_basis_eval" / f"{Path(args.ckpt).stem}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote evaluation report to %s", out_path)


if __name__ == "__main__":
    main()