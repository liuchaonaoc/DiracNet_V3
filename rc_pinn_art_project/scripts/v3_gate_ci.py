#!/usr/bin/env python3
"""Gate B — synthetic CI spectra + eigh grad + manifest E_csf consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

from pinn_art.ci.racah_cache import RacahCache
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params, merge_params
from pinn_art.training.gate_b import (
    run_eigh_gradient_gate,
    run_manifest_ci_gate,
    run_synthetic_gate,
)
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def main():
    ap = argparse.ArgumentParser(description="PINN-ART Gate B (CI)")
    ap.add_argument("--config", default="configs/v3_phase1_stage_b_z1_8.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--skip-manifest", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    gate_cfg = getattr(cfg.stage_b, "gate", cfg.stage_b)

    print("==> Gate B.1 synthetic 2x2 / 3x3 Hamiltonians")
    syn = run_synthetic_gate(float(getattr(gate_cfg, "synthetic_meV_threshold", 10.0)))
    for name, r in syn.items():
        if name == "verdict":
            continue
        print(f"  {name}: {'PASS' if r['pass'] else 'FAIL'}  err_meV={r['err_meV']:.3f}  E_pred={r['E_pred']}")
    print(f"  synthetic VERDICT={syn['verdict']}")

    print("\n==> Gate B.2 eigh gradient finite")
    grad_gate = run_eigh_gradient_gate()
    print(f"  grad_finite: {'PASS' if grad_gate['pass'] else 'FAIL'}  trials={grad_gate['n_trials']}")

    manifest_verdict = "SKIP"
    manifest_report = None
    if not args.skip_manifest and args.ckpt:
        print("\n==> Gate B.3 manifest CI consistency (|E_csf - E_orb|)")
        manifest = ROOT / cfg.dataset.manifest
        racah_cache = RacahCache.load(ROOT / cfg.ci.racah_cache)
        k_list = tuple(int(k) for k in cfg.ci.k_list)
        ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
        grid = make_radial_grid(
            float(cfg.grid.r_min),
            float(cfg.grid.r_max),
            int(cfg.grid.n_grid),
            str(cfg.grid.scheme),
        )
        model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
        loaded = load_params(ROOT / args.ckpt)
        params = merge_params(params, loaded)

        def collate_fn(items):
            return collate_batches(
                items,
                n_csf_max=int(cfg.model.n_csf_max),
                racah_cache=racah_cache,
                k_list=k_list,
            )

        apply_fn = lambda p, b, g, **kw: model.apply(p, b, g, **kw)
        manifest_report = run_manifest_ci_gate(
            apply_fn,
            params,
            ds,
            grid,
            collate_fn,
            meV_threshold=float(getattr(gate_cfg, "manifest_dE_meV_threshold", 50.0)),
        )
        for r in manifest_report["rows"][:8]:
            print(
                f"  row {r['idx']:2d} Z={r['Z']} ({r['element']}) n={r['n']}: "
                f"{'PASS' if r['pass'] else 'FAIL'}  dE_meV={r['dE_meV']:.2f}"
            )
        if len(manifest_report["rows"]) > 8:
            print(f"  ... ({len(manifest_report['rows'])} rows total)")
        print(
            f"  manifest SUMMARY: {manifest_report['n_pass']}/{manifest_report['n_total']}  "
            f"VERDICT={manifest_report['verdict']}"
        )
        manifest_verdict = manifest_report["verdict"]

    overall = "PASS"
    if syn["verdict"] != "PASS" or not grad_gate["pass"]:
        overall = "FAIL"
    if manifest_verdict == "FAIL":
        overall = "FAIL"

    report = {
        "synthetic": syn,
        "eigh_gradient": grad_gate,
        "manifest": manifest_report,
        "verdict": overall,
    }
    log_dir = ROOT / cfg.training.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "gate_b_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nOVERALL VERDICT={overall}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
