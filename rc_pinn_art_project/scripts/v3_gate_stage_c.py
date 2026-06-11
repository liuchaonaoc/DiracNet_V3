#!/usr/bin/env python3
"""Gate C — Stage C eval + fallback smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_art.ci.racah_cache import RacahCache
from pinn_art.constants import hartree_to_meV, meV_to_hartree
from pinn_art.data.collate import collate_batches
from pinn_art.data.config_parser import encode_config_to_array, parse_config_string
from pinn_art.data.dataset import ManifestDataset
from pinn_art.evaluation.stage_c_eval import run_stage_c_evaluation
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.hydrogenic import hydrogenic_energy
from pinn_art.training.checkpoint import load_params, merge_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def run_fallback_smoke(cfg, params, model, grid, racah_cache, k_list) -> dict:
    Z, n = 3, 4
    E_n_ha = hydrogenic_energy(Z, n)
    E_1_ha = hydrogenic_energy(Z, 1)
    shells = parse_config_string("4s1")
    n_orb = int(cfg.model.n_orb_max)
    kappas, omegas, orb_mask = [], [], []
    for i in range(n_orb):
        if i < len(shells):
            kappas.append(shells[i].kappa)
            omegas.append(float(shells[i].occ))
            orb_mask.append(True)
        else:
            kappas.append(0)
            omegas.append(0.0)
            orb_mask.append(False)

    item = {
        "Z": Z,
        "ion_charge": 2,
        "nele": 1,
        "parent_config": "1s1",
        "level_config": "4s1",
        "shell_table": encode_config_to_array(shells, n_orb),
        "kappa": np.array(kappas, dtype=np.int32),
        "omega": np.array(omegas, dtype=np.float32),
        "orb_mask": np.array(orb_mask, dtype=bool),
        "E_nist_scalar": np.nan,
        "nist_mask_scalar": False,
        "csf_slot": 0,
    }
    batch = collate_batches(
        [item], n_csf_max=int(cfg.model.n_csf_max), racah_cache=racah_cache, k_list=k_list
    )
    out = model.apply(params, batch, grid, train=False, return_ci=True)
    d_e = float(hartree_to_meV(abs(out["E_csf"][0, 0] - out["E_orb"][0, 0])))
    diag = int(out["diag_source"][0, 0])
    ok = diag == 0 and d_e < 50.0 and not bool(batch["nist_mask"][0, 0])
    return {"pass": ok, "diag_source": diag, "dE_meV": d_e}


def main():
    ap = argparse.ArgumentParser(description="Gate C")
    ap.add_argument("--config", default="configs/v3_phase1_stage_c_z1_8.yaml")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    ckpt = ROOT / (args.ckpt or cfg.stage_c.ckpt)
    manifest = ROOT / cfg.dataset.manifest
    log_dir = ROOT / cfg.training.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics, _, _ = run_stage_c_evaluation(cfg, ckpt, manifest, log_dir, root=ROOT)

    racah_cache = RacahCache.load(ROOT / cfg.ci.racah_cache)
    k_list = tuple(int(k) for k in cfg.ci.k_list)
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max), int(cfg.grid.n_grid), str(cfg.grid.scheme)
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(1))
    params = merge_params(params, load_params(ckpt))
    fallback = run_fallback_smoke(cfg, params, model, grid, racah_cache, k_list)

    overall = "PASS"
    if metrics["layer2_inject"]["verdict"] != "PASS" or not fallback["pass"]:
        overall = "FAIL"
    # layer2_orb is informational (E_orb vs NIST); not gating in Phase 2a

    report = {
        "layer2_orb": metrics["layer2_orb"],
        "layer2_inject": metrics["layer2_inject"],
        "fallback_smoke": fallback,
        "verdict": overall,
    }
    out_path = log_dir / "gate_c_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"layer2_orb: {metrics['layer2_orb']['verdict']}  MAE={metrics['layer2_orb']['mae_exc_orb_vs_nist_meV']:.2f} meV")
    print(f"layer2_inject: {metrics['layer2_inject']['verdict']}")
    print(f"fallback_smoke: {'PASS' if fallback['pass'] else 'FAIL'}")
    print(f"OVERALL VERDICT={overall}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
