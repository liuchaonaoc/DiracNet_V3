#!/usr/bin/env python3
"""Run Gate A on manifest rows (per-Z 1s samples + summary)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import pandas as pd

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params
from pinn_art.training.gate_a import (
    check_gate_a,
    parse_n_l_from_level_config,
    resolve_gate_thresholds,
)
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_phase1_stage_a_z1_8.yaml")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows (default: all)")
    ap.add_argument(
        "--gate-mode",
        default="default",
        choices=["default", "relaxed", "mid", "strict"],
        help="Gate tier: default/relaxed=stage_a.gate, mid=gate_mid, strict=gate_strict",
    )
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config)
    manifest = Path(args.manifest) if args.manifest else ROOT / cfg.dataset.manifest
    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max), int(cfg.grid.n_grid), str(cfg.grid.scheme)
    )
    key = jax.random.PRNGKey(0)
    model, params = build_model_and_params(cfg, grid, key)
    if args.ckpt:
        params = load_params(ROOT / args.ckpt)
        print(f"Loaded checkpoint: {args.ckpt}")

    th = resolve_gate_thresholds(cfg.stage_a, args.gate_mode)
    print(
        f"Gate mode={args.gate_mode}: cos>={th['cos_threshold']}  "
        f"pde<={th['pde_threshold']}  dE<={th['e_orb_meV_threshold']} meV"
    )
    n_test = len(ds) if args.max_rows is None else min(args.max_rows, len(ds))

    reports = []
    rows_detail = []
    df = pd.read_parquet(manifest)
    for i in range(n_test):
        batch = collate_batches([ds[i]], n_csf_max=int(getattr(cfg.model, "n_csf_max", 8)))
        out = model.apply(params, batch, grid, train=False, return_ci=False)
        level_cfg = str(df.iloc[i].get("level_config", "1s1"))
        n, l = parse_n_l_from_level_config(level_cfg)
        rep = check_gate_a(
            out, batch, grid,
            cos_threshold=th["cos_threshold"],
            pde_threshold=th["pde_threshold"],
            e_orb_meV_threshold=th["e_orb_meV_threshold"],
            level_config=level_cfg,
        )
        reports.append(rep)
        Z = int(df.iloc[i]["Z"])
        el = df.iloc[i].get("element", f"Z{Z}")
        rows_detail.append({
            "row": i, "Z": Z, "element": str(el), "n": n, "l": l,
            "level_config": level_cfg,
            "verdict": rep.verdict,
            "cos": rep.cos_min, "pde": rep.pde_max, "dE_meV": rep.e_orb_mae_meV,
        })
        print(
            f"row {i:2d} Z={Z:2d} ({el}) n={n} l={l}: {rep.verdict}  "
            f"cos={rep.cos_min:.4f}  pde={rep.pde_max:.3e}  dE_meV={rep.e_orb_mae_meV:.2f}"
        )

    n_pass = sum(1 for r in reports if r.verdict == "PASS")
    verdict = "PASS" if n_pass == len(reports) else "FAIL"
    print(f"\nSUMMARY: {n_pass}/{len(reports)} passed  VERDICT={verdict}")

    log_dir = ROOT / cfg.training.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    out_json = log_dir / "gate_a_report.json"
    with out_json.open("w") as f:
        json.dump(
            {
                "verdict": verdict,
                "n_pass": n_pass,
                "n_total": len(reports),
                "gate_mode": args.gate_mode,
                "thresholds": th,
                "rows": rows_detail,
            },
            f,
            indent=2,
        )
    pd.DataFrame(rows_detail).to_csv(log_dir / "gate_a_detailed.csv", index=False)
    print(f"Report written: {out_json}")
    print(f"CSV written:    {log_dir / 'gate_a_detailed.csv'}")


if __name__ == "__main__":
    main()
