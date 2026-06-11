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
    ap.add_argument("--group", default=None, choices=["single_valence", "multi_electron"],
                    help="Filter manifest rows by group (Round 2)")
    ap.add_argument("--sample-per-parent", type=int, default=None,
                    help="Sample up to N rows per (Z,ion,parent_config) incl. ground")
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

    df = pd.read_parquet(manifest)
    if args.group and "group" in df.columns:
        df = df[df["group"] == args.group].copy()
        print(f"Filtered to group={args.group}: {len(df)} rows")

    if args.sample_per_parent is not None and "parent_config" in df.columns:
        parts = []
        for _, grp in df.groupby(["Z", "ion_charge", "parent_config"], sort=False):
            g = grp.sort_values("is_ground", ascending=False) if "is_ground" in grp.columns else grp
            parts.append(g.head(args.sample_per_parent))
        df = pd.concat(parts, ignore_index=True)
        print(f"Sampled {args.sample_per_parent}/parent: {len(df)} rows")

    # Map manifest rows -> ManifestDataset indices
    ds_index = {
        (int(r["Z"]), int(r["ion_charge"]), str(r["level_config"])): i
        for i, r in ds.df.iterrows()
    }
    test_rows = []
    for _, row in df.iterrows():
        key = (int(row["Z"]), int(row["ion_charge"]), str(row["level_config"]))
        if key in ds_index:
            test_rows.append((ds_index[key], row))
    if args.max_rows is not None:
        test_rows = test_rows[: args.max_rows]
    n_test = len(test_rows)

    reports = []
    rows_detail = []
    for ds_idx, row in test_rows:
        batch = collate_batches([ds[ds_idx]], n_csf_max=int(getattr(cfg.model, "n_csf_max", 8)))
        out = model.apply(params, batch, grid, train=False, return_ci=False)
        level_cfg = str(row.get("level_config", "1s1"))
        n, l = parse_n_l_from_level_config(level_cfg)
        rep = check_gate_a(
            out, batch, grid,
            cos_threshold=th["cos_threshold"],
            pde_threshold=th["pde_threshold"],
            e_orb_meV_threshold=th["e_orb_meV_threshold"],
            level_config=level_cfg,
        )
        reports.append(rep)
        Z = int(row["Z"])
        el = row.get("element", f"Z{Z}")
        rows_detail.append({
            "row": ds_idx, "Z": Z, "element": str(el), "n": n, "l": l,
            "level_config": level_cfg,
            "group": str(row.get("group", "")),
            "is_ground": bool(row.get("is_ground", False)),
            "verdict": rep.verdict,
            "cos": rep.cos_min, "pde": rep.pde_max, "dE_meV": rep.e_orb_mae_meV,
        })
        print(
            f"row {ds_idx:4d} Z={Z:2d} ({el}) n={n} l={l}: {rep.verdict}  "
            f"cos={rep.cos_min:.4f}  pde={rep.pde_max:.3e}  dE_meV={rep.e_orb_mae_meV:.2f}"
        )

    n_pass = sum(1 for r in reports if r.verdict == "PASS")
    verdict = "PASS" if n_pass == len(reports) else "FAIL"
    print(f"\nSUMMARY: {n_pass}/{len(reports)} passed  VERDICT={verdict}")

    detail_df = pd.DataFrame(rows_detail)
    group_summary = {}
    if "group" in detail_df.columns and detail_df["group"].str.len().gt(0).any():
        for g, gdf in detail_df.groupby("group"):
            if not g:
                continue
            gp = int((gdf["verdict"] == "PASS").sum())
            group_summary[g] = {"n": len(gdf), "n_pass": gp, "pass_fraction": gp / max(len(gdf), 1)}

    log_dir = ROOT / cfg.training.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    out_json = log_dir / "gate_a_report.json"
    report_payload = {
        "verdict": verdict,
        "n_pass": n_pass,
        "n_total": len(reports),
        "gate_mode": args.gate_mode,
        "thresholds": th,
        "group_summary": group_summary,
        "rows": rows_detail,
    }
    with out_json.open("w") as f:
        json.dump(report_payload, f, indent=2)
    detail_df.to_csv(log_dir / "gate_a_detailed.csv", index=False)

    md_lines = [
        "# Gate A 报告（Stage A Round 2）",
        "",
        f"- Checkpoint: `{args.ckpt or 'init'}`",
        f"- Config: `{args.config}`",
        f"- Manifest: `{manifest.name}`",
        f"- Gate mode: `{args.gate_mode}`",
        f"- 阈值: cos≥{th['cos_threshold']}  pde≤{th['pde_threshold']}  dE≤{th['e_orb_meV_threshold']} meV",
        "",
        f"**总判定: {verdict}** — {n_pass}/{len(reports)} 通过",
        "",
    ]
    if group_summary:
        md_lines.extend([
            "## 分组通过率",
            "",
            "| 组 | 行数 | 通过 | 通过率 |",
            "|----|------|------|--------|",
        ])
        for g, gs in sorted(group_summary.items()):
            md_lines.append(
                f"| `{g}` | {gs['n']} | {gs['n_pass']} | {100*gs['pass_fraction']:.1f}% |"
            )
    md_lines.extend([
        "",
        "## 指标分布",
        "",
        f"| 指标 | min | median | max |",
        f"|------|-----|--------|-----|",
        f"| cos | {detail_df['cos'].min():.4f} | {detail_df['cos'].median():.4f} | {detail_df['cos'].max():.4f} |",
        f"| pde | {detail_df['pde'].min():.2e} | {detail_df['pde'].median():.2e} | {detail_df['pde'].max():.2e} |",
        f"| dE_meV | {detail_df['dE_meV'].min():.2f} | {detail_df['dE_meV'].median():.2f} | {detail_df['dE_meV'].max():.2f} |",
        "",
        f"详细 CSV: `gate_a_detailed.csv`",
    ])
    md_path = log_dir / "GATE_A_REPORT.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Report written: {out_json}")
    print(f"CSV written:    {log_dir / 'gate_a_detailed.csv'}")
    print(f"Markdown:       {md_path}")


if __name__ == "__main__":
    main()
