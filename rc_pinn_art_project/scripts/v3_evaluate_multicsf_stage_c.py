#!/usr/bin/env python3
"""Stage C Phase 2c: multi-CSF spectrum evaluation (2p fine structure)."""

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
from pinn_art.constants import ENERGY_UNIT, hartree_to_meV
from pinn_art.data.collate import collate_spectrum_group
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params, merge_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_phase1_stage_c_multicsf_z1_8.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    ckpt = ROOT / (args.ckpt or cfg.stage_c.ckpt)
    manifest = ROOT / cfg.dataset.manifest
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg.training.log_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    racah_cache = RacahCache.load(ROOT / cfg.ci.racah_cache)
    k_list = tuple(int(k) for k in cfg.ci.k_list)
    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    mdf = ds.df
    grid = make_radial_grid(
        float(cfg.grid.r_min),
        float(cfg.grid.r_max),
        int(cfg.grid.n_grid),
        str(cfg.grid.scheme),
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    params = merge_params(params, load_params(ckpt))

    rows = []
    for spectrum_id, grp in mdf.groupby("spectrum_id"):
        items = [ds[int(i)] for i in grp.index]
        batch = collate_spectrum_group(
            items,
            n_csf_max=int(cfg.model.n_csf_max),
            racah_cache=racah_cache,
            k_list=k_list,
        )
        out = model.apply(params, batch, grid, train=False, return_ci=True)
        H = np.asarray(out["H"][0])
        M = int(np.sum(np.asarray(batch["csf_mask"][0])))
        off = H[:M, :M] - np.diag(np.diag(H[:M, :M]))
        max_off = float(np.max(np.abs(off))) if M > 1 else 0.0

        E_csf = np.asarray(out["E_csf"][0, :M])
        E_orb0 = float(out["E_orb"][0, 0])
        grp_s = grp.sort_values("csf_slot")
        for slot in range(M):
            r = grp_s.iloc[slot]
            rows.append({
                "spectrum_id": spectrum_id,
                "Z": int(r["Z"]),
                "element": str(r["element"]),
                "level_config": str(r["level_config"]),
                "J": float(r["J"]),
                "csf_slot": slot,
                "level_meV": float(r["level_meV"]),
                "E_csf_meV": float(hartree_to_meV(E_csf[slot])),
                "E_orb_meV": float(hartree_to_meV(E_orb0)),
                "nist_mask": bool(batch["nist_mask"][0, slot]),
                "diag_source": int(out["diag_source"][0, slot]),
                "max_offdiag_H_meV": float(hartree_to_meV(max_off)),
                "has_offdiag_coupling": max_off > 1e-12,
            })

    df = pd.DataFrame(rows)
    abs_refs = []
    for _, r in df.iterrows():
        mrow = mdf[(mdf["spectrum_id"] == r["spectrum_id"]) & (mdf["csf_slot"] == r["csf_slot"])].iloc[0]
        abs_refs.append(float(mrow["level_abs_meV"]))
    df["level_abs_meV"] = abs_refs
    df["err_abs_csf_vs_nist_meV"] = (df["E_csf_meV"] - df["level_abs_meV"]).abs()

    mae_abs = float(df["err_abs_csf_vs_nist_meV"].mean())
    n_groups = int(mdf["spectrum_id"].nunique())
    n_off_groups = int(df.groupby("spectrum_id")["has_offdiag_coupling"].any().sum())

    # Fine-structure splitting: ΔE_csf vs ΔE_NIST per spectrum
    split_errs = []
    for sid, g in df.groupby("spectrum_id"):
        if len(g) < 2:
            continue
        g = g.sort_values("csf_slot")
        d_csf = float(g["E_csf_meV"].iloc[-1] - g["E_csf_meV"].iloc[0])
        d_nist = float(g["level_meV"].iloc[-1] - g["level_meV"].iloc[0])
        split_errs.append(abs(d_csf - d_nist))
    mae_split = float(np.mean(split_errs)) if split_errs else float("nan")

    metrics = {
        "phase": "2c_multicsf",
        "n_spectrum_groups": n_groups,
        "n_rows": len(df),
        "mae_abs_csf_vs_nist_inject_meV": mae_abs,
        "mae_fine_structure_split_meV": mae_split,
        "n_groups_with_offdiag": n_off_groups,
        "max_offdiag_H_meV_median": float(df["max_offdiag_H_meV"].median()) if len(df) else 0.0,
        "energy_unit": ENERGY_UNIT,
        "verdict_pipeline": "PASS" if n_off_groups == n_groups and n_groups > 0 else "FAIL",
        "verdict": "PASS" if n_off_groups == n_groups and n_groups > 0 else "FAIL",
    }

    csv_path = out_dir / "stage_c_multicsf_levels.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    with (out_dir / "metrics_multicsf.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    md = [
        "# Stage C Phase 2c — 多 CSF 评估",
        "",
        f"- Manifest: `{manifest.name}`",
        f"- 谱组数: **{metrics['n_spectrum_groups']}**",
        f"- 非对角 |H_off|>0 的组: **{n_off_groups}/{n_groups}**",
        f"- 精细结构裂距 MAE (ΔE_csf vs ΔE_NIST): **{mae_split:.2f} meV**",
        f"- |E_csf − E_nist_abs| MAE（含 off-diagonal 混合，仅参考）: {mae_abs:.2f} meV",
        f"- VERDICT（管线）: **{metrics['verdict']}**",
        "",
        f"CSV: `{csv_path.name}`",
    ]
    (out_dir / "STAGE_C_MULTICSF_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
