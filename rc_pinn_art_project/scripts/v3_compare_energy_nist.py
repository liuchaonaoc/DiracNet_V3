#!/usr/bin/env python3
"""Compare predicted orbital energies (Phase 3 ckpt) vs NIST ASD & references."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = ROOT.parent.parent / "DiracNet_V1" / "rc_diracnet_project"
sys.path.insert(0, str(ROOT))
if V1_ROOT.exists():
    sys.path.insert(0, str(V1_ROOT))

from pinn_art.constants import ENERGY_UNIT, ev_to_meV, hartree_to_meV
from pinn_art.ci.racah_cache import RacahCache
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.hydrogenic import hydrogenic_energy
from pinn_art.training.checkpoint import load_params
from pinn_art.training.gate_a import parse_n_l_from_level_config
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid

_LEVEL_CFG = re.compile(r"^(\d+)([spdfghi])", re.IGNORECASE)

NIST_RAW = V1_ROOT / "data_raw" / "nist"


def _nist_csv_path(Z: int, ion_charge: int) -> Path:
    from rc_diracnet.data.nist_parser import Z_to_symbol, int_to_roman

    return NIST_RAW / f"{Z_to_symbol(Z)}{int_to_roman(ion_charge + 1)}.csv"


def _load_nist_ns_levels(Z: int, ion_charge: int) -> dict[tuple[int, int], dict]:
    """Map (n, l=0) -> {nist_exc_meV, ...} (NIST CSV ``level_eV`` converted to meV)."""
    path = _nist_csv_path(Z, ion_charge)
    if not path.exists():
        return {}
    from rc_diracnet.data.nist_parser import parse_nist_levels_csv

    df = parse_nist_levels_csv(path, Z, ion_charge)
    out: dict[tuple[int, int], dict] = {}
    for _, r in df.iterrows():
        cfg = str(r["level_config"]).strip().lower().replace(" ", "")
        m = _LEVEL_CFG.match(cfg)
        if not m:
            continue
        n, l_letter = int(m.group(1)), m.group(2).lower()
        if l_letter != "s":
            continue
        key = (n, 0)
        # 同一 n 可能多 J；取 2S 1/2 或第一条
        if key not in out or abs(float(r["J"]) - 0.5) < abs(out[key]["J"] - 0.5):
            out[key] = {
                "nist_level_config": cfg,
                "nist_term": str(r.get("term", "")),
                "nist_J": float(r["J"]),
                "nist_exc_meV": float(ev_to_meV(float(r["level_eV"]))),
                "nist_unc_meV": float(ev_to_meV(float(r.get("uncertainty_eV", 0.0)))),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3_phase1_stage_a_z1_8_phase3.yaml")
    ap.add_argument(
        "--ckpt",
        default="checkpoints/v3_phase1_stage_a_z1_8_phase3/stage_a_last.msgpack",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--return-ci",
        action="store_true",
        help="Also run CI branch (Stage C: E_csf + nist inject)",
    )
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    manifest = ROOT / cfg.dataset.manifest
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg.training.log_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    grid = make_radial_grid(
        float(cfg.grid.r_min),
        float(cfg.grid.r_max),
        int(cfg.grid.n_grid),
        str(cfg.grid.scheme),
    )
    model, _ = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    params = load_params(ROOT / args.ckpt)

    use_ci = args.return_ci or bool(getattr(getattr(cfg, "ci", None), "enabled", False))
    racah_cache = None
    k_list = ()
    if use_ci and getattr(cfg, "ci", None):
        racah_path = ROOT / cfg.ci.racah_cache
        if racah_path.exists():
            racah_cache = RacahCache.load(racah_path)
            k_list = tuple(int(k) for k in cfg.ci.k_list)

    mdf = pd.read_parquet(manifest)
    rows = []
    e_pred_1s: dict[int, float] = {}
    e_csf_1s: dict[int, float] = {}

    for i in range(len(ds)):
        batch = collate_batches(
            [ds[i]],
            n_csf_max=int(getattr(cfg.model, "n_csf_max", 8)),
            racah_cache=racah_cache,
            k_list=k_list if racah_cache else None,
        )
        out = model.apply(params, batch, grid, train=False, return_ci=use_ci)
        Z = int(batch["Z"][0])
        ion = int(batch["ion_charge"][0])
        level_cfg = str(mdf.iloc[i]["level_config"])
        n, l = parse_n_l_from_level_config(level_cfg)
        E_pred_ha = float(out["E_orb"][0, 0])
        E_csf_ha = float(out["E_csf"][0, 0]) if use_ci and out.get("E_csf") is not None else float("nan")
        if n == 1 and l == 0:
            e_pred_1s[Z] = E_pred_ha
            if np.isfinite(E_csf_ha):
                e_csf_1s[Z] = E_csf_ha

        rows.append({
            "row": i,
            "Z": Z,
            "element": str(mdf.iloc[i].get("element", f"Z{Z}")),
            "ion_charge": ion,
            "level_config": level_cfg,
            "n": n,
            "l": l,
            "E_pred_meV": float(hartree_to_meV(E_pred_ha)),
            "E_csf_meV": float(hartree_to_meV(E_csf_ha)) if np.isfinite(E_csf_ha) else np.nan,
        })

    df = pd.DataFrame(rows)
    e_pred_1s_meV = {z: float(hartree_to_meV(ha)) for z, ha in e_pred_1s.items()}
    df["E_pred_exc_meV"] = df.apply(
        lambda r: (r["E_pred_meV"] - e_pred_1s_meV[r["Z"]]) if r["n"] > 1 else 0.0,
        axis=1,
    )
    if use_ci and e_csf_1s:
        e_csf_1s_meV = {z: float(hartree_to_meV(ha)) for z, ha in e_csf_1s.items()}
        df["E_csf_exc_meV"] = df.apply(
            lambda r: (r["E_csf_meV"] - e_csf_1s_meV[r["Z"]]) if r["n"] > 1 and np.isfinite(r["E_csf_meV"]) else 0.0,
            axis=1,
        )
    df["E_hydrogenic_meV"] = df.apply(
        lambda r: float(hartree_to_meV(hydrogenic_energy(int(r["Z"]), int(r["n"])))), axis=1
    )
    df["E_hydrogenic_exc_meV"] = df.apply(
        lambda r: (
            0.0
            if r["n"] == 1
            else (
                r["E_hydrogenic_meV"]
                - float(hartree_to_meV(hydrogenic_energy(int(r["Z"]), 1)))
            )
        ),
        axis=1,
    )
    if "level_meV" in mdf.columns:
        df["manifest_exc_meV"] = mdf["level_meV"].values
    else:
        df["manifest_exc_meV"] = mdf["level_eV"].values * 1000.0
    df["err_pred_vs_hydrogenic_meV"] = (df["E_pred_meV"] - df["E_hydrogenic_meV"]).abs()
    df["err_exc_vs_manifest_meV"] = (df["E_pred_exc_meV"] - df["manifest_exc_meV"]).abs()

    # NIST ASD (real)
    nist_exc = []
    nist_cfg = []
    nist_term = []
    nist_unc = []
    nist_matched = []
    for _, r in df.iterrows():
        Z, ion, n, l = int(r["Z"]), int(r["ion_charge"]), int(r["n"]), int(r["l"])
        ns = _load_nist_ns_levels(Z, ion)
        hit = ns.get((n, l))
        if hit:
            nist_exc.append(hit["nist_exc_meV"])
            nist_cfg.append(hit["nist_level_config"])
            nist_term.append(hit["nist_term"])
            nist_unc.append(hit["nist_unc_meV"])
            nist_matched.append(True)
        else:
            nist_exc.append(np.nan)
            nist_cfg.append("")
            nist_term.append("")
            nist_unc.append(np.nan)
            nist_matched.append(False)

    df["nist_matched"] = nist_matched
    df["nist_level_config"] = nist_cfg
    df["nist_term"] = nist_term
    df["nist_exc_meV"] = nist_exc
    df["nist_unc_meV"] = nist_unc
    df["err_exc_vs_nist_meV"] = (df["E_pred_exc_meV"] - df["nist_exc_meV"]).abs()
    df["err_exc_vs_nist_minus_manifest_meV"] = (df["nist_exc_meV"] - df["manifest_exc_meV"]).abs()
    if use_ci and "E_csf_exc_meV" in df.columns:
        df["err_exc_csf_vs_nist_meV"] = (df["E_csf_exc_meV"] - df["nist_exc_meV"]).abs()

    csv_path = out_dir / "energy_vs_nist_comparison.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    # Summary markdown
    n_nist = int(df["nist_matched"].sum())
    mae_nist = float(df.loc[df["nist_matched"], "err_exc_vs_nist_meV"].mean()) if n_nist else float("nan")
    mae_h = float(df["err_pred_vs_hydrogenic_meV"].mean())
    mae_man = float(df["err_exc_vs_manifest_meV"].mean())

    mae_csf_line = ""
    if use_ci and "err_exc_csf_vs_nist_meV" in df.columns:
        mae_csf = float(df.loc[df["nist_matched"] & (df["n"] > 1), "err_exc_csf_vs_nist_meV"].mean())
        mae_csf_line = f"| 预测 E_csf 激发能 vs NIST（inject 后） | **{mae_csf:.4f}** meV | inject 自检 |"

    lines = [
        "# 预测能量 vs NIST / 参考",
        "",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Config: `{args.config}`",
        f"- CI / return_ci: **{use_ci}**",
        f"- Manifest: `{manifest.name}`",
        f"- NIST 数据源: `DiracNet_V1/.../data_raw/nist/{{Element}}{{Roman}}.csv`（ASD Levels，读入后转为 **{ENERGY_UNIT}**）",
        f"- 对外能量单位: **{ENERGY_UNIT}**（内部 Dirac/CI 仍为 Hartree）",
        "",
        "## 说明",
        "",
        "| 列 | 含义 |",
        "|----|------|",
        "| `E_pred_meV` | 模型 `E_orb`（绝对能量，meV） |",
        "| `E_pred_exc_meV` | 相对同 Z 预测 1s 的激发能（meV） |",
        "| `manifest_exc_meV` | parquet `level_meV`（类氢理论激发能） |",
        "| `nist_exc_meV` | NIST ASD `{n}s` 激发能（meV，由 CSV eV×1000） |",
        "| `E_hydrogenic_meV` | 类氢解析 `-Z²/(2n²)`（绝对，meV） |",
        "",
        "## 汇总误差（激发能，相对 1s）",
        "",
        f"| 对比 | MAE (meV) | 备注 |",
        f"|------|-----------|------|",
        f"| 预测 E_orb 激发能 vs **真实 NIST** `{n_nist}/48` 行匹配 | **{mae_nist:.2f}** | 主指标 (layer2_orb) |",
    ]
    if mae_csf_line:
        lines.append(mae_csf_line)
    lines.extend([
        f"| 预测 vs manifest（类氢） | {mae_man:.2f} | Stage A 训练标签 |",
        f"| 预测绝对能量 vs 类氢解析 | {mae_h:.2f} | 与 Gate dE 一致 |",
        "",
        "## 按元素：激发能 MAE vs NIST (meV)",
        "",
        "| Z | 元素 | 匹配 NIST | MAE vs NIST | MAE vs manifest |",
        "|---|------|-----------|-------------|-----------------|",
    ])
    for Z, g in df.groupby("Z"):
        el = g["element"].iloc[0]
        nm = int(g["nist_matched"].sum())
        mae_n = g.loc[g["nist_matched"], "err_exc_vs_nist_meV"].mean() if nm else float("nan")
        mae_m = g["err_exc_vs_manifest_meV"].mean()
        lines.append(f"| {int(Z)} | {el} | {nm}/6 | {mae_n:.2f} | {mae_m:.2f} |")

    lines.extend([
        "",
        "## 逐行表（前 12 行示例）",
        "",
        df.head(12)[
            [
                "Z", "element", "level_config", "E_pred_exc_meV", "nist_exc_meV",
                "manifest_exc_meV", "err_exc_vs_nist_meV", "nist_matched",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"完整 CSV: `{csv_path.name}`",
        "",
        "## 最大偏差（相对 NIST 激发能）",
        "",
    ])
    if n_nist:
        worst = df[df["nist_matched"]].nlargest(8, "err_exc_vs_nist_meV")
        lines.append(
            worst[
                ["Z", "element", "level_config", "E_pred_exc_meV", "nist_exc_meV", "err_exc_vs_nist_meV"]
            ].to_markdown(index=False, floatfmt=".4f")
        )

    md_path = out_dir / "ENERGY_VS_NIST.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"NIST matched: {n_nist}/48")
    print(f"MAE exc vs NIST: {mae_nist:.2f} meV")
    print(f"MAE exc vs manifest (hydrogenic): {mae_man:.2f} meV")
    print(f"MAE |E_pred| vs hydrogenic abs: {mae_h:.2f} meV")


if __name__ == "__main__":
    main()
