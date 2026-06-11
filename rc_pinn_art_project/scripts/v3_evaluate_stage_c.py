#!/usr/bin/env python3
"""Stage C evaluation: CI + NIST hybrid diagonal fill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_art.constants import ENERGY_UNIT
from pinn_art.evaluation.stage_c_eval import run_stage_c_evaluation
from pinn_art.utils.config import load_config


def write_report(metrics, out_dir: Path, ckpt: Path, manifest: Path, cfg, csv_name: str) -> None:
    orb = metrics["layer2_orb"]
    inj = metrics["layer2_inject"]
    lines = [
        "# Stage C 评估报告",
        "",
        f"- Checkpoint: `{ckpt.relative_to(ROOT)}`",
        f"- Manifest: `{manifest.name}`",
        f"- `ci.nist_inject`: **{cfg.ci.nist_inject}**",
        f"- 能量单位: **{ENERGY_UNIT}**",
        "",
        "## 结论（Phase 2a 指标拆分）",
        "",
        "| 层级 | 指标 | 结果 | 门禁 |",
        "|------|------|------|------|",
        f"| **layer2_orb**（主） | E_orb 激发能 vs NIST MAE | **{orb['mae_exc_orb_vs_nist_meV']:.2f} meV** | {orb['verdict']} ({orb['threshold_meV']} meV) |",
        f"| layer2_inject（自检） | E_csf 激发能 vs NIST MAE | {inj['mae_exc_csf_vs_nist_meV']:.4f} meV | {inj['verdict']} |",
        f"| layer2_inject（自检） | |E_csf − E_nist_abs| MAE | {inj['mae_abs_csf_vs_nist_inject_meV']:.4f} meV | — |",
        f"| 数据 | NIST 行 (n>1) / Fall-back | {orb['n_nist_matched']} / {orb['n_theory_fallback']} | — |",
        "",
        "## 说明",
        "",
        "- **主指标** `layer2_orb`：未注入的 Dirac `E_orb` 激发能。",
        "- **inject 自检** `layer2_inject`：对角 NIST 填充管线验证。",
        "",
        f"完整 CSV: `{csv_name}`",
    ]
    (out_dir / "STAGE_C_EVALUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="PINN-ART Stage C evaluation")
    ap.add_argument("--config", default="configs/v3_phase1_stage_c_z1_8.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    ckpt = ROOT / (args.ckpt or getattr(cfg.stage_c, "ckpt", None) or "")
    manifest = ROOT / cfg.dataset.manifest
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg.training.log_dir

    metrics, _, csv_path = run_stage_c_evaluation(cfg, ckpt, manifest, out_dir, root=ROOT)
    write_report(metrics, out_dir, ckpt, manifest, cfg, csv_path.name)

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'STAGE_C_EVALUATION_REPORT.md'}")


if __name__ == "__main__":
    main()
