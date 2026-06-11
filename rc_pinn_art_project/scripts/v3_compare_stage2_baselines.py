#!/usr/bin/env python3
"""Quick baseline comparison: H excitation + optional metrics.json paths."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CFG = "configs/v3_stage_a_z1_26_n10.yaml"
NIST_H_2S_EV = 10.19881

BASELINE_CKPTS = {
    "p1lowz": "checkpoints/v3_stage_a_z1_26_n10_p1lowz/stage_a_last.msgpack",
    "p1z8_ground": "checkpoints/v3_stage_a_z1_26_n10_p1z8_ground/stage_a_last.msgpack",
    "round2": "checkpoints/v3_stage_a_z1_26_n10/stage_a_last.msgpack",
    "round1": "checkpoints/v3_phase1_stage_a_z1_8_phase3/stage_a_last.msgpack",
}


def _run_diag(ckpt: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "v3_diag_h_potential.py"),
        "--config", CFG,
        "--ckpt", str(ckpt),
    ]
    out = subprocess.check_output(cmd, text=True, cwd=str(ROOT), stderr=subprocess.STDOUT)
    exc_m = re.search(r"predicted 1s->2s excitation = ([\d.]+) eV", out)
    r1_m = re.search(r"r=\s+1\.026\s+V_net=\s*([-\d.]+).*diff=\s*([-\d.]+)", out)
    e1s_m = re.search(r"\[1s1\].*E_orb=([-\d.]+) Ha", out)
    return {
        "exc_eV": float(exc_m.group(1)) if exc_m else float("nan"),
        "exc_err_meV": (float(exc_m.group(1)) - NIST_H_2S_EV) * 1000 if exc_m else float("nan"),
        "v_r1_diff": float(r1_m.group(2)) if r1_m else float("nan"),
        "E_1s_ha": float(e1s_m.group(1)) if e1s_m else float("nan"),
        "raw_tail": out.strip().split("\n")[-3:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Primary ckpt to highlight")
    ap.add_argument("--baselines", default="p1lowz,p1z8_ground,round2",
                    help="Comma-separated keys from built-in map")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keys = ["target"] + [k.strip() for k in args.baselines.split(",") if k.strip()]
    ckpts = {"target": args.target}
    for k in args.baselines.split(","):
        k = k.strip()
        if k in BASELINE_CKPTS:
            ckpts[k] = BASELINE_CKPTS[k]

    lines = [
        "# Stage 2 基线对比（氢锚点）",
        "",
        f"NIST H 1s→2s = {NIST_H_2S_EV:.4f} eV",
        "",
        "| 标签 | checkpoint | E(1s) Ha | 激发能 (eV) | 误差 (meV) | V_net(r≈1)−(−1/r) |",
        "|------|------------|----------|-------------|------------|-------------------|",
    ]
    for label, path in ckpts.items():
        p = ROOT / path
        if not p.exists():
            lines.append(f"| {label} | `{path}` | — | — | — | **missing** |")
            continue
        d = _run_diag(p)
        lines.append(
            f"| {label} | `{path}` | {d['E_1s_ha']:.5f} | {d['exc_eV']:.4f} | "
            f"{d['exc_err_meV']:.1f} | {d['v_r1_diff']:+.4f} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
