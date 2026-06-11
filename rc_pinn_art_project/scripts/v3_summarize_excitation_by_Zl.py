#!/usr/bin/env python3
"""Post-process excitation_vs_nist.csv: MAE by Z and by l (single_valence excited rows)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="single_valence")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sub = df[(df["group"] == args.group) & (~df["is_ground"]) & df["has_nist_level"]].copy()
    sub = sub[np.isfinite(sub["err_meV"])]

    lines = [
        f"# 激发能拆解（group={args.group}，非基态，有 NIST）",
        f"",
        f"- 行数: {len(sub)}",
        f"- 全局 MAE: {sub['err_meV'].abs().mean():.2f} meV",
        f"- 全局中位: {sub['err_meV'].abs().median():.2f} meV",
        f"",
        f"## 按元素 (Z)",
        f"",
        f"| Z | 元素 | N | MAE (meV) | 中位数 |",
        f"|---|------|---|-----------|--------|",
    ]
    for Z, g in sub.groupby("Z", sort=True):
        elem = g["element"].iloc[0]
        ae = g["err_meV"].abs()
        lines.append(f"| {Z} | {elem} | {len(g)} | {ae.mean():.2f} | {ae.median():.2f} |")

    lines += [
        f"",
        f"## 按角动量 l (l_active)",
        f"",
        f"| l | N | MAE (meV) | 中位数 |",
        f"|---|---|-----------|--------|",
    ]
    for l, g in sub.groupby("l_active", sort=True):
        ae = g["err_meV"].abs()
        lines.append(f"| {l} | {len(g)} | {ae.mean():.2f} | {ae.median():.2f} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
