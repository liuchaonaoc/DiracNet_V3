#!/usr/bin/env python3
"""Build NIST manifest subset (Z<=z_max). Phase 2c entry: delegates to hydrogenic+NIST builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "v3_prepare_nist_manifest", ROOT / "scripts" / "v3_prepare_nist_manifest.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_nist_manifest = _mod.build_nist_manifest


def main():
    p = argparse.ArgumentParser(description="NIST manifest subset (Z=1..z_max, ns levels)")
    p.add_argument("--z-min", type=int, default=1)
    p.add_argument("--z-max", type=int, default=26)
    p.add_argument("--n-levels", type=int, default=6)
    p.add_argument("--out", default="data_cache/manifest_nist_subset_z1_26.parquet")
    args = p.parse_args()

    df = build_nist_manifest(args.z_min, args.z_max, args.n_levels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    print(f"  Z={args.z_min}..{args.z_max}  n=1..{args.n_levels}")
    print(f"  NIST matched: {int(df['has_nist_level'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
