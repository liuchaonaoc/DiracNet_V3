#!/usr/bin/env python3
"""Precompute Racah / angular coefficient cache from manifest parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from pinn_art.ci.racah_cache import build_cache_from_manifest_rows


def main():
    ap = argparse.ArgumentParser(description="Build Racah angular coefficient cache (.npz)")
    ap.add_argument("--manifest", default="data_cache/manifest_hydrogenic_v3.parquet")
    ap.add_argument("--out", default="data_cache/racah_cache_v3.npz")
    ap.add_argument("--config", default=None, help="Optional yaml for k_list / M_max")
    ap.add_argument("--M-max", type=int, default=8, dest="M_max")
    ap.add_argument("--k-list", default="0,1,2,3,4", help="Comma-separated k values")
    args = ap.parse_args()

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    M_max = args.M_max

    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with cfg_path.open() as f:
                cfg = yaml.safe_load(f) or {}
            ci = cfg.get("ci", {})
            k_list = list(ci.get("k_list", k_list))
            M_max = int(ci.get("M_max", M_max))

    manifest = ROOT / args.manifest
    if not manifest.exists():
        print(f"Manifest not found: {manifest}", file=sys.stderr)
        print("Run: python scripts/v3_prepare_hydrogenic.py", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(manifest)
    rows = df.to_dict(orient="records")
    cache = build_cache_from_manifest_rows(rows, k_list=k_list, M_max=M_max)

    out = ROOT / args.out
    cache.save(out)
    print(f"Wrote Racah cache: {out}")
    print(f"  parents={len(cache.parent_ids)}  k_list={list(cache.k_list)}  M_max={cache.M_max}")
    print(f"  C shape={cache.C.shape}  version={cache.version}")


if __name__ == "__main__":
    main()
