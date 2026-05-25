#!/usr/bin/env python3
"""Generate hydrogenic manifest for Stage A (one-electron ions Z=1..z_max)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pinn_art.constants import HARTREE_TO_EV
from pinn_art.ci.racah_compute import parent_config_hash


def build_hydrogenic_manifest(
    z_min: int = 1,
    z_max: int = 8,
    n_levels: int = 6,
) -> pd.DataFrame:
    """
    类氢离子：核电荷 Z、仅 1 个电子（ion_charge = Z-1, nele = 1）。

    每行对应不同主量子数 n；level_eV 为相对该 Z 的 1s 基态激发能（eV）。
    parent_config / level_config 均为 ``{n}s1``（Phase-1 语义，见 prompts/09）。
    """
    rows = []
    for Z in range(z_min, z_max + 1):
        ion_charge = Z - 1
        nele = 1
        E_1_ha = -(float(Z) ** 2) / 2.0
        for n in range(1, n_levels + 1):
            E_n_ha = -(float(Z) ** 2) / (2.0 * float(n) ** 2)
            level_eV = 0.0 if n == 1 else (E_n_ha - E_1_ha) * HARTREE_TO_EV
            cfg = f"{n}s1"
            parent = "1s1"
            rows.append(
                {
                    "Z": Z,
                    "ion_charge": ion_charge,
                    "nele": nele,
                    "parent_config": parent,
                    "level_config": cfg,
                    "J": 0.5,
                    "parity": 0,
                    "term": "2S",
                    "level_eV": float(level_eV),
                    "has_nist_level": True,
                    "parent_config_id": int(parent_config_hash(parent)),
                    "element": _element_symbol(Z),
                }
            )
    return pd.DataFrame(rows)


def _element_symbol(Z: int) -> str:
    symbols = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B",
        6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    }
    return symbols.get(Z, f"Z{Z}")


def main():
    p = argparse.ArgumentParser(description="Build hydrogenic manifest (Stage A)")
    p.add_argument("--z-min", type=int, default=1)
    p.add_argument("--z-max", type=int, default=8, help="Up to O (Z=8) for round-1")
    p.add_argument("--n-levels", type=int, default=6, help="Principal quantum numbers 1..n_levels")
    p.add_argument("--out", default="data_cache/manifest_hydrogenic_z1_8.parquet")
    args = p.parse_args()

    df = build_hydrogenic_manifest(args.z_min, args.z_max, args.n_levels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"Wrote {len(df)} rows -> {out}")
    print(f"  Z range: {args.z_min}..{args.z_max} ({df['element'].unique().tolist()})")
    print(f"  n_levels: 1..{args.n_levels}")
    print(f"  unique parents: {df['parent_config'].nunique()}")


if __name__ == "__main__":
    main()
