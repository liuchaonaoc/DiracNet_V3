#!/usr/bin/env python3
"""R0.1 — Enumerate Z=1..26, n<=10 ground + single-electron excitation configs.

Produces the Round-2 configuration inventory used to drive the NIST manifest
(``v3_prepare_nist_manifest.py``) and Stage-A training. Each row carries:

    Z, ion_charge, nele, parent_config, level_config, J, parity, term,
    n_active, l_active, is_ground, group, parent_config_id

``group`` is the ``single_valence`` / ``multi_electron`` split (R0.3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_art.ci.racah_compute import parent_config_hash
from pinn_art.data.config_enum import enumerate_all

_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe",
]


def _element_symbol(Z: int) -> str:
    return _ELEMENTS[Z - 1] if 1 <= Z <= len(_ELEMENTS) else f"Z{Z}"


def build_config_table(
    z_min: int = 1,
    z_max: int = 26,
    n_max: int = 10,
    ion_charges: str = "all",
) -> pd.DataFrame:
    rows = []
    for lv in enumerate_all(z_min, z_max, n_max=n_max, ion_charges=ion_charges):
        rows.append({
            "Z": lv.Z,
            "ion_charge": lv.ion_charge,
            "nele": lv.nele,
            "parent_config": lv.parent_config,
            "level_config": lv.level_config,
            "J": lv.J,
            "parity": lv.parity,
            "term": lv.term,
            "n_active": lv.n_active,
            "l_active": lv.l_active,
            "is_ground": lv.is_ground,
            "group": lv.group,
            "parent_config_id": int(parent_config_hash(lv.parent_config)),
            "element": _element_symbol(lv.Z),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Enumerate Round-2 configs (R0.1)")
    p.add_argument("--z-min", type=int, default=1)
    p.add_argument("--z-max", type=int, default=26)
    p.add_argument("--n-max", type=int, default=10)
    p.add_argument(
        "--ion-charges", choices=["all", "neutral", "bare"], default="all",
        help="which ion stages to enumerate (default: all stages)",
    )
    p.add_argument("--out", default="data_cache/configs_z1_26_n10.parquet")
    args = p.parse_args()

    df = build_config_table(args.z_min, args.z_max, args.n_max, args.ion_charges)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    grp = df.groupby("group").size()
    print(f"Wrote {len(df)} config rows -> {out}")
    print(f"  Z range: {args.z_min}..{args.z_max}   n_max: {args.n_max}   ions: {args.ion_charges}")
    print(f"  (Z, ion) pairs: {df[['Z', 'ion_charge']].drop_duplicates().shape[0]}")
    print(f"  unique level_config: {df['level_config'].nunique()}")
    print("  group breakdown:")
    for g in ("single_valence", "multi_electron"):
        n = int(grp.get(g, 0))
        print(f"    {g:16s}: {n} rows")


if __name__ == "__main__":
    main()
