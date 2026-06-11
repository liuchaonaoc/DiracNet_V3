#!/usr/bin/env python3
"""Build multi-CSF manifest: 2p fine-structure doublet per Z (H..O), one spectrum per ion."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = ROOT.parent.parent / "DiracNet_V1" / "rc_diracnet_project"
sys.path.insert(0, str(ROOT))
if V1_ROOT.exists():
    sys.path.insert(0, str(V1_ROOT))

from pinn_art.ci.racah_compute import parent_config_hash
from pinn_art.constants import ev_to_meV, hartree_to_meV, meV_to_hartree
from pinn_art.physics.hydrogenic import hydrogenic_energy

_LEVEL_CFG = re.compile(r"^(\d+)([spdfghi])", re.IGNORECASE)
NIST_RAW = V1_ROOT / "data_raw" / "nist"


def _nist_csv_path(Z: int, ion_charge: int) -> Path:
    from rc_diracnet.data.nist_parser import Z_to_symbol, int_to_roman

    return NIST_RAW / f"{Z_to_symbol(Z)}{int_to_roman(ion_charge + 1)}.csv"


def _element_symbol(Z: int) -> str:
    return {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O"}.get(Z, f"Z{Z}")


def _load_2p_csfs(Z: int, ion_charge: int) -> list[dict]:
    path = _nist_csv_path(Z, ion_charge)
    if not path.exists():
        return []
    from rc_diracnet.data.nist_parser import parse_nist_levels_csv

    df = parse_nist_levels_csv(path, Z, ion_charge)
    out = []
    for _, r in df.iterrows():
        cfg = str(r["level_config"]).strip().lower().replace(" ", "")
        m = _LEVEL_CFG.match(cfg)
        if not m or m.group(2).lower() != "p" or int(m.group(1)) != 2:
            continue
        out.append({
            "level_config": "2p1",
            "J": float(r["J"]),
            "parity": int(r.get("parity", 1)),
            "term": str(r.get("term", "2P")),
            "level_meV": float(ev_to_meV(float(r["level_eV"]))),
        })
    # dedupe by J
    by_j: dict[float, dict] = {}
    for row in out:
        by_j[row["J"]] = row
    return [by_j[j] for j in sorted(by_j.keys())]


def build_multicsf_manifest(z_min: int = 1, z_max: int = 8) -> pd.DataFrame:
    rows = []
    for Z in range(z_min, z_max + 1):
        ion_charge = Z - 1
        parent = "1s1"
        E_1_ha = hydrogenic_energy(Z, 1)
        E_2p_ha = hydrogenic_energy(Z, 2)  # crude; l=1 not in hydrogenic_energy but n=2 scale
        hyd_2p_exc = float(hartree_to_meV(E_2p_ha - E_1_ha))

        csfs = _load_2p_csfs(Z, ion_charge)
        if len(csfs) < 2:
            continue
        spectrum_id = f"Z{Z}_ion{ion_charge}_2p"
        for slot, c in enumerate(csfs[:4]):
            nist_exc = c["level_meV"]
            hyd_abs = float(hartree_to_meV(E_1_ha)) + hyd_2p_exc
            level_abs = hyd_abs + (nist_exc - hyd_2p_exc)
            rows.append({
                "Z": Z,
                "ion_charge": ion_charge,
                "nele": 1,
                "parent_config": parent,
                "level_config": c["level_config"],
                "J": c["J"],
                "parity": c["parity"],
                "term": c["term"],
                "level_meV": float(nist_exc),
                "level_abs_meV": float(level_abs),
                "level_eV": float(nist_exc) / 1000.0,
                "has_nist_level": True,
                "parent_config_id": int(parent_config_hash(parent)),
                "element": _element_symbol(Z),
                "spectrum_id": spectrum_id,
                "csf_slot": slot,
                "n_csf_in_spectrum": min(len(csfs), 4),
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Build multi-CSF 2p manifest (Stage C Phase 2c)")
    p.add_argument("--z-min", type=int, default=1)
    p.add_argument("--z-max", type=int, default=8)
    p.add_argument("--out", default="data_cache/manifest_nist_multicsf_z1_8.parquet")
    args = p.parse_args()

    if not NIST_RAW.exists():
        raise FileNotFoundError(NIST_RAW)

    df = build_multicsf_manifest(args.z_min, args.z_max)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    print(f"  spectrum groups: {df['spectrum_id'].nunique()}")
    print(f"  CSFs per group: {df.groupby('spectrum_id').size().tolist()}")


if __name__ == "__main__":
    main()
