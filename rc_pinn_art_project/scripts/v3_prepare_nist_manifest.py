#!/usr/bin/env python3
"""R0.2/R0.3 — Round-2 NIST manifest: arbitrary l, J, level_config + grouping.

Drives the configuration inventory from ``config_enum.enumerate_all`` (R0.1),
then annotates each enumerated level with NIST ASD data when a matching level
exists. No longer restricted to ``ns`` levels.

Columns
-------
  Z, ion_charge, nele, parent_config, level_config, J, parity, term
  group           — single_valence / multi_electron (R0.3)
  level_meV       — excitation vs ground (NIST when matched, else hydrogenic)
  level_abs_meV   — absolute energy on (hydrogenic) E_orb scale
  has_nist_level  — True when an NIST ASD level matched this row
  n_active, l_active, is_ground, parent_config_id, element

Discipline (00_overview §2.1): NIST values are reference annotations only and
never enter Stage-A backward; this script merely tags coverage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = ROOT.parent.parent / "DiracNet_V1" / "rc_diracnet_project"
sys.path.insert(0, str(ROOT))
if V1_ROOT.exists():
    sys.path.insert(0, str(V1_ROOT))

from pinn_art.ci.racah_compute import parent_config_hash
from pinn_art.constants import ev_to_meV, hartree_to_meV
from pinn_art.data.config_enum import (
    canonical_config,
    enumerate_levels,
    parse_config_tuples,
)
from pinn_art.physics.hydrogenic import hydrogenic_energy

NIST_RAW = V1_ROOT / "data_raw" / "nist"

_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe",
]


def _element_symbol(Z: int) -> str:
    return _ELEMENTS[Z - 1] if 1 <= Z <= len(_ELEMENTS) else f"Z{Z}"


def _nist_csv_path(Z: int, ion_charge: int) -> Path:
    from rc_diracnet.data.nist_parser import Z_to_symbol, int_to_roman

    return NIST_RAW / f"{Z_to_symbol(Z)}{int_to_roman(ion_charge + 1)}.csv"


def _hyd_total_ha(Z: int, shells) -> float:
    """Crude hydrogenic total energy (Σ E_{n} · occ) for an absolute baseline."""
    return sum(hydrogenic_energy(Z, n) * occ for (n, _, occ) in shells)


class NistLevels:
    """Per-(Z, ion) NIST lookup keyed by canonical config + 2J."""

    def __init__(self) -> None:
        self.by_cfg_j: dict[tuple[str, int], float] = {}   # -> excitation eV
        self.by_cfg_min: dict[str, float] = {}             # -> min excitation eV
        self.term_by_cfg_j: dict[tuple[str, int], str] = {}

    @classmethod
    def load(cls, Z: int, ion_charge: int) -> "NistLevels":
        self = cls()
        path = _nist_csv_path(Z, ion_charge)
        if not path.exists():
            return self
        from rc_diracnet.data.nist_parser import parse_nist_levels_csv

        try:
            df = parse_nist_levels_csv(path, Z, ion_charge)
        except Exception:
            return self
        nele = Z - ion_charge
        for _, r in df.iterrows():
            cfg = canonical_config(str(r["level_config"]), nele=nele)
            if not cfg:
                continue
            J = r.get("J")
            if J is None or pd.isna(J):
                continue
            exc_eV = float(r["level_eV"])
            twice_j = int(round(float(J) * 2))
            key = (cfg, twice_j)
            if key not in self.by_cfg_j or exc_eV < self.by_cfg_j[key]:
                self.by_cfg_j[key] = exc_eV
                self.term_by_cfg_j[key] = str(r.get("term", ""))
            if cfg not in self.by_cfg_min or exc_eV < self.by_cfg_min[cfg]:
                self.by_cfg_min[cfg] = exc_eV
        return self

    def match(self, cfg: str, twice_j: int) -> tuple[float, str] | None:
        """Return (excitation_eV, term) for a level, or None.

        ``twice_j >= 0`` -> exact (config, 2J) match (single_valence / fine
        structure). ``twice_j < 0`` (multi_electron placeholder) -> fall back to
        the lowest NIST level of that configuration as a representative.
        """
        if twice_j >= 0:
            key = (cfg, twice_j)
            if key in self.by_cfg_j:
                return self.by_cfg_j[key], self.term_by_cfg_j.get(key, "")
            return None
        if cfg in self.by_cfg_min:
            return self.by_cfg_min[cfg], ""
        return None


def build_nist_manifest(
    z_min: int = 1,
    z_max: int = 26,
    n_max: int = 10,
    ion_charges: str = "all",
) -> pd.DataFrame:
    rows = []
    for Z in range(z_min, z_max + 1):
        if ion_charges == "neutral":
            charges = [0]
        elif ion_charges == "bare":
            charges = [Z - 1]
        else:
            charges = list(range(0, Z))
        for q in charges:
            nist = NistLevels.load(Z, q)
            levels = enumerate_levels(Z, q, n_max=n_max)
            if not levels:
                continue
            ground_shells = parse_config_tuples(levels[0].parent_config)
            e_ground_ha = _hyd_total_ha(Z, ground_shells)
            ground_abs_meV = float(hartree_to_meV(e_ground_ha))
            for lv in levels:
                lvl_shells = parse_config_tuples(lv.level_config)
                hyd_exc_meV = float(
                    hartree_to_meV(_hyd_total_ha(Z, lvl_shells) - e_ground_ha)
                )
                hit = nist.match(lv.level_config, lv.twice_j)
                if hit is not None and not lv.is_ground:
                    level_meV = float(ev_to_meV(hit[0]))
                    term = hit[1] or lv.term
                    has_nist = True
                elif hit is not None and lv.is_ground:
                    # ground reference is 0 by construction
                    level_meV = 0.0
                    term = hit[1] or lv.term
                    has_nist = True
                else:
                    level_meV = 0.0 if lv.is_ground else hyd_exc_meV
                    term = lv.term
                    has_nist = False
                rows.append({
                    "Z": Z,
                    "ion_charge": q,
                    "nele": lv.nele,
                    "parent_config": lv.parent_config,
                    "level_config": lv.level_config,
                    "J": lv.J,
                    "parity": lv.parity,
                    "term": term,
                    "group": lv.group,
                    "level_meV": float(level_meV),
                    "level_abs_meV": float(ground_abs_meV + level_meV),
                    "level_eV": float(level_meV) / 1000.0,
                    "has_nist_level": bool(has_nist),
                    "n_active": lv.n_active,
                    "l_active": lv.l_active,
                    "is_ground": lv.is_ground,
                    "parent_config_id": int(parent_config_hash(lv.parent_config)),
                    "element": _element_symbol(Z),
                })
    return pd.DataFrame(rows)


def _print_group_coverage(df: pd.DataFrame) -> None:
    print(f"Total rows: {len(df)}")
    print(f"  (Z, ion) pairs: {df[['Z', 'ion_charge']].drop_duplicates().shape[0]}")
    overall = int(df["has_nist_level"].sum())
    print(f"  overall NIST coverage: {overall}/{len(df)} ({100*overall/max(len(df),1):.1f}%)")
    for g in ("single_valence", "multi_electron"):
        sub = df[df["group"] == g]
        n = len(sub)
        hit = int(sub["has_nist_level"].sum()) if n else 0
        pct = 100 * hit / n if n else 0.0
        print(f"  [{g:16s}] rows={n:6d}  has_nist_level={hit:6d}  coverage={pct:5.1f}%")


def main():
    p = argparse.ArgumentParser(description="Build Round-2 NIST manifest (R0.2/R0.3)")
    p.add_argument("--z-min", type=int, default=1)
    p.add_argument("--z-max", type=int, default=26)
    p.add_argument("--n-levels", type=int, default=10, help="n_max for enumeration")
    p.add_argument(
        "--ion-charges", choices=["all", "neutral", "bare"], default="all",
        help="which ion stages to enumerate (default: all stages)",
    )
    p.add_argument("--out", default="data_cache/manifest_nist_z1_26_n10.parquet")
    args = p.parse_args()

    if not NIST_RAW.exists():
        raise FileNotFoundError(f"NIST raw dir not found: {NIST_RAW}")

    df = build_nist_manifest(args.z_min, args.z_max, args.n_levels, args.ion_charges)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"Wrote {len(df)} rows -> {out}")
    print(f"  Z range: {args.z_min}..{args.z_max}   n_max: {args.n_levels}   ions: {args.ion_charges}")
    _print_group_coverage(df)


if __name__ == "__main__":
    main()
