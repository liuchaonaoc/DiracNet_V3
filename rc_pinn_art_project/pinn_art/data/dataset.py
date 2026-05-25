"""Parquet manifest dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config_parser import encode_config_to_array, parse_config_string


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


class ManifestDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        n_orb_max: int = 16,
        n_csf_max: int = 32,
        align_ground: bool = True,
    ) -> None:
        self.df = load_manifest(manifest_path)
        self.n_orb_max = n_orb_max
        self.n_csf_max = n_csf_max
        self.align_ground = align_ground

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        Z = int(row["Z"])
        ion = int(row.get("ion_charge", 0))
        nele = int(row.get("nele", Z - ion))
        parent = str(row.get("parent_config", row.get("level_config", "1s1")))
        level = str(row.get("level_config", parent))
        # Stage A 类氢/单电子体系：level_config 表示该电子真实状态（n 不同）
        # 多电子体系：仍用 level_config（包含核心 + 激发轨道）
        shells = parse_config_string(level)
        shell_table = encode_config_to_array(shells, self.n_orb_max)

        kappas = []
        omegas = []
        orb_mask = []
        for i in range(self.n_orb_max):
            if i < len(shells):
                sh = shells[i]
                kappas.append(sh.kappa)
                omegas.append(float(sh.occ))
                orb_mask.append(True)
            else:
                kappas.append(0)
                omegas.append(0.0)
                orb_mask.append(False)

        level_eV = float(row.get("level_eV", np.nan))
        has_nist = bool(row.get("has_nist_level", np.isfinite(level_eV)))
        if np.isfinite(level_eV):
            E_ha = level_eV / 27.211386245988
        else:
            E_ha = np.nan

        return {
            "Z": Z,
            "ion_charge": ion,
            "nele": nele,
            "parent_config": parent,
            "level_config": level,
            "shell_table": shell_table,
            "kappa": np.array(kappas, dtype=np.int32),
            "omega": np.array(omegas, dtype=np.float32),
            "orb_mask": np.array(orb_mask, dtype=bool),
            "J": float(row.get("J", 0.5)),
            "parity": int(row.get("parity", 0)),
            "E_nist_scalar": E_ha,
            "nist_mask_scalar": has_nist,
        }
