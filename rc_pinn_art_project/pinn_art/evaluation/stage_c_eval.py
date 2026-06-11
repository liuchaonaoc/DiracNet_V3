"""Stage C full-manifest evaluation loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import pandas as pd

from ..ci.racah_cache import RacahCache
from ..constants import hartree_to_meV
from ..data.collate import collate_batches
from ..data.dataset import ManifestDataset
from ..models.pinn_art_model import build_model_and_params
from ..training.checkpoint import load_params, merge_params
from ..training.gate_a import parse_n_l_from_level_config
from ..utils.grid import make_radial_grid
from .stage_c_metrics import build_level_table, summarize_layer2


def run_stage_c_evaluation(
    cfg,
    ckpt: Path,
    manifest: Path,
    out_dir: Path,
    *,
    root: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    racah_cache = RacahCache.load((root or Path(".")) / cfg.ci.racah_cache)
    k_list = tuple(int(k) for k in cfg.ci.k_list)
    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    mdf = ds.df
    grid = make_radial_grid(
        float(cfg.grid.r_min),
        float(cfg.grid.r_max),
        int(cfg.grid.n_grid),
        str(cfg.grid.scheme),
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    params = merge_params(params, load_params(ckpt))

    gate_cfg = getattr(cfg.stage_c, "gate", cfg.stage_c)
    mae_thr_orb = float(getattr(gate_cfg, "mae_exc_orb_nist_meV_threshold", 500.0))
    mae_thr_inject = float(getattr(gate_cfg, "mae_inject_sanity_meV_threshold", 1.0))

    rows = []
    e_csf_1s: dict[int, float] = {}
    e_orb_1s: dict[int, float] = {}

    for i in range(len(ds)):
        batch = collate_batches(
            [ds[i]],
            n_csf_max=int(cfg.model.n_csf_max),
            racah_cache=racah_cache,
            k_list=k_list,
        )
        out = model.apply(params, batch, grid, train=False, return_ci=True)
        Z = int(batch["Z"][0])
        level_cfg = str(mdf.iloc[i]["level_config"])
        n, l = parse_n_l_from_level_config(level_cfg)

        E_orb_ha = float(out["E_orb"][0, 0])
        E_csf_ha = float(out["E_csf"][0, 0]) if out.get("E_csf") is not None else float("nan")
        if n == 1 and l == 0:
            e_csf_1s[Z] = E_csf_ha
            e_orb_1s[Z] = E_orb_ha

        import numpy as np

        H = out.get("H")
        has_offdiag = False
        if H is not None:
            H0 = np.asarray(H[0])
            m = int(np.sum(np.asarray(batch["csf_mask"][0])))
            if m > 1:
                blk = H0[:m, :m]
                has_offdiag = float(np.max(np.abs(blk - np.diag(np.diag(blk))))) > 1e-12

        rows.append({
            "row": i,
            "Z": Z,
            "element": str(mdf.iloc[i].get("element", f"Z{Z}")),
            "level_config": level_cfg,
            "n": n,
            "l": l,
            "has_nist_level": bool(mdf.iloc[i].get("has_nist_level", False)),
            "nist_mask": bool(batch["nist_mask"][0, 0]),
            "diag_source": int(out["diag_source"][0, 0]) if out.get("diag_source") is not None else -1,
            "level_meV": float(mdf.iloc[i]["level_meV"]),
            "E_orb_meV": float(hartree_to_meV(E_orb_ha)),
            "E_csf_meV": float(hartree_to_meV(E_csf_ha)),
            "E_nist_abs_meV": float(hartree_to_meV(float(batch["E_nist"][0, 0]))),
            "spectrum_id": str(mdf.iloc[i].get("spectrum_id", "")),
            "has_offdiag_coupling": has_offdiag,
        })

    df = build_level_table(rows, e_csf_1s, e_orb_1s)
    summary = summarize_layer2(df, mae_thr_orb=mae_thr_orb, mae_thr_inject=mae_thr_inject)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stage_c_levels.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    metrics = {
        **summary,
        "checkpoint": str(ckpt),
        "manifest": str(manifest),
        "nist_inject": bool(cfg.ci.nist_inject),
    }
    return metrics, df, csv_path
