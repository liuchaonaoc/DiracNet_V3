"""Shared Stage C metric computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..constants import ENERGY_UNIT, hartree_to_meV


def build_level_table(rows: list[dict], e_csf_1s: dict[int, float], e_orb_1s: dict[int, float]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["E_csf_exc_meV"] = df.apply(
        lambda r: 0.0
        if r["n"] == 1
        else (r["E_csf_meV"] - float(hartree_to_meV(e_csf_1s[r["Z"]]))),
        axis=1,
    )
    df["E_orb_exc_meV"] = df.apply(
        lambda r: 0.0
        if r["n"] == 1
        else (r["E_orb_meV"] - float(hartree_to_meV(e_orb_1s[r["Z"]]))),
        axis=1,
    )
    df["err_exc_csf_vs_nist_meV"] = (df["E_csf_exc_meV"] - df["level_meV"]).abs()
    df["err_exc_orb_vs_nist_meV"] = (df["E_orb_exc_meV"] - df["level_meV"]).abs()
    df["err_abs_csf_vs_nist_meV"] = (df["E_csf_meV"] - df["E_nist_abs_meV"]).abs()
    return df


def summarize_layer2(
    df: pd.DataFrame,
    *,
    mae_thr_orb: float = 500.0,
    mae_thr_inject: float = 1.0,
) -> dict[str, Any]:
    """Primary metric: E_orb_exc (no inject). Inject sanity: E_csf_exc / abs inject."""
    nist_exc = df[df["has_nist_level"] & (df["n"] > 1)]
    fallback = df[~df["has_nist_level"]]

    def _mae(col: str, sub: pd.DataFrame) -> float:
        if sub.empty:
            return float("nan")
        return float(sub[col].mean())

    mae_orb = _mae("err_exc_orb_vs_nist_meV", nist_exc)
    mae_csf_exc = _mae("err_exc_csf_vs_nist_meV", nist_exc)
    mae_abs = _mae("err_abs_csf_vs_nist_meV", df.loc[df["nist_mask"]])

    has_offdiag = bool(df.get("has_offdiag_coupling", pd.Series([False])).any())
    offdiag_rows = df[df.get("has_offdiag_coupling", False)] if "has_offdiag_coupling" in df.columns else pd.DataFrame()

    layer2_orb = {
        "mae_exc_orb_vs_nist_meV": mae_orb,
        "n_nist_matched": int(nist_exc.shape[0]),
        "n_theory_fallback": int(fallback.shape[0]),
        "fallback_fraction": float(fallback.shape[0] / max(len(df), 1)),
        "verdict": "PASS" if np.isfinite(mae_orb) and mae_orb <= mae_thr_orb else "FAIL",
        "threshold_meV": mae_thr_orb,
        "energy_unit": ENERGY_UNIT,
    }
    layer2_inject = {
        "mae_exc_csf_vs_nist_meV": mae_csf_exc,
        "mae_abs_csf_vs_nist_inject_meV": mae_abs,
        "verdict": "PASS" if np.isfinite(mae_abs) and mae_abs <= mae_thr_inject else "FAIL",
        "threshold_meV": mae_thr_inject,
        "note": "inject_sanity — not a physics-quality metric for single-CSF diagonal fill",
    }
    layer2_multicsf = {}
    if not offdiag_rows.empty:
        layer2_multicsf = {
            "n_spectrum_groups": int(offdiag_rows["spectrum_id"].nunique()) if "spectrum_id" in offdiag_rows else 0,
            "mae_exc_csf_vs_nist_meV": _mae("err_exc_csf_vs_nist_meV", offdiag_rows[offdiag_rows["n"] > 1]),
        }

    return {
        "layer2_orb": layer2_orb,
        "layer2_inject": layer2_inject,
        "layer2_multicsf": layer2_multicsf,
        "layer2": {
            **layer2_orb,
            "inject_sanity_mae_exc_csf_meV": mae_csf_exc,
            "inject_sanity_mae_abs_meV": mae_abs,
        },
    }
