"""Lightweight anchor checks during curriculum training (H / He guard)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..data.collate import collate_batches
from ..data.dataset import ManifestDataset
from ..utils.grid import RadialGrid

HARTREE_EV = 27.211386245988
NIST_H_2S_EV = 10.19881


@dataclass
class AnchorMetrics:
    h_exc_ev: float
    h_exc_err_meV: float
    v_r1_diff_ha: float
    e_orb_1s_ha: float
    e_orb_2s_ha: float
    ok_h_exc: bool
    ok_v_r1: bool
    ok: bool
    message: str


def _forward_level(ds, model, params, grid, Z, ion, level_config, n_csf_max):
    hits = ds.df[
        (ds.df["Z"] == Z)
        & (ds.df["ion_charge"] == ion)
        & (ds.df["level_config"] == level_config)
    ].index
    if len(hits) == 0:
        return None
    batch = collate_batches([ds[int(hits[0])]], n_csf_max=n_csf_max)
    out = model.apply(params, batch, grid, train=False)
    return batch, out


def hydrogen_anchor_metrics(
    model,
    params,
    grid: RadialGrid,
    manifest_path: str | Path,
    *,
    n_orb_max: int = 16,
    n_csf_max: int = 8,
    max_h_exc_meV: float = 100.0,
    max_v_r1_diff_ha: float = 0.05,
) -> AnchorMetrics:
    """Forward H 1s/2s; return guard metrics (no file I/O)."""
    ds = ManifestDataset(manifest_path, n_orb_max=n_orb_max, n_csf_max=n_csf_max)
    r = np.asarray(grid.r, dtype=np.float64)
    e_levels: dict[str, float] = {}
    v_r1_diff = float("nan")

    for lc in ("1s1", "2s1"):
        res = _forward_level(ds, model, params, grid, 1, 0, lc, n_csf_max)
        if res is None:
            return AnchorMetrics(
                float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                False, False, False, f"missing H config {lc}",
            )
        batch, out = res
        E_orb = np.asarray(out["E_orb"][0], dtype=np.float64)
        mask = np.asarray(batch["orb_mask"][0], dtype=bool)
        active = int(np.where(mask)[0][-1])
        e_levels[lc] = float(E_orb[active])
        if lc == "2s1":
            V = np.asarray(out["V"][0], dtype=np.float64)
            i = int(np.argmin(np.abs(r - 1.026)))
            v_r1_diff = float(V[i] + 1.0 / r[i])

    exc_ev = (e_levels["2s1"] - e_levels["1s1"]) * HARTREE_EV
    err_meV = (exc_ev - NIST_H_2S_EV) * 1000.0
    ok_h = abs(err_meV) <= max_h_exc_meV
    ok_v = abs(v_r1_diff) <= max_v_r1_diff_ha
    ok = ok_h and ok_v
    msg = (
        f"H exc={exc_ev:.4f} eV err={err_meV:+.1f} meV  "
        f"V(r~1) diff={v_r1_diff:+.4f} Ha  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return AnchorMetrics(
        exc_ev, err_meV, v_r1_diff,
        e_levels["1s1"], e_levels["2s1"],
        ok_h, ok_v, ok, msg,
    )


def metrics_from_train_state(state, model, grid, manifest_path, **kw) -> AnchorMetrics:
    return hydrogen_anchor_metrics(model, state.params, grid, manifest_path, **kw)
