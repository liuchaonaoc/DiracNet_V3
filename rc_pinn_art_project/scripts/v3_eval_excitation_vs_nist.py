#!/usr/bin/env python3
"""R2.2 — Dual-group excitation energy vs NIST (Layer-2b, no NIST inject).

Groups (see prompts/08_evaluation.md §Layer-2b):
  single_valence  — active valence E_orb difference vs ground
  multi_electron  — omega-weighted E_orb + Slater/Racah CI (nist_inject:false)

Outputs:
  EXCITATION_VS_NIST.md   — two sections (single_valence / multi_electron)
  metrics.json            — layer2b summary
  excitation_vs_nist.csv  — per-row detail
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import pandas as pd

from pinn_art.ci.racah_cache import RacahCache, build_cache_from_manifest_rows
from pinn_art.data.dataset import ManifestDataset
from pinn_art.evaluation.excitation_eval import (
    assemble_excitation_table,
    build_prediction_cache,
    resolve_eval_thresholds,
    summarize_group,
    write_excitation_report,
)
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.training.checkpoint import load_params, merge_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.utils.logging import get_logger


def _ensure_racah_cache(manifest: Path, cache_path: Path, k_list: list[int], M_max: int) -> RacahCache:
    if cache_path.exists():
        return RacahCache.load(cache_path)
    log = get_logger()
    log.info("Building Racah cache for multi_electron CI eval -> %s", cache_path)
    df = pd.read_parquet(manifest)
    cache = build_cache_from_manifest_rows(df.to_dict(orient="records"), k_list=k_list, M_max=M_max)
    cache.save(cache_path)
    log.info("Racah cache: %d parent groups", len(cache.parent_ids))
    return cache


def main():
    ap = argparse.ArgumentParser(description="Layer-2b excitation vs NIST (dual group)")
    ap.add_argument("--config", default="configs/v3_stage_a_z1_26_n10.yaml")
    ap.add_argument("--ckpt", default="checkpoints/v3_stage_a_z1_26_n10/stage_a_last.msgpack")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-configs", type=int, default=None, help="Limit unique level_config forward passes")
    ap.add_argument("--nist-only", action="store_true", help="Only evaluate rows with has_nist_level")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    log = get_logger()
    manifest = Path(args.manifest) if args.manifest else ROOT / cfg.dataset.manifest
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg.training.log_dir
    ckpt = ROOT / args.ckpt

    th = resolve_eval_thresholds(cfg)
    ci_cfg = getattr(cfg, "ci", None)
    k_list = list(getattr(ci_cfg, "k_list", [0, 1, 2])) if ci_cfg else [0, 1, 2]
    M_max = int(getattr(ci_cfg, "M_max", 8))
    racah_path = ROOT / getattr(ci_cfg, "racah_cache", "data_cache/racah_cache_z1_26_n10.npz")
    racah_cache = _ensure_racah_cache(manifest, racah_path, k_list, M_max)
    k_tuple = tuple(k_list)

    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    mdf = ds.df.copy()
    if args.nist_only:
        mdf = mdf[mdf["has_nist_level"]].copy()

    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max),
        int(cfg.grid.n_grid), str(cfg.grid.scheme),
    )
    model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    ckpt_params = load_params(ckpt)
    # Stage-A ckpt lacks CI params; merge into a CI-capable template for eval.
    from pinn_art.models.pinn_art_model import _dummy_batch
    n_orb = int(cfg.model.n_orb_max)
    n_csf = int(getattr(cfg.model, "n_csf_max", 8))
    dummy = _dummy_batch(n_orb, n_csf)
    params_ci = model.init(jax.random.PRNGKey(1), dummy, grid, train=False, return_ci=True)
    params = merge_params(params_ci, ckpt_params)
    log.info("Loaded checkpoint: %s", ckpt)

    # Unique configs to forward (dedupe expensive inference)
    cfg_keys = list({
        (int(r["Z"]), int(r["ion_charge"]), str(r["level_config"]))
        for _, r in mdf.iterrows()
    })
    ground_keys = list({
        (int(r["Z"]), int(r["ion_charge"]), str(r["parent_config"]))
        for _, r in mdf[mdf["is_ground"]].iterrows()
    })
    all_keys = list({*cfg_keys, *ground_keys})
    if args.max_configs:
        all_keys = all_keys[: args.max_configs]

    row_lookup = {
        (int(r["Z"]), int(r["ion_charge"]), str(r["level_config"])): int(i)
        for i, r in mdf.iterrows()
    }
    n_csf = int(getattr(cfg.model, "n_csf_max", 8))

    t0 = time.perf_counter()
    log.info("Forward pass: %d unique configs (batched, use_ci=True)...",
             len(all_keys))
    # 一次性 return_ci=True 拿到 E_orb (单电子) + E_csf (多电子),
    # 避免之前双 forward 的 2x 开销
    pred_cache = build_prediction_cache(
        all_keys, ds, model, params, grid,
        n_csf_max=n_csf, racah_cache=racah_cache, k_list=k_tuple,
        use_ci=True, row_lookup=row_lookup,
    )

    ground_cache = {
        k: pred_cache[k] for k in ground_keys if k in pred_cache
    }

    df = assemble_excitation_table(mdf, pred_cache, ground_cache=ground_cache)
    sv_sum = summarize_group(df, "single_valence", th)
    me_sum = summarize_group(df, "multi_electron", th)
    metrics = {
        "checkpoint": str(ckpt),
        "config": str(args.config),
        "manifest": str(manifest),
        "nist_inject": False,
        "n_rows_evaluated": len(df),
        "forward_unique_configs": len(all_keys),
        "forward_elapsed_s": time.perf_counter() - t0,
        "thresholds": {
            "single_valence": {"mae_meV": th.single_mae_meV, "rel_err": th.single_rel_err},
            "multi_electron": {"mae_meV": th.multi_mae_meV, "rel_err": th.multi_rel_err},
        },
        "layer2b": {
            "single_valence": sv_sum,
            "multi_electron": me_sum,
        },
    }

    md_path, json_path = write_excitation_report(
        df, metrics, out_dir, ckpt=str(args.ckpt), config=str(args.config),
    )
    log.info("Wrote %s", md_path)
    log.info("Wrote %s", json_path)
    for g in ("single_valence", "multi_electron"):
        m = metrics["layer2b"][g]
        log.info(
            "[%s] n=%d  MAE=%.2f meV  median=%.2f meV  pass=%.1f%%  %s",
            g, m["n"], m["mae_meV"], m["median_meV"], 100 * m["pass_fraction"], m["verdict"],
        )


if __name__ == "__main__":
    main()
