"""Layer-2b dual-group excitation energy evaluation (Stage A Round 2, R2.2).

Groups (see ``prompts/08_evaluation.md`` §Layer-2b):
  single_valence  — E_exc from active valence ``E_orb`` difference vs ground
  multi_electron  — E_exc from config total energy (omega-weighted ``E_orb`` +
                    Slater/Racah CI, ``nist_inject:false``) vs ground
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..constants import ENERGY_UNIT, hartree_to_meV


@dataclass(frozen=True)
class EvalThresholds:
    single_mae_meV: float = 50.0
    single_rel_err: float = 0.002
    multi_mae_meV: float = 500.0
    multi_rel_err: float = 0.01


def resolve_eval_thresholds(cfg) -> EvalThresholds:
    ev = getattr(getattr(cfg, "stage_a", None), "eval", None)
    sv = getattr(ev, "single_valence", ev) if ev else None
    me = getattr(ev, "multi_electron", ev) if ev else None
    return EvalThresholds(
        single_mae_meV=float(getattr(sv, "mae_meV_threshold", 50.0)) if sv else 50.0,
        single_rel_err=float(getattr(sv, "rel_err_threshold", 0.002)) if sv else 0.002,
        multi_mae_meV=float(getattr(me, "mae_meV_threshold", 500.0)) if me else 500.0,
        multi_rel_err=float(getattr(me, "rel_err_threshold", 0.01)) if me else 0.01,
    )


def find_active_orb_index(
    shell_table: np.ndarray,
    orb_mask: np.ndarray,
    n_active: int,
    l_active: int,
    *,
    twice_j: int | None = None,
) -> int:
    """Index of the active valence orbital in the collated batch."""
    if n_active >= 1 and l_active >= 0:
        for i in range(len(orb_mask)):
            if not orb_mask[i]:
                continue
            n, l, tj, occ = (int(shell_table[i, j]) for j in range(4))
            if n == n_active and l == l_active and occ > 0:
                if twice_j is not None and tj != twice_j:
                    continue
                return i
    # fallback: last occupied orbital
    for i in range(len(orb_mask) - 1, -1, -1):
        if orb_mask[i] and shell_table[i, 3] > 0:
            return i
    return 0


def config_total_energy_meV(
    E_orb_ha: np.ndarray,
    omega: np.ndarray,
    orb_mask: np.ndarray,
    *,
    E_csf_ha: float | None = None,
    use_ci: bool = False,
) -> float:
    """Total config energy (meV): omega-weighted E_orb sum, or CI E_csf when valid."""
    if use_ci and E_csf_ha is not None and np.isfinite(E_csf_ha):
        return float(hartree_to_meV(E_csf_ha))
    m = orb_mask.astype(bool)
    w = omega.astype(np.float64) * m
    if w.sum() <= 0:
        return float(hartree_to_meV(E_orb_ha[0]))
    return float(hartree_to_meV(np.sum(E_orb_ha * w)))


def valence_excitation_meV(
    E_orb_ha: np.ndarray,
    shell_table: np.ndarray,
    orb_mask: np.ndarray,
    n_active: int,
    l_active: int,
    *,
    twice_j: int | None = None,
    E_orb_ground_ha: np.ndarray | None = None,
    ground_shell_table: np.ndarray | None = None,
    ground_orb_mask: np.ndarray | None = None,
    ground_n_active: int = 1,
    ground_l_active: int = 0,
    ground_twice_j: int | None = None,
) -> float:
    """Active valence ``E_orb`` difference (meV) for ``single_valence`` rows."""
    idx = find_active_orb_index(shell_table, orb_mask, n_active, l_active, twice_j=twice_j)
    e_exc = float(hartree_to_meV(E_orb_ha[idx]))
    if E_orb_ground_ha is None:
        return e_exc
    g_idx = find_active_orb_index(
        ground_shell_table, ground_orb_mask,
        ground_n_active, ground_l_active, twice_j=ground_twice_j,
    )
    return e_exc - float(hartree_to_meV(E_orb_ground_ha[g_idx]))


def build_prediction_cache(
    unique_keys: list[tuple[int, int, str]],
    ds,
    model,
    params,
    grid,
    *,
    n_csf_max: int,
    racah_cache=None,
    k_list: tuple[int, ...] | None = None,
    use_ci: bool = False,
    row_lookup: dict[tuple[int, int, str], int] | None = None,
    batch_size: int = 64,
) -> dict[tuple[int, int, str], dict]:
    """Run forward pass once per unique (Z, ion, level_config).

    Performance notes (R2.2 speedup, see PROGRESS §18):
      - 一次性用 return_ci=True 跑, 同时得到 E_orb (单电子) 和 E_csf (多电子)
        避免单/多电子两轮独立 forward 浪费 2x 时间
      - batch_size 个 config 一批, 一次 model.apply -> 一次 JAX dispatch
        避免 N 个 config 各自一次 dispatch
      - 外层用 jax.jit 包裹整批 apply, 编译一次反复调用
    """
    import jax
    from ..data.collate import collate_batches

    # 1) 收集所有 idx + key
    key_idx: list[tuple[tuple[int, int, str], int]] = []
    for key in unique_keys:
        if row_lookup is not None:
            key_idx.append((key, row_lookup[key]))
        else:
            Z, ion, cfg = key
            hits = ds.df[
                (ds.df["Z"] == Z)
                & (ds.df["ion_charge"] == ion)
                & (ds.df["level_config"] == cfg)
            ].index
            if len(hits) == 0:
                continue
            key_idx.append((key, int(hits[0])))

    if not key_idx:
        return {}

    # 2) 固定 batch_size + pad, 这样 JAX 一次 trace 后所有 chunk 复用
    n_total = len(key_idx)
    n_full = (n_total // batch_size) * batch_size

    @jax.jit
    def _batched_apply(batch_dict):
        return model.apply(params, batch_dict, grid, train=False, return_ci=True)

    cache: dict[tuple[int, int, str], dict] = {}
    t0 = time.perf_counter()
    # 3) 完整 chunk (B == batch_size) 用 jit 加速
    for chunk_start in range(0, n_full, batch_size):
        chunk = key_idx[chunk_start: chunk_start + batch_size]
        items = [ds[idx] for _, idx in chunk]
        batch = collate_batches(
            items,
            n_csf_max=n_csf_max,
            racah_cache=racah_cache,
            k_list=k_list if racah_cache else None,
        )
        if use_ci:
            n_orb = batch["omega"].shape[1]
            csf_to_orb = np.zeros((batch_size, n_csf_max, n_orb), dtype=np.float32)
            om = np.asarray(batch["omega"], dtype=np.float32)
            mask = np.asarray(batch["orb_mask"], dtype=bool)
            csf_to_orb[:, 0, :] = om * mask
            batch["csf_to_orb"] = jax.numpy.asarray(csf_to_orb)

        out = _batched_apply(batch)
        _accumulate_cache(out, batch, chunk, cache, use_ci)

        if (chunk_start // batch_size) % 5 == 0:
            elapsed = time.perf_counter() - t0
            done = chunk_start + batch_size
            rate = done / max(elapsed, 1e-6)
            eta_s = (n_total - done) / max(rate, 1e-6)
            print(f"  eval forward: {done}/{n_total} configs  "
                  f"({rate:.1f}/s, ETA {eta_s:.0f}s)", flush=True)

    # 4) 剩余不完整 chunk: 单跑, 接受 re-trace 开销
    if n_total > n_full:
        chunk = key_idx[n_full:]
        items = [ds[idx] for _, idx in chunk]
        batch = collate_batches(
            items,
            n_csf_max=n_csf_max,
            racah_cache=racah_cache,
            k_list=k_list if racah_cache else None,
        )
        if use_ci:
            n_orb = batch["omega"].shape[1]
            csf_to_orb = np.zeros((len(chunk), n_csf_max, n_orb), dtype=np.float32)
            om = np.asarray(batch["omega"], dtype=np.float32)
            mask = np.asarray(batch["orb_mask"], dtype=bool)
            csf_to_orb[:, 0, :] = om * mask
            batch["csf_to_orb"] = jax.numpy.asarray(csf_to_orb)
        out = model.apply(params, batch, grid, train=False, return_ci=True)
        _accumulate_cache(out, batch, chunk, cache, use_ci)

    return cache


def _accumulate_cache(out, batch, chunk, cache, use_ci):
    """把一个 chunk 的 forward 结果拆分成 cache 字典."""
    E_orb_b = np.asarray(out["E_orb"])  # [B, n_orb]
    E_csf_b = np.asarray(out["E_csf"]) if out.get("E_csf") is not None else None
    omega_b = np.asarray(batch["omega"])
    mask_b = np.asarray(batch["orb_mask"])
    shell_b = np.asarray(batch["shell_table"])
    for j, (key, _) in enumerate(chunk):
        E_csf = float(E_csf_b[j, 0]) if (use_ci and E_csf_b is not None) else float("nan")
        cache[key] = {
            "E_orb_ha": E_orb_b[j],
            "E_csf_ha": E_csf,
            "omega": omega_b[j],
            "orb_mask": mask_b[j],
            "shell_table": shell_b[j],
        }


def assemble_excitation_table(
    mdf: pd.DataFrame,
    pred_cache: dict[tuple[int, int, str], dict],
    *,
    ground_cache: dict[tuple[int, int, str], dict],
) -> pd.DataFrame:
    """Join predictions with manifest; compute per-row excitation energies."""
    rows = []
    for i, r in mdf.iterrows():
        Z = int(r["Z"])
        ion = int(r["ion_charge"])
        cfg = str(r["level_config"])
        parent = str(r["parent_config"])
        group = str(r.get("group", "single_valence"))
        key = (Z, ion, cfg)
        gkey = (Z, ion, str(r.get("parent_config", cfg)))
        if key not in pred_cache:
            continue
        pred = pred_cache[key]
        ground = ground_cache.get(gkey) or pred_cache.get(
            (Z, ion, parent)
        )
        is_ground = bool(r.get("is_ground", False))
        n_act = int(r.get("n_active", -1))
        l_act = int(r.get("l_active", -1))
        J = r.get("J", 0.5)
        twice_j = None if (pd.isna(J) or group == "multi_electron") else int(round(float(J) * 2))

        if group == "single_valence":
            if is_ground:
                e_pred = 0.0
            elif ground is not None:
                g_row = mdf[
                    (mdf["Z"] == Z) & (mdf["ion_charge"] == ion)
                    & (mdf["parent_config"] == parent) & (mdf["is_ground"] == True)
                ]
                gn = int(g_row.iloc[0]["n_active"]) if len(g_row) else 1
                gl = int(g_row.iloc[0]["l_active"]) if len(g_row) else 0
                gtj = None
                if len(g_row) and not pd.isna(g_row.iloc[0]["J"]):
                    gtj = int(round(float(g_row.iloc[0]["J"]) * 2))
                e_pred = valence_excitation_meV(
                    pred["E_orb_ha"], pred["shell_table"], pred["orb_mask"],
                    n_act, l_act, twice_j=twice_j,
                    E_orb_ground_ha=ground["E_orb_ha"],
                    ground_shell_table=ground["shell_table"],
                    ground_orb_mask=ground["orb_mask"],
                    ground_n_active=gn, ground_l_active=gl, ground_twice_j=gtj,
                )
            else:
                e_pred = 0.0
        else:
            e_tot = config_total_energy_meV(
                pred["E_orb_ha"], pred["omega"], pred["orb_mask"],
                E_csf_ha=pred.get("E_csf_ha"), use_ci=True,
            )
            if is_ground or ground is None:
                e_pred = 0.0
            else:
                e_g = config_total_energy_meV(
                    ground["E_orb_ha"], ground["omega"], ground["orb_mask"],
                    E_csf_ha=ground.get("E_csf_ha"), use_ci=True,
                )
                e_pred = e_tot - e_g

        nist_exc = float(r["level_meV"]) if bool(r.get("has_nist_level", False)) else float("nan")
        err = abs(e_pred - nist_exc) if np.isfinite(nist_exc) and not is_ground else float("nan")
        rel = err / max(abs(nist_exc), 1e-6) if np.isfinite(err) else float("nan")

        rows.append({
            "row": int(i),
            "Z": Z,
            "ion_charge": ion,
            "element": str(r.get("element", f"Z{Z}")),
            "parent_config": parent,
            "level_config": cfg,
            "group": group,
            "is_ground": is_ground,
            "has_nist_level": bool(r.get("has_nist_level", False)),
            "nist_exc_meV": nist_exc,
            "pred_exc_meV": float(e_pred),
            "err_meV": err,
            "rel_err": rel,
            "n_active": n_act,
            "l_active": l_act,
            "J": J,
        })
    return pd.DataFrame(rows)


def summarize_group(
    df: pd.DataFrame,
    group: str,
    th: EvalThresholds,
) -> dict[str, Any]:
    """MAE / median / coverage / pass fraction for one evaluation group."""
    sub = df[(df["group"] == group) & (~df["is_ground"]) & df["has_nist_level"]]
    n = len(sub)
    if n == 0:
        return {"group": group, "n": 0, "n_total_excited": 0,
                "mae_meV": float("nan"), "median_meV": float("nan"),
                "median_rel_err": float("nan"), "coverage": 0.0, "pass_fraction": 0.0,
                "mae_threshold_meV": float("nan"),
                "rel_err_threshold": float("nan"),
                "verdict": "FAIL"}
    mae = float(sub["err_meV"].mean())
    med = float(sub["err_meV"].median())
    med_rel = float(sub["rel_err"].median())
    mae_thr = th.single_mae_meV if group == "single_valence" else th.multi_mae_meV
    rel_thr = th.single_rel_err if group == "single_valence" else th.multi_rel_err
    pass_mask = (sub["err_meV"] < mae_thr) | (sub["rel_err"] < rel_thr)
    total_excited = int((df["group"] == group).sum() - (df[df["group"] == group]["is_ground"].sum()))
    return {
        "group": group,
        "n": n,
        "n_total_excited": total_excited,
        "coverage": float(n / max(total_excited, 1)),
        "mae_meV": mae,
        "median_meV": med,
        "median_rel_err": med_rel,
        "pass_fraction": float(pass_mask.mean()),
        "mae_threshold_meV": mae_thr,
        "rel_err_threshold": rel_thr,
        "verdict": "PASS" if pass_mask.mean() >= 0.5 else "FAIL",
    }


def write_excitation_report(
    df: pd.DataFrame,
    metrics: dict[str, Any],
    out_dir,
    *,
    ckpt: str,
    config: str,
) -> tuple[Any, Any]:
    """Write ``EXCITATION_VS_NIST.md`` and return (md_path, json_path)."""
    import json
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "excitation_vs_nist.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    lines = [
        "# 激发能 vs NIST（Layer-2b 双组）",
        "",
        f"- Checkpoint: `{ckpt}`",
        f"- Config: `{config}`",
        f"- 能量单位: **{ENERGY_UNIT}**",
        f"- 纪律: **NIST 不注入**（`nist_inject: false`）",
        "",
        "## 汇总",
        "",
        "| 组 | N (NIST) | MAE (meV) | 中位数 (meV) | 中位相对误差 | 覆盖率 | 达标占比 | 判定 |",
        "|----|----------|-----------|--------------|--------------|--------|----------|------|",
    ]
    for g in ("single_valence", "multi_electron"):
        m = metrics["layer2b"][g]
        lines.append(
            f"| `{g}` | {m['n']} | {m['mae_meV']:.2f} | {m['median_meV']:.2f} | "
            f"{m['median_rel_err']:.4f} | {100*m['coverage']:.1f}% | "
            f"{100*m['pass_fraction']:.1f}% | **{m['verdict']}** |"
        )

    for g in ("single_valence", "multi_electron"):
        sub = df[(df["group"] == g) & (~df["is_ground"]) & df["has_nist_level"]]
        if sub.empty:
            continue
        lines.extend([
            "",
            f"## `{g}` — 按元素 MAE (meV)",
            "",
            "| Z | 元素 | N | MAE | 中位数 |",
            "|---|------|---|-----|--------|",
        ])
        for Z, gdf in sub.groupby("Z"):
            el = gdf["element"].iloc[0]
            lines.append(
                f"| {int(Z)} | {el} | {len(gdf)} | {gdf['err_meV'].mean():.2f} | "
                f"{gdf['err_meV'].median():.2f} |"
            )
        worst = sub.nlargest(8, "err_meV")
        lines.extend([
            "",
            f"### `{g}` — 最大偏差",
            "",
            worst[["Z", "element", "level_config", "pred_exc_meV", "nist_exc_meV", "err_meV"]]
            .to_markdown(index=False, floatfmt=".2f"),
        ])

    lines.extend(["", f"完整 CSV: `{csv_path.name}`", ""])
    md_path = out_dir / "EXCITATION_VS_NIST.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path = out_dir / "metrics.json"
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    return md_path, json_path
