#!/usr/bin/env python3
"""Compare Li V/P/Q: cFAC (Li_*.dat) vs PINN-ART DeepONet.

cFAC file naming vs quantum numbers (read headers, do not trust filenames):
  Li_V.dat   — i, r, Vn, Vc, U, u, w  →  V(r) = Vc + U  (cols 3+4)
  Li_1s.dat  — n=1, κ=-1  → 1s
  Li_s1.dat  — n=2, κ=-1  → 2s
  Li_s2.dat  — n=3, κ=-1  → 3s

    cd DiracNet_V3/rc_pinn_art_project && export PYTHONPATH=.
    python scripts/v3_compare_fac_li_pqv.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRAC_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pinn_art.constants import HARTREE_TO_EV
from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.hydrogenic import cosine_signed
from pinn_art.training.checkpoint import load_params
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid

OUT_DIR = ROOT / "logs" / "fac_vs_pinn_li"
FAC_DIR = DIRAC_ROOT

_HEADER_INT = re.compile(r"^\s*#\s*n\s*=\s*(\d+)", re.I)
_HEADER_KAPPA = re.compile(r"^\s*#\s*kappa\s*=\s*(-?\d+)", re.I)
_HEADER_E = re.compile(r"^\s*#\s*energy\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", re.I)


def parse_fac_dat(path: Path) -> tuple[dict, np.ndarray]:
    """Parse cFAC ASCII table; return header dict + float array [N, ncol]."""
    meta: dict = {}
    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                m = _HEADER_INT.search(line)
                if m:
                    meta["n"] = int(m.group(1))
                m = _HEADER_KAPPA.search(line)
                if m:
                    meta["kappa"] = int(m.group(1))
                m = _HEADER_E.search(line)
                if m:
                    meta["energy_Ha"] = float(m.group(1))
                if "Navg" in line:
                    meta.setdefault("comments", []).append(line.strip())
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"no data rows in {path}")
    return meta, np.asarray(rows, dtype=np.float64)


def load_fac_potential(path: Path) -> dict:
    meta, arr = parse_fac_dat(path)
    r = arr[:, 1]
    vn = arr[:, 2]
    vc = arr[:, 3]
    u = arr[:, 4]
    v_total = vc + u
    return {"meta": meta, "r": r, "Vn": vn, "Vc": vc, "U": u, "V": v_total}


def load_fac_wavefunc(path: Path) -> dict:
    meta, arr = parse_fac_dat(path)
    r = arr[:, 1]
    vc_r = arr[:, 2]
    u_r = arr[:, 3]
    p = arr[:, 4]
    q = arr[:, 5]
    v = (vc_r + u_r) / np.maximum(r, 1e-30)
    label = f"{meta.get('n', '?')}s" if meta.get("kappa") == -1 else f"n={meta.get('n')}κ={meta.get('kappa')}"
    return {
        "meta": meta,
        "label": label,
        "r": r,
        "P": p,
        "Q": q,
        "V": v,
        "energy_Ha": meta.get("energy_Ha"),
    }


def trapz_weights(r: np.ndarray) -> np.ndarray:
    n = len(r)
    w = np.zeros(n)
    if n > 1:
        w[0] = 0.5 * (r[1] - r[0])
        w[-1] = 0.5 * (r[-1] - r[-2])
    if n > 2:
        w[1:-1] = 0.5 * (r[2:] - r[:-2])
    return w


def normalize_pq(r: np.ndarray, p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    w = trapz_weights(r)
    norm = float(np.sum((p * p + q * q) * w))
    s = np.sqrt(max(norm, 1e-30))
    return p / s, q / s, norm


def interp_to(r_tgt: np.ndarray, r_src: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    """Linear interp; clip to src range instead of NaN tails."""
    r_min, r_max = float(r_src[0]), float(r_src[-1])
    rt = np.clip(r_tgt, r_min, r_max)
    return np.interp(rt, r_src, y_src)


def integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def align_sign(p: np.ndarray, q: np.ndarray, p_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Flip (P,Q) if that improves overlap with reference P."""
    num = float(np.sum(p * p_ref))
    if num < 0:
        return -p, -q, -1.0
    return p, q, 1.0


def cosine_on_grid(p: np.ndarray, q: np.ndarray, pr: np.ndarray, qr: np.ndarray, r: np.ndarray) -> float:
    w = trapz_weights(r)
    num = float(np.sum(w * (p * pr + q * qr)))
    den = np.sqrt(
        float(np.sum(w * (p * p + q * q))) * float(np.sum(w * (pr * pr + qr * qr)))
    )
    return num / max(den, 1e-30)


def count_nodes(p: np.ndarray, frac: float = 0.9) -> int:
    n = int(len(p) * frac)
    core = p[:n]
    return int(np.sum(core[1:] * core[:-1] < 0))


def find_manifest_idx(manifest: Path, z: int, ion: int, level: str) -> int:
    df = pd.read_parquet(manifest)
    hits = df[(df["Z"] == z) & (df["ion_charge"] == ion) & (df["level_config"] == level)]
    if len(hits) == 0:
        raise KeyError(f"manifest missing Z={z} ion={ion} level={level!r}")
    return int(hits.index[0])


def pinn_forward(
    cfg_path: Path,
    ckpt_path: Path,
    manifest_row: int,
) -> dict:
    cfg = load_config(cfg_path)
    grid = make_radial_grid(
        float(cfg.grid.r_min), float(cfg.grid.r_max), int(cfg.grid.n_grid), str(cfg.grid.scheme)
    )
    manifest = ROOT / cfg.dataset.manifest
    ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max))
    model, _ = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
    params = load_params(ckpt_path)
    batch = collate_batches([ds[manifest_row]], n_csf_max=int(getattr(cfg.model, "n_csf_max", 8)))
    out = model.apply(params, batch, grid, train=False, return_ci=False)

    wf = out["wavefunctions"]
    shells = []
    for a in range(int(cfg.model.n_orb_max)):
        if not bool(np.asarray(batch["orb_mask"][0, a])):
            continue
        n = int(np.asarray(batch["shell_table"][0, a, 0]))
        l = int(np.asarray(batch["shell_table"][0, a, 1]))
        kappa = int(np.asarray(batch["kappa"][0, a]))
        omega = float(np.asarray(batch["omega"][0, a]))
        shells.append({
            "idx": a,
            "n": n,
            "l": l,
            "kappa": kappa,
            "omega": omega,
            "label": f"{n}s" if l == 0 else f"{n}{'spdfghi'[l]}",
            "P": np.asarray(wf["P"][0, a]),
            "Q": np.asarray(wf["Q"][0, a]),
            "E_orb_Ha": float(np.asarray(out["E_orb"][0, a])),
        })

    row_data = ds[manifest_row]
    return {
        "grid": grid,
        "r": np.asarray(grid.r),
        "V": np.asarray(out["V"][0]),
        "shells": shells,
        "level_config": str(row_data.get("level_config", "")),
        "Z": int(batch["Z"][0]),
    }


def fac_energy_to_ev(e_header: float | None) -> float:
    """FAC WaveFuncTable header energy — Li exports use eV (|E| ≲ 100)."""
    if e_header is None:
        return np.nan
    if abs(e_header) <= 100.0:
        return float(e_header)
    return float(e_header * HARTREE_TO_EV)


def pick_shell(shells: list[dict], n: int, l: int = 0) -> dict | None:
    for sh in shells:
        if sh["n"] == n and sh["l"] == l:
            return sh
    return None


def discover_fac_wavefuncs(fac_dir: Path) -> list[dict]:
    """Load all Li_*.dat wavefunction tables (exclude Li_V.dat), sorted by n."""
    wfs = []
    for path in sorted(fac_dir.glob("Li_*.dat")):
        if path.name.lower() in ("li_v.dat",):
            continue
        try:
            wfs.append({**load_fac_wavefunc(path), "path": path})
        except (ValueError, KeyError):
            continue
    wfs.sort(key=lambda w: int(w["meta"].get("n", 99)))
    return wfs


def has_fac_1s(fac_wfs: list[dict]) -> bool:
    return any(int(w["meta"].get("n", 0)) == 1 and w["meta"].get("kappa") == -1 for w in fac_wfs)


def plot_comparison(
    fac_v: dict,
    fac_wfs: list[dict],
    pinn: dict,
    *,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    r_pinn = pinn["r"]
    r_fac = fac_v["r"]
    r_lo = max(r_pinn[0], r_fac[0], 1e-4)
    r_hi = min(r_pinn[-1], r_fac[-1])
    mask = (r_pinn >= r_lo) & (r_pinn <= r_hi)
    r_plot = r_pinn[mask]

    v_pinn = interp_to(r_plot, r_pinn, pinn["V"])
    v_fac = interp_to(r_plot, r_fac, fac_v["V"])
    vn_fac = interp_to(r_plot, r_fac, fac_v["Vn"])

    metrics_rows = []

    # --- V(r) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].semilogy(r_plot, np.abs(vn_fac), "k--", alpha=0.4, label=r"FAC $-Z/r$ (Vn)")
    axes[0].plot(r_plot, v_fac, "C0-", lw=2, label="cFAC $V$")
    axes[0].plot(r_plot, v_pinn, "C3--", lw=2, label="PINN-ART $V$")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_xlabel(r"$r$ (Bohr)")
    axes[0].set_ylabel(r"$V(r)$ (Ha)")
    axes[0].set_title("Li: central potential")
    axes[0].legend(fontsize=8)

    axes[1].plot(r_plot, v_fac, "C0-", lw=2, label="cFAC")
    axes[1].plot(r_plot, v_pinn, "C3--", lw=2, label="PINN-ART")
    axes[1].set_xlim(0, min(12, r_hi))
    axes[1].set_ylim(min(v_fac.min(), v_pinn.min()) * 1.05, 0.5)
    axes[1].set_xlabel(r"$r$")
    axes[1].set_title(r"Valence region ($r \lesssim 12$)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "V_fac_vs_pinn.png", dpi=160)
    plt.close(fig)

    v_rmse = float(np.sqrt(np.mean((v_pinn - v_fac) ** 2)))
    mid = (r_plot >= 1.0) & (r_plot <= 8.0)
    v_mid = float(np.max(np.abs(v_pinn[mid] - v_fac[mid]))) if mid.any() else np.nan
    metrics_rows.append({"target": "V", "cos_P": np.nan, "rel_L2_PQ": v_rmse, "node_err": np.nan,
                         "E_fac_eV": np.nan, "E_pinn_eV": np.nan, "note": f"RMSE={v_rmse:.4f} Ha, mid_max={v_mid:.4f}"})

    # --- P, Q per FAC orbital ---
    fig, axes = plt.subplots(len(fac_wfs), 2, figsize=(11, 4 * len(fac_wfs)), squeeze=False)
    for i, fwf in enumerate(fac_wfs):
        n = int(fwf["meta"]["n"])
        l = 0
        sh = pick_shell(pinn["shells"], n, l)
        fac_label = f"FAC {fwf['label']} (file {fwf['path'].name})"
        file_note = f"{fwf['path'].name} → n={n}, κ={fwf['meta'].get('kappa')}"

        r_f = fwf["r"]
        p_f, q_f, norm_f = normalize_pq(r_f, fwf["P"], fwf["Q"])
        r_common = r_plot
        p_fi = interp_to(r_common, r_f, p_f)
        q_fi = interp_to(r_common, r_f, q_f)

        if sh is None:
            axes[i, 0].text(0.5, 0.5, f"No PINN orbital n={n}, l={l}\n(config {pinn['level_config']})",
                            ha="center", va="center", transform=axes[i, 0].transAxes)
            axes[i, 1].axis("off")
            metrics_rows.append({"target": fwf["label"], "cos_P": np.nan, "rel_L2_PQ": np.nan,
                                 "node_err": np.nan, "E_fac_eV": fac_energy_to_ev(fwf["energy_Ha"]),
                                 "E_pinn_eV": np.nan, "note": "missing PINN shell"})
            continue

        p_p = interp_to(r_common, r_pinn, sh["P"])
        q_p = interp_to(r_common, r_pinn, sh["Q"])
        p_p, q_p, flip = align_sign(p_p, q_p, p_fi)

        cos_p = cosine_on_grid(p_p, q_p, p_fi, q_fi, r_common)
        rel_l2 = float(np.sqrt(integrate_trapz((p_p - p_fi) ** 2 + (q_p - q_fi) ** 2, r_common)) /
                       max(np.sqrt(integrate_trapz(p_fi ** 2 + q_fi ** 2, r_common)), 1e-30))
        nodes_f = count_nodes(p_fi)
        nodes_p = count_nodes(p_p)

        axes[i, 0].plot(r_common, p_fi, "C0-", lw=2, label=fac_label)
        axes[i, 0].plot(r_common, p_p, "C3--", lw=2, label=f"PINN {sh['label']}")
        axes[i, 0].axhline(0, color="gray", lw=0.5)
        axes[i, 0].set_title(rf"${fwf['label']}$: $P(r)$  [{file_note}]")
        axes[i, 0].set_xlabel(r"$r$")
        axes[i, 0].legend(fontsize=7)
        axes[i, 0].set_xlim(0, min(15, r_hi))

        axes[i, 1].plot(r_common, q_fi, "C0-", lw=2, label="cFAC")
        axes[i, 1].plot(r_common, q_p, "C3--", lw=2, label="PINN-ART")
        axes[i, 1].set_title(rf"${fwf['label']}$: $Q(r)$")
        axes[i, 1].set_xlabel(r"$r$")
        axes[i, 1].legend(fontsize=7)
        axes[i, 1].set_xlim(0, min(15, r_hi))

        metrics_rows.append({
            "target": fwf["label"],
            "cos_P": cos_p,
            "rel_L2_PQ": rel_l2,
            "node_err": nodes_p - nodes_f,
            "E_fac_eV": fac_energy_to_ev(fwf["energy_Ha"]),
            "E_pinn_eV": sh["E_orb_Ha"] * HARTREE_TO_EV,
            "note": f"nodes FAC={nodes_f} PINN={nodes_p}, sign_flip={flip}",
        })

    fig.suptitle(
        f"Li cFAC vs PINN-ART  (config={pinn['level_config']}, Z={pinn['Z']})",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "PQ_fac_vs_pinn.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- PINN 1s only when no cFAC 1s file ---
    sh1 = pick_shell(pinn["shells"], 1, 0)
    if sh1 is not None and not has_fac_1s(fac_wfs):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].plot(r_pinn[mask], sh1["P"][mask], "C3-", lw=2, label="PINN 1s")
        axes[0].set_title(r"PINN $1s$ $P(r)$ (no cFAC 1s file)")
        axes[1].plot(r_pinn[mask], sh1["Q"][mask], "C3-", lw=2)
        axes[1].set_title(r"PINN $1s$ $Q(r)$")
        for ax in axes:
            ax.set_xlabel(r"$r$")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "Pinn_1s_only.png", dpi=160)
        plt.close(fig)

    return pd.DataFrame(metrics_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fac-dir", type=Path, default=FAC_DIR)
    ap.add_argument("--config", default="configs/v3_stage_a_z1_26_n10.yaml")
    ap.add_argument(
        "--ckpt",
        default="checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack",
    )
    ap.add_argument("--level", default="1s2 2s1", help="Li manifest level_config")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    fac_dir = args.fac_dir
    fac_v = load_fac_potential(fac_dir / "Li_V.dat")
    fac_wfs = discover_fac_wavefuncs(fac_dir)

    manifest = ROOT / load_config(ROOT / args.config).dataset.manifest
    row = find_manifest_idx(manifest, z=3, ion=0, level=args.level)
    pinn = pinn_forward(ROOT / args.config, ROOT / args.ckpt, row)

    print("\n=== cFAC wavefunction files (from headers) ===")
    for w in fac_wfs:
        p = w["path"].name
        n, k = w["meta"].get("n"), w["meta"].get("kappa")
        print(f"  {p:12s} → n={n}, κ={k}  ({w['label']})")
    print("Li_V.dat  → V(r) = Vc + U\n")

    print("=== PINN shells ===")
    for sh in pinn["shells"]:
        print(f"  {sh['label']:4s} n={sh['n']} l={sh['l']} ω={sh['omega']:.0f}  E={sh['E_orb_Ha']*HARTREE_TO_EV:.2f} eV")

    df = plot_comparison(fac_v, fac_wfs, pinn, out_dir=args.out_dir)
    df.to_csv(args.out_dir / "metrics.csv", index=False)

    print("\n=== Comparison metrics ===")
    print(df.to_string(index=False))
    print(f"\nFigures: {args.out_dir}/")
    print("  V_fac_vs_pinn.png")
    print("  PQ_fac_vs_pinn.png")
    if not has_fac_1s(fac_wfs):
        print("  Pinn_1s_only.png")
    print("  metrics.csv\n")


if __name__ == "__main__":
    main()
