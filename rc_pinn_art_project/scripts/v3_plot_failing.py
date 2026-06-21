#!/usr/bin/env python3
"""Plot P_model vs P_ref for the failing rows to diagnose whether the
SIREN trunk can fit long-range oscillatory orbitals."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.hydrogenic import hydrogenic_P_jax
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.training.checkpoint import load_params

cfg = load_config(ROOT / "configs/v3_stage_a_laguerre_basis_c.yaml")
grid = make_radial_grid(
    float(cfg.grid.r_min), float(cfg.grid.r_max),
    int(cfg.grid.n_grid), str(cfg.grid.scheme),
)
model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
params = load_params(ROOT / "checkpoints/v3_stage_a_laguerre_basis_c_c/stage_a_last.msgpack")

ds = ManifestDataset(ROOT / cfg.dataset.manifest,
                     n_orb_max=int(cfg.model.n_orb_max),
                     n_csf_max=int(cfg.model.n_csf_max))

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
targets = [(0, 0, "H 1s"), (4, 1, "H 2s"), (8, 1, "H 9s"),
           (10, 2, "He 1s"), (16, 2, "He 7s"), (18, 2, "He 9s"),
           (20, 3, "Li 1s"), (27, 3, "Li 8s"), (29, 3, "Li 10s")]

for ax, (i, _, title) in zip(axes.flat, targets):
    batch = collate_batches([ds[i]], n_csf_max=int(cfg.model.n_csf_max))
    out = model.apply(params, batch, grid, train=False, return_ci=False)
    P = np.asarray(out["wavefunctions"]["P"][0])
    r_np = np.asarray(grid.r)
    sh = np.asarray(batch["shell_table"][0])
    n_arr = sh[:, 0].astype(int)
    l_arr = sh[:, 1].astype(int)
    P_ref = np.asarray(hydrogenic_P_jax(grid.r, batch["Z"], jnp.asarray(n_arr)[None, :], jnp.asarray(l_arr)[None, :]))[0]
    for a in range(P.shape[0]):
        if n_arr[a] == 0:
            continue
        ax.plot(r_np, P[a], label=f"P_model (n={int(n_arr[a])})", lw=1)
        ax.plot(r_np, P_ref[a], '--', label=f"P_ref (n={int(n_arr[a])})", lw=1)
        ax.set_title(f"{title} — cos={np.corrcoef(P[a], P_ref[a])[0,1]:.3f}")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("r")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        break

fig.tight_layout()
out = ROOT / "results/laguerre_basis_eval/diag_failing_rows.png"
fig.savefig(out, dpi=100)
print(f"saved -> {out}")