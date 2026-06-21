#!/usr/bin/env python3
"""Quick diagnostic: print P(r) for failing rows to determine whether the
issue is "node fell off the grid" vs "model cannot represent the long
orbital"."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.hydrogenic import cosine_signed, hydrogenic_P_jax
from pinn_art.utils.config import load_config
from pinn_art.utils.grid import make_radial_grid
from pinn_art.training.checkpoint import load_params

import json

cfg_path = ROOT / "configs/v3_stage_a_laguerre_basis_c.yaml"
ckpt = ROOT / "checkpoints/v3_stage_a_laguerre_basis_c_c/stage_a_last.msgpack"

cfg = load_config(cfg_path)
grid = make_radial_grid(
    float(cfg.grid.r_min), float(cfg.grid.r_max),
    int(cfg.grid.n_grid), str(cfg.grid.scheme),
)
model, params = build_model_and_params(cfg, grid, jax.random.PRNGKey(0))
params = load_params(ckpt)

manifest = ROOT / cfg.dataset.manifest
ds = ManifestDataset(manifest, n_orb_max=int(cfg.model.n_orb_max), n_csf_max=int(cfg.model.n_csf_max))

print(f"{'i':>3} {'Z':>3} {'n':>3} {'lambda':>7} {'nodes_obs/exp':>12} {'cos':>8} {'pmax':>9} {'argmax(r)':>10} {'last_node_r':>12}")
for i in range(30):
    batch = collate_batches([ds[i]], n_csf_max=int(cfg.model.n_csf_max))
    out = model.apply(params, batch, grid, train=False, return_ci=False)

    P = np.asarray(out["wavefunctions"]["P"][0])  # [N_orb, N_g]
    r_np = np.asarray(grid.r)

    sh = np.asarray(batch["shell_table"][0])
    n_arr = sh[:, 0].astype(int)
    l_arr = sh[:, 1].astype(int)
    Z_val = int(np.asarray(batch["Z"][0]))

    valid = (n_arr > l_arr) & (n_arr > 0)
    P_ref = np.asarray(hydrogenic_P_jax(grid.r, batch["Z"], jnp.asarray(n_arr)[None, :], jnp.asarray(l_arr)[None, :]))[0]
    P_ref = P_ref * valid[:, None]

    lag = out.get("laguerre_lambdas")
    if lag is not None and lag[0] is not None:
        arr = np.asarray(lag[0])
        lambdas = arr if arr.ndim > 0 else np.full(P.shape[0], float(arr))
    else:
        lambdas = None

    for a in range(P.shape[0]):
        if not valid[a]:
            continue
        pa = P[a]
        pmax = float(np.max(np.abs(pa)))
        idx_peak = int(np.argmax(np.abs(pa)))
        r_peak = float(r_np[idx_peak])

        sign_arr = np.sign(pa)
        sign_arr = np.where(np.abs(pa) > 1e-3 * pmax, sign_arr, 0)
        nonzero = sign_arr[sign_arr != 0]
        diffs = np.diff(nonzero) if nonzero.size > 1 else np.zeros(0)
        n_nodes = int(np.sum(diffs != 0))
        expected = int(n_arr[a] - l_arr[a] - 1)

        # location of last observed sign-change boundary
        last_node_r = float("nan")
        if n_nodes > 0 and nonzero.size > 1:
            nonzero_idx = np.where(sign_arr != 0)[0]
            for k in range(n_nodes):
                last_node_r = float(r_np[nonzero_idx[k+1]])

        cos_v = float(cosine_signed(pa, P_ref[a], grid))
        lam_v = float(lambdas[a]) if lambdas is not None else float("nan")
        print(f"{i:3d} {Z_val:3d} {int(n_arr[a]):3d} {lam_v:7.4f} {n_nodes:3d}/{expected:3d}    {cos_v:8.4f} {pmax:9.2e} {r_peak:10.3f} {last_node_r:12.3f}")