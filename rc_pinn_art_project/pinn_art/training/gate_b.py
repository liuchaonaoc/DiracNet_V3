"""Gate B — synthetic CI Hamiltonians + eigh gradient health."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ci.eigen_solver import safe_eigh
from ..constants import hartree_to_meV
from ..losses.ci_loss import spectrum_error_meV


def _gate_synthetic_case(H: jnp.ndarray, E_ref: jnp.ndarray, meV_threshold: float) -> dict:
    import numpy as np

    E_pred, _ = safe_eigh(H[None, :, :])
    err_meV = float(spectrum_error_meV(E_pred[0], E_ref))
    return {
        "pass": err_meV <= meV_threshold,
        "err_meV": err_meV,
        "E_pred": np.asarray(E_pred[0]).tolist(),
        "E_ref": np.asarray(E_ref).tolist(),
    }


def run_synthetic_gate(meV_threshold: float = 10.0) -> dict:
    """2×2 and 3×3 known spectra (07_training_pipeline Gate B)."""
    H2 = jnp.array([[[-1.0, 0.05], [0.05, -0.5]]])
    E2_ref = jnp.array([-1.025, -0.475])  # approximate; checked via eigh below
    E2_ref = jnp.sort(jnp.linalg.eigh(H2[0])[0])

    H3 = jnp.diag(jnp.array([-2.0, -1.0, -0.5]))[None, :, :]
    E3_ref = jnp.array([-2.0, -1.0, -0.5])

    r2 = _gate_synthetic_case(H2, E2_ref, meV_threshold)
    r3 = _gate_synthetic_case(H3, E3_ref, meV_threshold)
    return {"2x2": r2, "3x3": r3, "verdict": "PASS" if r2["pass"] and r3["pass"] else "FAIL"}


def run_eigh_gradient_gate(n_trials: int = 8) -> dict:
    """Mirror tests/test_eigh_gradient_finite.py."""

    def loss(H):
        E, _ = safe_eigh(H[None, :, :])
        return jnp.sum(E)

    key = jax.random.PRNGKey(0)
    ok = True
    for _ in range(n_trials):
        key, k = jax.random.split(key)
        H = jax.random.normal(k, (3, 3))
        H = 0.5 * (H + H.T)
        g = jax.grad(lambda h: loss(h))(H)
        if not bool(jnp.isfinite(g).all()):
            ok = False
            break
    return {"pass": ok, "n_trials": n_trials}


def run_manifest_ci_gate(
    apply_fn,
    params,
    dataset,
    grid,
    collate_fn,
    *,
    meV_threshold: float = 50.0,
    max_rows: int | None = None,
) -> dict:
    """Per-row |E_csf - E_orb| on leading CSF (hydrogenic single-ref)."""
    n = len(dataset) if max_rows is None else min(max_rows, len(dataset))
    rows = []
    for i in range(n):
        batch = collate_fn([dataset[i]])
        out = apply_fn(params, batch, grid, train=False, return_ci=True)
        dE_meV = float(
            hartree_to_meV(jnp.abs(out["E_csf"][0, 0] - out["E_orb"][0, 0]))
        )
        row = dataset.df.iloc[i]
        rows.append(
            {
                "idx": i,
                "Z": int(row["Z"]),
                "element": str(row.get("element", "")),
                "n": int(row.get("n", 0)),
                "dE_meV": dE_meV,
                "pass": dE_meV <= meV_threshold,
            }
        )
    n_pass = sum(r["pass"] for r in rows)
    return {
        "rows": rows,
        "n_pass": n_pass,
        "n_total": n,
        "verdict": "PASS" if n_pass == n else "FAIL",
        "meV_threshold": meV_threshold,
    }
