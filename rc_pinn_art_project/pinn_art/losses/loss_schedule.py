"""Stage A loss weights."""

from __future__ import annotations


def stage_a_weights(cfg) -> dict[str, float]:
    w = getattr(cfg, "stage_a", None)
    if w is None:
        return {"pde": 1.0, "ortho": 100.0, "asym": 0.01, "norm": 10.0, "v_prior": 0.1, "v_smooth": 1e-3}
    weights = getattr(w, "weights", w)
    return {
        "pde": float(getattr(weights, "pde", 1.0)),
        "ortho": float(getattr(weights, "ortho", 100.0)),
        "asym": float(getattr(weights, "asym", 0.01)),
        "norm": float(getattr(weights, "norm", 10.0)),
        "v_prior": float(getattr(weights, "v_prior", 0.1)),
        "v_smooth": float(getattr(weights, "v_smooth", 1e-3)),
    }


def stage_b_weights(cfg) -> dict[str, float]:
    w = getattr(cfg, "stage_b", None)
    if w is None:
        return {"slat": 1.0, "offdiag": 0.1, "e_csf": 10.0, "pde": 0.01}
    weights = getattr(w, "weights", w)
    return {
        "slat": float(getattr(weights, "slat", 1.0)),
        "offdiag": float(getattr(weights, "offdiag", 0.1)),
        "e_csf": float(getattr(weights, "e_csf", 10.0)),
        "pde": float(getattr(weights, "pde", 0.01)),
    }
