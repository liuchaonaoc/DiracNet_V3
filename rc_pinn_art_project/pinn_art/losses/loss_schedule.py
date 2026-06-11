"""Stage A loss weights."""

from __future__ import annotations


def stage_a_weights(cfg) -> dict[str, float]:
    w = getattr(cfg, "stage_a", None)
    if w is None:
        return {
            "pde": 1.0, "ortho": 100.0, "asym": 0.01, "norm": 10.0,
            "v_prior": 0.1, "v_smooth": 1e-3, "scf": 0.0,
        }
    weights = getattr(w, "weights", w)
    return {
        "pde": float(getattr(weights, "pde", 1.0)),
        "ortho": float(getattr(weights, "ortho", 100.0)),
        "asym": float(getattr(weights, "asym", 0.01)),
        "norm": float(getattr(weights, "norm", 10.0)),
        "v_prior": float(getattr(weights, "v_prior", 0.1)),
        "v_smooth": float(getattr(weights, "v_smooth", 1e-3)),
        "scf": float(getattr(weights, "scf", 0.0)),
    }


def stage_a_dfs_cfg(cfg) -> tuple[bool, float, bool, bool, bool, str, bool]:
    """Read Stage-A DFS settings.

    -> (enabled, alpha_x, latter_tail, anchor_vprior, fermi_amaldi,
        scf_weight_mode, anchor_vprior_zeff)

    ``enabled`` defaults to True when ``cfg.dfs`` exists or ``weights.scf>0``.
    ``fermi_amaldi`` defaults to True (P0 self-interaction-correction fix).
    ``scf_weight_mode`` (P1) selects the radial weighting of the SCF loss:
        "density" (default) | "r2" | "uniform" | "r".
    ``anchor_vprior_zeff`` (R1.5, 路径 B) replaces the V_dfs anchor with
        the Slater-screened effective-charge potential ``-Z_anchor/r`` (when
        both are requested, zeff takes precedence). False by default.
    Returned as a hashable tuple so it can be a static JIT argument.
    """
    dfs = getattr(cfg, "dfs", None)
    weights = getattr(getattr(cfg, "stage_a", None), "weights", None)
    scf_w = float(getattr(weights, "scf", 0.0)) if weights is not None else 0.0
    if dfs is None:
        return (scf_w > 0.0, 1.0, True, scf_w > 0.0, True, "density", False)
    enabled = bool(getattr(dfs, "enabled", True))
    alpha_x = float(getattr(dfs, "alpha_x", 1.0))
    latter_tail = bool(getattr(dfs, "latter_tail", True))
    anchor_vprior = bool(getattr(dfs, "anchor_vprior", enabled))
    fermi_amaldi = bool(getattr(dfs, "fermi_amaldi", True))
    scf_weight_mode = str(getattr(dfs, "scf_weight_mode", "density"))
    anchor_vprior_zeff = bool(getattr(dfs, "anchor_vprior_zeff", False))
    return (enabled, alpha_x, latter_tail, anchor_vprior, fermi_amaldi,
            scf_weight_mode, anchor_vprior_zeff)


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
