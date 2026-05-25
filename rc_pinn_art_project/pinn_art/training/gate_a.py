"""Stage A analytic gate."""

from __future__ import annotations

import re
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..constants import HARTREE_TO_EV
from ..physics.hydrogenic import cosine_signed, hydrogenic_energy, hydrogenic_P_analytic
from ..utils.grid import RadialGrid

_LEVEL_PATTERN = re.compile(r"^(\d+)([spdfghi])", re.IGNORECASE)


@dataclass
class GateReport:
    cos_min: float
    pde_max: float
    e_orb_mae_meV: float
    verdict: str
    n: int = 1
    l: int = 0

    def to_markdown(self) -> str:
        return (
            f"| n, l | {self.n}, {self.l} |\n"
            f"| cos_min | {self.cos_min:.4f} |\n"
            f"| pde_max | {self.pde_max:.4e} |\n"
            f"| e_orb_mae_meV | {self.e_orb_mae_meV:.4f} |\n"
            f"| **VERDICT** | **{self.verdict}** |\n"
        )


def parse_n_l_from_level_config(level_config: str) -> tuple[int, int]:
    """Parse principal n and orbital l from strings like ``1s1``, ``6s1``, ``2p3``."""
    s = str(level_config).strip()
    if not s:
        return 1, 0
    m = _LEVEL_PATTERN.match(s)
    if not m:
        return 1, 0
    n = int(m.group(1))
    l_letter = m.group(2).lower()
    l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
    l = l_map.get(l_letter, 0)
    return max(n, 1), l


def principal_quantum_from_batch(
    batch: dict,
    b: int = 0,
    orb: int = 0,
    *,
    level_config: str | None = None,
) -> tuple[int, int]:
    """Resolve (n, l) for gate reference from collated batch or manifest string."""
    shell_table = batch.get("shell_table")
    if shell_table is not None:
        n = int(np.asarray(shell_table[b, orb, 0]))
        l = int(np.asarray(shell_table[b, orb, 1]))
        if n >= 1:
            return n, max(l, 0)
    if level_config is not None:
        return parse_n_l_from_level_config(level_config)
    return 1, 0


def resolve_gate_thresholds(stage_a_cfg, mode: str = "default") -> dict[str, float]:
    """Pick Gate thresholds from config.

    ``mode``: ``default`` | ``relaxed`` | ``mid`` | ``strict``
    For phase2 yaml: ``default``/``relaxed`` → ``stage_a.gate``;
    ``mid`` → ``gate_mid``; ``strict`` → ``gate_strict``.
    """
    if mode in ("default", "relaxed"):
        g = getattr(stage_a_cfg, "gate", stage_a_cfg)
    elif mode == "mid":
        g = getattr(stage_a_cfg, "gate_mid", getattr(stage_a_cfg, "gate", stage_a_cfg))
    elif mode == "strict":
        g = getattr(stage_a_cfg, "gate_strict", getattr(stage_a_cfg, "gate", stage_a_cfg))
    else:
        raise ValueError(f"unknown gate mode: {mode!r}")
    return {
        "cos_threshold": float(getattr(g, "cos_threshold", 0.99)),
        "pde_threshold": float(getattr(g, "pde_threshold", 1e-2)),
        "e_orb_meV_threshold": float(getattr(g, "e_orb_meV_threshold", 50.0)),
    }


def check_gate_a(
    out: dict,
    batch: dict,
    grid: RadialGrid,
    *,
    cos_threshold: float = 0.99,
    pde_threshold: float = 1e-3,
    e_orb_meV_threshold: float = 1.0,
    n: int | None = None,
    l: int | None = None,
    level_config: str | None = None,
) -> GateReport:
    """Compare model output to hydrogenic reference for the row's (n, l).

    ``n`` / ``l`` can be passed explicitly; otherwise read from ``batch['shell_table']``
    or ``level_config`` (e.g. ``"3s1"`` → n=3, l=0).
    """
    from ..losses.pde_loss import dirac_pde_loss

    wf = out["wavefunctions"]
    P = wf["P"]
    Z = int(np.asarray(batch["Z"][0]))

    cos_vals = []
    e_mae_vals = []
    n_ref, l_ref = 1, 0
    for b in range(P.shape[0]):
        for a in range(P.shape[1]):
            if not bool(np.asarray(batch["orb_mask"][b, a])):
                continue
            if n is not None and l is not None:
                n_b, l_b = n, l
            else:
                n_b, l_b = principal_quantum_from_batch(
                    batch, b=b, orb=a, level_config=level_config
                )
            if b == 0 and a == 0:
                n_ref, l_ref = n_b, l_b
            P_ref = hydrogenic_P_analytic(grid.r, Z, n_b, l_b)
            c = float(cosine_signed(P[b, a], P_ref, grid))
            cos_vals.append(abs(c))
            e_ref = hydrogenic_energy(Z, n_b)
            e_pred = float(np.asarray(out["E_orb"][b, a]))
            e_mae_vals.append(abs(e_pred - e_ref) * HARTREE_TO_EV)

    cos_min = min(cos_vals) if cos_vals else 0.0
    e_orb_mae_meV = max(e_mae_vals) if e_mae_vals else float("inf")

    pde = float(
        dirac_pde_loss(
            P, wf["Q"], wf["dPdr"], wf["dQdr"], out["V"],
            batch["kappa"], grid.r, grid, batch["orb_mask"], out["E_orb"],
        )
    )

    ok = (
        cos_min >= cos_threshold
        and pde <= pde_threshold
        and e_orb_mae_meV <= e_orb_meV_threshold
    )
    return GateReport(
        cos_min=cos_min,
        pde_max=pde,
        e_orb_mae_meV=e_orb_mae_meV,
        verdict="PASS" if ok else "FAIL",
        n=n_ref,
        l=l_ref,
    )
