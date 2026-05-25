"""Compute jj-coupling angular coefficients (Wigner-Eckart / triangle heuristic)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CSFSpec:
    """One CSF row within a parent configuration."""

    level_config: str
    J: float
    parity: int
    term: str = ""


def _triangle_allowed(Ji: float, Jj: float, k: int) -> bool:
    if k < 0:
        return False
    if abs(Ji - Jj) > k + 1e-9:
        return False
    if Ji + Jj + 1e-9 < k:
        return False
    return True


def wigner_eckart_coefficient(Ji: float, Jj: float, k: int) -> float:
    """
    Reduced angular coupling factor for rank-k two-body operator between CSFs.

    Uses standard triangle selection + a normalized Wigner-Eckart scalar
    (Phase-1 surrogate; replace with full 6j when Racah tables are wired).
    """
    if not _triangle_allowed(Ji, Jj, k):
        return 0.0
    if k == 0:
        return 1.0 if abs(Ji - Jj) < 1e-9 else 0.0
    phase = -1.0 if int(round(Ji + Jj + k)) % 2 else 1.0
    norm = math.sqrt((2.0 * Ji + 1.0) * (2.0 * Jj + 1.0))
    return phase * norm / (Ji + Jj + k + 1.0)


def build_angular_matrices(
    csfs: list[CSFSpec],
    k_list: list[int],
) -> np.ndarray:
    """
    Build C[k_idx, i, j] for one parent group.

    Returns float64 array [n_k, M, M].
    """
    M = len(csfs)
    n_k = len(k_list)
    C = np.zeros((n_k, M, M), dtype=np.float64)
    for ki, k in enumerate(k_list):
        for i, ci in enumerate(csfs):
            for j, cj in enumerate(csfs):
                if ci.parity != cj.parity and k % 2 == 1:
                    continue
                C[ki, i, j] = wigner_eckart_coefficient(ci.J, cj.J, k)
    return C


def parent_config_hash(parent_config: str) -> int:
    """Stable int32 hash for parent_config string."""
    import zlib

    return np.int32(zlib.adler32(parent_config.encode("utf-8")) & 0x7FFFFFFF)
