"""Load / lookup precomputed Racah angular coefficient cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class RacahCache:
    """In-memory Racah cache."""

    k_list: np.ndarray          # [n_k] int
    parent_ids: np.ndarray      # [P] int32
    M_max: int
    C: np.ndarray               # [P, n_k, M_max, M_max] float64
    csf_J: np.ndarray           # [P, M_max] float64
    csf_parity: np.ndarray      # [P, M_max] int8
    version: str = "1.0"

    @classmethod
    def load(cls, path: str | Path) -> RacahCache:
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        return cls(
            k_list=np.asarray(data["k_list"], dtype=np.int32),
            parent_ids=np.asarray(data["parent_ids"], dtype=np.int32),
            M_max=int(data["M_max"]),
            C=np.asarray(data["C"], dtype=np.float64),
            csf_J=np.asarray(data["csf_J"], dtype=np.float64),
            csf_parity=np.asarray(data["csf_parity"], dtype=np.int8),
            version=str(data["version"]) if "version" in data else "1.0",
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            k_list=self.k_list,
            parent_ids=self.parent_ids,
            M_max=np.int32(self.M_max),
            C=self.C,
            csf_J=self.csf_J,
            csf_parity=self.csf_parity,
            version=np.array(str(self.version)),
            coupling_scheme=np.array("jj"),
        )

    def parent_index(self, parent_id: int) -> int | None:
        hits = np.where(self.parent_ids == parent_id)[0]
        return int(hits[0]) if hits.size else None


def lookup_angular(
    parent_ids: jnp.ndarray,
    cache: RacahCache,
) -> jnp.ndarray:
    """
    parent_ids [B] int32 -> C_ang [B, n_k, M_max, M_max] (stop_gradient).
    Unknown parents → identity on k=0 block, zeros elsewhere.
    """
    B = parent_ids.shape[0]
    P, n_k, M, _ = cache.C.shape
    out = np.zeros((B, n_k, M, M), dtype=np.float64)
    pid_np = np.asarray(parent_ids, dtype=np.int32)

    for b in range(B):
        idx = cache.parent_index(int(pid_np[b]))
        if idx is None:
            out[b, 0] = np.eye(M)
        else:
            out[b] = cache.C[idx]

    return jax.lax.stop_gradient(jnp.asarray(out))


def build_cache_from_manifest_rows(
    rows: list[dict],
    k_list: list[int],
    M_max: int,
) -> RacahCache:
    """Build cache from grouped manifest rows (see scripts/v3_build_racah_cache.py)."""
    from .racah_compute import CSFSpec, build_angular_matrices, parent_config_hash

    groups: dict[str, list[CSFSpec]] = {}
    for row in rows:
        parent = str(row["parent_config"])
        spec = CSFSpec(
            level_config=str(row.get("level_config", parent)),
            J=float(row.get("J", 0.5)),
            parity=int(row.get("parity", 0)),
            term=str(row.get("term", "")),
        )
        groups.setdefault(parent, [])
        if not any(s.level_config == spec.level_config and abs(s.J - spec.J) < 1e-6 for s in groups[parent]):
            groups[parent].append(spec)

    parent_ids = []
    C_list = []
    J_list = []
    P_list = []
    for parent, csfs in sorted(groups.items()):
        csfs = csfs[:M_max]
        M = len(csfs)
        C = build_angular_matrices(csfs, k_list)
        pad_C = np.zeros((len(k_list), M_max, M_max), dtype=np.float64)
        pad_C[:, :M, :M] = C
        pad_J = np.zeros(M_max, dtype=np.float64)
        pad_P = np.zeros(M_max, dtype=np.int8)
        for i, c in enumerate(csfs):
            pad_J[i] = c.J
            pad_P[i] = c.parity
        parent_ids.append(parent_config_hash(parent))
        C_list.append(pad_C)
        J_list.append(pad_J)
        P_list.append(pad_P)

    if not parent_ids:
        n_k = len(k_list)
        return RacahCache(
            k_list=np.array(k_list, dtype=np.int32),
            parent_ids=np.zeros(0, dtype=np.int32),
            M_max=M_max,
            C=np.zeros((0, n_k, M_max, M_max), dtype=np.float64),
            csf_J=np.zeros((0, M_max), dtype=np.float64),
            csf_parity=np.zeros((0, M_max), dtype=np.int8),
        )

    return RacahCache(
        k_list=np.array(k_list, dtype=np.int32),
        parent_ids=np.array(parent_ids, dtype=np.int32),
        M_max=M_max,
        C=np.stack(C_list, axis=0),
        csf_J=np.stack(J_list, axis=0),
        csf_parity=np.stack(P_list, axis=0),
    )
