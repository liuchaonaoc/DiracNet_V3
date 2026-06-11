"""Batch collation for JAX training."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _lookup_C_ang(parent_config: str, cache, n_csf_max: int, n_k: int) -> np.ndarray:
    from ..ci.racah_compute import parent_config_hash

    M = cache.M_max
    out = np.zeros((n_k, n_csf_max, n_csf_max), dtype=np.float64)
    idx = cache.parent_index(int(parent_config_hash(parent_config)))
    if idx is None:
        out[0, 0, 0] = 1.0
    else:
        block = cache.C[idx, :n_k, :M, :M]
        take = min(n_csf_max, M)
        out[:, :take, :take] = block[:, :take, :take]
    return out


def collate_batches(
    items: list[dict],
    n_csf_max: int = 32,
    *,
    racah_cache=None,
    k_list: tuple[int, ...] | None = None,
) -> dict:
    B = len(items)
    n_orb = items[0]["kappa"].shape[0]

    def stack(key, dtype=None):
        arrs = [it[key] for it in items]
        return jnp.asarray(np.stack(arrs), dtype=dtype)

    E_nist = np.full((B, n_csf_max), np.nan, dtype=np.float32)
    nist_mask = np.zeros((B, n_csf_max), dtype=bool)
    for b, it in enumerate(items):
        slot = int(it.get("csf_slot", 0))
        if slot < 0 or slot >= n_csf_max:
            slot = 0
        if it.get("nist_mask_scalar", False) and np.isfinite(it.get("E_nist_scalar", np.nan)):
            E_nist[b, slot] = it["E_nist_scalar"]
            nist_mask[b, slot] = True

    csf_mask = np.zeros((B, n_csf_max), dtype=bool)
    for b, it in enumerate(items):
        slot = int(it.get("csf_slot", 0))
        if 0 <= slot < n_csf_max:
            csf_mask[b, slot] = True
        else:
            csf_mask[b, 0] = True
    if not csf_mask.any():
        csf_mask[:, 0] = True

    batch = {
        "Z": stack("Z", jnp.int32),
        "ion_charge": stack("ion_charge", jnp.int32),
        "nele": stack("nele", jnp.int32),
        "shell_table": stack("shell_table", jnp.int32),
        "kappa": stack("kappa", jnp.int32),
        "omega": stack("omega", jnp.float32),
        "orb_mask": stack("orb_mask", bool),
        "csf_mask": jnp.asarray(csf_mask),
        "E_nist": jnp.asarray(E_nist),
        "nist_mask": jnp.asarray(nist_mask),
    }

    if racah_cache is not None and k_list is not None:
        n_k = len(k_list)
        C = np.stack(
            [_lookup_C_ang(str(it.get("parent_config", "1s1")), racah_cache, n_csf_max, n_k) for it in items],
            axis=0,
        )
        batch["C_ang"] = jnp.asarray(C, dtype=jnp.float32)

    if items[0].get("csf_to_orb") is not None:
        batch["csf_to_orb"] = jnp.asarray(np.stack([it["csf_to_orb"] for it in items]), dtype=jnp.float32)

    return batch


def collate_spectrum_group(
    items: list[dict],
    n_csf_max: int = 32,
    *,
    racah_cache=None,
    k_list: tuple[int, ...] | None = None,
) -> dict:
    """Collate multiple CSF rows sharing one spatial (level_config) into batch B=1."""
    if not items:
        raise ValueError("empty spectrum group")
    base = items[0]
    n_orb = base["kappa"].shape[0]
    M = min(len(items), n_csf_max)

    E_nist = np.full((1, n_csf_max), np.nan, dtype=np.float32)
    nist_mask = np.zeros((1, n_csf_max), dtype=bool)
    csf_mask = np.zeros((1, n_csf_max), dtype=bool)
    csf_to_orb = np.zeros((1, n_csf_max, n_orb), dtype=np.float32)
    active_orb = int(np.argmax(base["orb_mask"])) if bool(np.any(base["orb_mask"])) else 0
    for i, it in enumerate(items[:M]):
        csf_mask[0, i] = True
        csf_to_orb[0, i, active_orb] = 1.0
        if it.get("nist_mask_scalar", False) and np.isfinite(it.get("E_nist_scalar", np.nan)):
            E_nist[0, i] = it["E_nist_scalar"]
            nist_mask[0, i] = True

    batch = {
        "Z": jnp.asarray([int(base["Z"])], dtype=jnp.int32),
        "ion_charge": jnp.asarray([int(base["ion_charge"])], dtype=jnp.int32),
        "nele": jnp.asarray([int(base["nele"])], dtype=jnp.int32),
        "shell_table": jnp.asarray(base["shell_table"][None, ...], dtype=jnp.int32),
        "kappa": jnp.asarray(base["kappa"][None, ...], dtype=jnp.int32),
        "omega": jnp.asarray(base["omega"][None, ...], dtype=jnp.float32),
        "orb_mask": jnp.asarray(base["orb_mask"][None, ...], dtype=bool),
        "csf_mask": jnp.asarray(csf_mask),
        "E_nist": jnp.asarray(E_nist),
        "nist_mask": jnp.asarray(nist_mask),
        "csf_to_orb": jnp.asarray(csf_to_orb),
    }

    if racah_cache is not None and k_list is not None:
        n_k = len(k_list)
        parent = str(base.get("parent_config", "1s1"))
        C = _lookup_C_ang(parent, racah_cache, n_csf_max, n_k)
        batch["C_ang"] = jnp.asarray(C[None, ...], dtype=jnp.float32)

    return batch
