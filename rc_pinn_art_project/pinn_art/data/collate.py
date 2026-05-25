"""Batch collation for JAX training."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def collate_batches(items: list[dict], n_csf_max: int = 32) -> dict:
    B = len(items)
    n_orb = items[0]["kappa"].shape[0]

    def stack(key, dtype=None):
        arrs = [it[key] for it in items]
        return jnp.asarray(np.stack(arrs), dtype=dtype)

    E_nist = np.full((B, n_csf_max), np.nan, dtype=np.float32)
    nist_mask = np.zeros((B, n_csf_max), dtype=bool)
    for b, it in enumerate(items):
        if it.get("nist_mask_scalar", False) and np.isfinite(it.get("E_nist_scalar", np.nan)):
            E_nist[b, 0] = it["E_nist_scalar"]
            nist_mask[b, 0] = True

    csf_mask = np.zeros((B, n_csf_max), dtype=bool)
    csf_mask[:, 0] = True

    return {
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
