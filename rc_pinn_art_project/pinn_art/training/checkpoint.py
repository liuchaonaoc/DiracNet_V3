"""Checkpoint save/load."""

from __future__ import annotations

import json
from pathlib import Path

import flax.serialization
import jax


def save_checkpoint(path: str | Path, state, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(flax.serialization.to_bytes(state.params))
    if extra:
        with path.with_suffix(".json").open("w") as f:
            json.dump(extra, f, indent=2)


def load_params(path: str | Path):
    path = Path(path)
    with path.open("rb") as f:
        return flax.serialization.from_bytes(None, f.read())


def merge_params(template, loaded):
    """Load Stage A weights into a larger param tree (e.g. + slater_log_scale)."""
    from flax.core import freeze
    from flax.traverse_util import flatten_dict, unflatten_dict

    flat_t = flatten_dict(freeze(template), keep_empty_nodes=True)
    flat_l = flatten_dict(freeze(loaded), keep_empty_nodes=True)
    merged = dict(flat_t)
    n = 0
    for k, v in flat_l.items():
        if k in merged and getattr(merged[k], "shape", None) == getattr(v, "shape", None):
            merged[k] = v
            n += 1
    return unflatten_dict(merged)
