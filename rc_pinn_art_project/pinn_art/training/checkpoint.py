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
