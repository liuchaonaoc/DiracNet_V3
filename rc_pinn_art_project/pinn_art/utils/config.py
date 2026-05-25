"""YAML config loader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    defaults_path = path.parent / "default.yaml"
    if path.name != "default.yaml" and defaults_path.exists():
        with defaults_path.open() as f:
            base = yaml.safe_load(f) or {}

        def merge(a: dict, b: dict) -> dict:
            out = dict(a)
            for k, v in b.items():
                if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                    out[k] = merge(out[k], v)
                else:
                    out[k] = v
            return out

        raw = merge(base, raw or {})
    return _to_namespace(raw or {})
