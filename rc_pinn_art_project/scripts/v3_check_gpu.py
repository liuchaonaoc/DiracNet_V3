#!/usr/bin/env python3
"""Quick JAX GPU sanity check before training."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp


def main() -> None:
    print("python:", sys.executable)
    print("JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS", "(unset)"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    print("jax:", jax.__version__)
    print("default_backend:", jax.default_backend())
    print("devices:", jax.devices())

    if jax.default_backend() != "gpu":
        print("\n*** WARNING: not on GPU backend. Training will be very slow. ***")
        print("Fix: pip install -U 'jax[cuda12]' jax-cuda12-plugin jax-cuda12-pjrt")
        sys.exit(1)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4096, 4096))
    t0 = time.perf_counter()
    y = jnp.dot(x, x)
    jax.block_until_ready(y)
    dt = time.perf_counter() - t0
    print(f"4096 matmul: {dt:.3f}s on {y.devices()}")
    if dt > 2.0:
        print("*** matmul slow — GPU may not be active ***")
        sys.exit(2)
    print("\nOK — GPU looks healthy. Run training next.")


if __name__ == "__main__":
    main()
