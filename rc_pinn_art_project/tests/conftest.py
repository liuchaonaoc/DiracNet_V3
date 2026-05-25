from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_art.utils.grid import make_radial_grid


@pytest.fixture
def r_grid():
    return make_radial_grid(r_min=1e-4, r_max=50.0, n_grid=128, scheme="loglinear")


@pytest.fixture
def h1s_batch():
    shell_table = jnp.zeros((1, 16, 4), dtype=jnp.int32)
    # Slot 0 = 1s_{1/2}: n=1, l=0, 2j=1, occ=1
    shell_table = shell_table.at[0, 0].set(jnp.array([1, 0, 1, 1], dtype=jnp.int32))
    return {
        "Z": jnp.array([1], dtype=jnp.int32),
        "ion_charge": jnp.array([0], dtype=jnp.int32),
        "nele": jnp.array([1], dtype=jnp.int32),
        "omega": jnp.array([[1.0] + [0.0] * 15], dtype=jnp.float32),
        "kappa": jnp.array([[-1] + [0] * 15], dtype=jnp.int32),
        "orb_mask": jnp.array([[True] + [False] * 15]),
        "shell_table": shell_table,
        "csf_mask": jnp.array([[True] + [False] * 7]),
        "E_nist": jnp.array([[0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "nist_mask": jnp.array([[True] + [False] * 7]),
    }


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)
