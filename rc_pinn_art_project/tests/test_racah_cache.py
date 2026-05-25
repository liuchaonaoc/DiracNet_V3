from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from pinn_art.ci.racah_cache import RacahCache, build_cache_from_manifest_rows, lookup_angular
from pinn_art.ci.racah_compute import wigner_eckart_coefficient


def test_wigner_triangle_zero():
    assert wigner_eckart_coefficient(0.5, 2.5, 1) == 0.0


def test_build_and_roundtrip_cache(tmp_path):
    rows = [
        {"parent_config": "1s1", "level_config": "1s1", "J": 0.5, "parity": 0, "term": "2S"},
        {"parent_config": "1s1", "level_config": "2s1", "J": 0.5, "parity": 0, "term": "2S"},
    ]
    cache = build_cache_from_manifest_rows(rows, k_list=[0, 1], M_max=4)
    path = tmp_path / "racah.npz"
    cache.save(path)
    loaded = RacahCache.load(path)
    assert loaded.C.shape == cache.C.shape
    assert loaded.k_list.tolist() == [0, 1]


def test_lookup_angular_shape():
    rows = [{"parent_config": "1s1", "level_config": "1s1", "J": 0.5, "parity": 0}]
    cache = build_cache_from_manifest_rows(rows, k_list=[0], M_max=4)
    pid = jnp.array([cache.parent_ids[0]], dtype=jnp.int32)
    C = lookup_angular(pid, cache)
    assert C.shape == (1, 1, 4, 4)
    assert float(C[0, 0, 0, 0]) == 1.0
