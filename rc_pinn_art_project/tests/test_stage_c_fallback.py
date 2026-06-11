"""Stage C fallback: no NIST mask -> theory diagonal."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from pinn_art.ci.nist_inject import fill_h_diagonal_hybrid


def test_fallback_uses_theory_diagonal():
    H = jnp.array([[[-0.5, 0.1], [0.1, -0.4]]])
    E_nist = jnp.array([[-0.6, jnp.nan]])
    mask = jnp.array([[False, False]])
    H_out, src = fill_h_diagonal_hybrid(H, E_nist, mask)
    assert int(src[0, 0]) == 0
    assert float(H_out[0, 0, 0]) == pytest.approx(-0.5)
    assert float(H_out[0, 1, 1]) == pytest.approx(-0.4)


def test_partial_mask_only_first_csf():
    H = jnp.array([[[-1.0, 0.0], [0.0, -0.5]]])
    E_nist = jnp.array([[-0.7, jnp.nan]])
    mask = jnp.array([[True, False]])
    H_out, src = fill_h_diagonal_hybrid(H, E_nist, mask)
    assert float(H_out[0, 0, 0]) == pytest.approx(-0.7)
    assert float(H_out[0, 1, 1]) == pytest.approx(-0.5)
    assert int(src[0, 0]) == 1
    assert int(src[0, 1]) == 0
