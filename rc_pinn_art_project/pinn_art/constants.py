"""Physical constants in atomic units (Hartree)."""

from __future__ import annotations

import math

ALPHA: float = 1.0 / 137.035999084
C_LIGHT: float = 1.0 / ALPHA
HARTREE_TO_EV: float = 27.211386245988
RYDBERG_EV: float = HARTREE_TO_EV / 2.0
FINE_STRUCTURE: float = ALPHA
TWO_PI: float = 2.0 * math.pi
BOHR_RADIUS_M: float = 5.29177210903e-11


def hartree_to_ev(e):
    import jax.numpy as jnp

    return e * HARTREE_TO_EV


def ev_to_hartree(e):
    import jax.numpy as jnp

    return e / HARTREE_TO_EV
