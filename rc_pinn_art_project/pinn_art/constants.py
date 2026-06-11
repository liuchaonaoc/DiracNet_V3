"""Physical constants in atomic units (Hartree).

Energy convention for user-facing I/O (gates, evaluate, CSV, logs):
  **meV** (millielectron volts).

Internal JAX physics (Dirac, Hamiltonian, ``safe_eigh``):
  **Hartree** (atomic units).

NIST ASD / legacy parquet columns may supply **eV**; convert at ingest via
``ev_to_meV`` / ``ev_to_hartree``.
"""

from __future__ import annotations

import math

ALPHA: float = 1.0 / 137.035999084
C_LIGHT: float = 1.0 / ALPHA
HARTREE_TO_EV: float = 27.211386245988
HARTREE_TO_MEV: float = HARTREE_TO_EV * 1000.0
EV_TO_MEV: float = 1000.0
RYDBERG_EV: float = HARTREE_TO_EV / 2.0
FINE_STRUCTURE: float = ALPHA
TWO_PI: float = 2.0 * math.pi
BOHR_RADIUS_M: float = 5.29177210903e-11

# Canonical energy unit for reports / gates / evaluation scripts
ENERGY_UNIT: str = "meV"


def hartree_to_ev(e):
    import jax.numpy as jnp

    return e * HARTREE_TO_EV


def ev_to_hartree(e):
    import jax.numpy as jnp

    return e / HARTREE_TO_EV


def hartree_to_meV(e):
    """Hartree → meV (scalar or array)."""
    import jax.numpy as jnp

    return e * HARTREE_TO_MEV


def meV_to_hartree(e):
    """meV → Hartree."""
    import jax.numpy as jnp

    return e / HARTREE_TO_MEV


def ev_to_meV(e):
    """eV → meV (NIST ASD ingest)."""
    import jax.numpy as jnp

    return e * EV_TO_MEV


def meV_to_ev(e):
    """meV → eV."""
    import jax.numpy as jnp

    return e / EV_TO_MEV


# Backward-compatible aliases
hartree_to_ev = hartree_to_ev
ev_to_hartree = ev_to_hartree
