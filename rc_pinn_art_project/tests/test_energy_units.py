"""Energy unit conversions (meV vs Hartree vs eV)."""

import jax.numpy as jnp
import numpy as np

from pinn_art.constants import (
    ENERGY_UNIT,
    ev_to_meV,
    hartree_to_meV,
    meV_to_hartree,
)
from pinn_art.training.gate_a import check_gate_a


def test_energy_unit_label():
    assert ENERGY_UNIT == "meV"


def test_hartree_meV_roundtrip():
    ha = -0.5
    mev = float(hartree_to_meV(ha))
    assert abs(float(meV_to_hartree(mev)) - ha) < 1e-12


def test_ev_meV_nist_ingest():
    assert float(ev_to_meV(10.2)) == 10200.0


def test_gate_a_reports_meV(r_grid, h1s_batch):
    from pinn_art.constants import hartree_to_meV
    from pinn_art.physics.hydrogenic import hydrogenic_energy

    e_ref = hydrogenic_energy(1, 1)
    e_pred = -0.5
    out = {
        "wavefunctions": {
            "P": jnp.ones((1, 1, r_grid.r.shape[0])),
            "Q": jnp.zeros((1, 1, r_grid.r.shape[0])),
            "dPdr": jnp.zeros((1, 1, r_grid.r.shape[0])),
            "dQdr": jnp.zeros((1, 1, r_grid.r.shape[0])),
        },
        "V": -1.0 / r_grid.r[None, :],
        "E_orb": jnp.array([[e_pred]]),
        "LP": jnp.zeros((1, 1, r_grid.r.shape[0])),
        "LQ": jnp.zeros((1, 1, r_grid.r.shape[0])),
    }
    rep = check_gate_a(
        out,
        h1s_batch,
        r_grid,
        cos_threshold=0.0,
        pde_threshold=1e9,
        e_orb_meV_threshold=1e9,
    )
    expect = float(hartree_to_meV(abs(e_pred - e_ref)))
    assert abs(rep.e_orb_mae_meV - expect) < 1e-3
