import jax.numpy as jnp

from pinn_art.losses.pde_loss import dirac_pde_loss
from pinn_art.physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from pinn_art.physics.hydrogenic import cosine_signed, hydrogenic_energy, hydrogenic_P_analytic
from pinn_art.training.gate_a import check_gate_a


def _analytic_out(batch, grid):
    Z = int(jnp.asarray(batch["Z"][0]))
    r = grid.r
    P = hydrogenic_P_analytic(r, Z, 1, 0)[None, None, :]
    Q = P * 1e-4
    dP = jnp.gradient(P, r, axis=-1)
    dQ = jnp.gradient(Q, r, axis=-1)
    V = (-float(Z) / jnp.clip(r, 1e-8))[None, :]
    kappa = batch["kappa"]
    LP, LQ = dirac_apply(P, Q, dP, dQ, V, kappa, r)
    E_orb = orbital_energy_from_dirac(P, Q, LP, LQ, grid)
    return {
        "V": V,
        "wavefunctions": {"P": P, "Q": Q, "dPdr": dP, "dQdr": dQ},
        "E_orb": E_orb,
    }


def test_gate_pass_on_analytic_h1s(h1s_batch, r_grid):
    out = _analytic_out(h1s_batch, r_grid)
    rep = check_gate_a(
        out, h1s_batch, r_grid,
        cos_threshold=0.95,
        pde_threshold=100.0,
        e_orb_meV_threshold=500.0,
    )
    assert rep.cos_min >= 0.95
    e_ref = hydrogenic_energy(1, 1)
    assert abs(float(out["E_orb"][0, 0]) - e_ref) * 27.211 < 500.0
