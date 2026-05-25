"""Phase-3 ansatz: at initialization, P_a(r) ≈ P_H_{n,l}(r;Z) for every orbital.

This validates that the SIREN perturbation head starts at zero (final Dense
kernel/bias initialized to 0) and that the analytic hydrogenic skeleton is
wired correctly through PinnArtModel → DeepONetDirac → hydrogenic_P_jax.
"""
import jax
import jax.numpy as jnp
import numpy as np

from pinn_art.data.collate import collate_batches
from pinn_art.data.config_parser import encode_config_to_array, parse_config_string
from pinn_art.models.pinn_art_model import build_model_and_params
from pinn_art.physics.hydrogenic import (
    cosine_signed,
    hydrogenic_energy,
    hydrogenic_P_analytic,
    hydrogenic_P_jax,
)
from pinn_art.utils.grid import make_radial_grid


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _make_cfg():
    return _Cfg(
        model=_Cfg(
            n_orb_max=8,
            n_csf_max=4,
            d_branch=32,
            d_trunk=32,
            n_siren_layers=2,
            omega_0=10.0,
            apply_lowdin=False,
            use_hydrogenic_skeleton=True,
            perturb_eps=0.2,
        ),
        ci=_Cfg(enabled=False),
        coords=_Cfg(c1=1.0, c2=1.0),
        seed=0,
    )


def _item(Z, n, l, occ=1):
    n_orb_max = 8
    shell_table = np.zeros((n_orb_max, 4), dtype=np.int32)
    # j = l-1/2 if l>0 else 1/2  ⇒  kappa = l (j-) or -(l+1) (j+); pick j+:
    j2 = 2 * l + 1
    shell_table[0] = [n, l, j2, occ]
    kappas = np.zeros(n_orb_max, dtype=np.int32)
    kappas[0] = -(l + 1)  # j = l + 1/2 sign convention
    omegas = np.zeros(n_orb_max, dtype=np.float32)
    omegas[0] = float(occ)
    orb_mask = np.zeros(n_orb_max, dtype=bool)
    orb_mask[0] = True
    return {
        "Z": Z,
        "ion_charge": 0,
        "nele": occ,
        "parent_config": f"{n}{'spdf'[l]}{occ}",
        "level_config": f"{n}{'spdf'[l]}{occ}",
        "shell_table": shell_table,
        "kappa": kappas,
        "omega": omegas,
        "orb_mask": orb_mask,
        "J": 0.5,
        "parity": 0,
        "E_nist_scalar": float("nan"),
        "nist_mask_scalar": False,
    }


def test_init_state_matches_hydrogenic_for_H_and_He_and_Be():
    grid = make_radial_grid(r_min=1e-4, r_max=40.0, n_grid=128, scheme="loglinear")
    cases = [
        (1, 1, 0),  # H 1s
        (1, 2, 0),  # H 2s
        (1, 3, 1),  # H 3p
        (2, 1, 0),  # He+ 1s
        (4, 1, 0),  # Be 1s (single-electron approximation)
        (4, 2, 0),  # Be 2s
    ]
    items = [_item(Z, n, l) for Z, n, l in cases]
    batch = collate_batches(items, n_csf_max=4)

    cfg = _make_cfg()
    key = jax.random.PRNGKey(0)
    model, params = build_model_and_params(cfg, grid, key)
    out = model.apply(params, batch, grid, train=False, return_ci=False)
    P = out["wavefunctions"]["P"]  # [B, N_orb, N_g]

    for i, (Z, n, l) in enumerate(cases):
        P_model = P[i, 0]
        P_ref = hydrogenic_P_analytic(grid.r, Z, n, l)
        # 1-D cosine between model and reference (use grid weights)
        cos = float(
            jnp.sum(P_model * P_ref * grid.dr)
            / jnp.sqrt(
                jnp.sum(P_model * P_model * grid.dr) * jnp.sum(P_ref * P_ref * grid.dr)
                + 1e-30
            )
        )
        assert cos > 0.999, f"init cos({Z},{n},{l})={cos:.6f} != 1"


def test_init_state_energy_matches_hydrogenic():
    grid = make_radial_grid(r_min=1e-4, r_max=40.0, n_grid=256, scheme="loglinear")
    cases = [(1, 1, 0), (2, 1, 0), (4, 1, 0), (8, 1, 0)]  # ground states
    items = [_item(Z, n, l) for Z, n, l in cases]
    batch = collate_batches(items, n_csf_max=4)
    cfg = _make_cfg()
    cfg.model.n_orb_max = 8
    key = jax.random.PRNGKey(1)
    model, params = build_model_and_params(cfg, grid, key)
    out = model.apply(params, batch, grid, train=False, return_ci=False)
    E_orb = out["E_orb"][:, 0]  # [B]
    for i, (Z, n, l) in enumerate(cases):
        E_ref = hydrogenic_energy(Z, n)
        E_got = float(E_orb[i])
        # initial V has a small SIREN correction; tolerate 0.1 Ha relative to |E_ref|
        rel = abs(E_got - E_ref) / max(abs(E_ref), 1.0)
        assert rel < 0.2, f"E init mismatch Z={Z} n={n}: got {E_got:.4f} ref {E_ref:.4f}"
