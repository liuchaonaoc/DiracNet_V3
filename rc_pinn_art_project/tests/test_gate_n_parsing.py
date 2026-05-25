"""Gate A reads principal quantum number n from manifest / shell_table."""

import jax.numpy as jnp
import numpy as np

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset
from pinn_art.physics.hydrogenic import cosine_signed, hydrogenic_P_analytic
from pinn_art.training.gate_a import (
    check_gate_a,
    parse_n_l_from_level_config,
    principal_quantum_from_batch,
)


def test_parse_n_l_from_level_config():
    assert parse_n_l_from_level_config("1s1") == (1, 0)
    assert parse_n_l_from_level_config("6s1") == (6, 0)
    assert parse_n_l_from_level_config("2p3") == (2, 1)


def test_principal_quantum_from_shell_table(h1s_batch):
    n, l = principal_quantum_from_batch(h1s_batch)
    assert n == 1 and l == 0


def test_gate_cos_high_when_reference_matches_n(h1s_batch, r_grid):
    """3s reference vs 3s model P should cos≈1; vs 1s reference should be low."""
    Z = 3
    n = 3
    P_model = hydrogenic_P_analytic(r_grid.r, Z, n, 0)[None, None, :]
    Q = P_model * 1e-4
    dP = jnp.gradient(P_model, r_grid.r, axis=-1)
    dQ = jnp.zeros_like(Q)
    out = {
        "V": (-float(Z) / jnp.clip(r_grid.r, 1e-8))[None, :],
        "wavefunctions": {"P": P_model, "Q": Q, "dPdr": dP, "dQdr": dQ},
        "E_orb": jnp.array([[-float(Z) ** 2 / 18.0]]),
    }
    batch = dict(h1s_batch)
    batch["Z"] = jnp.array([Z], dtype=jnp.int32)
    st = np.zeros((1, 16, 4), dtype=np.int32)
    st[0, 0, 0] = n
    st[0, 0, 1] = 0
    batch["shell_table"] = jnp.asarray(st)

    rep3 = check_gate_a(out, batch, r_grid, cos_threshold=0.99, pde_threshold=1e6, e_orb_meV_threshold=1e6)
    assert rep3.n == 3
    assert rep3.cos_min >= 0.99

    rep1 = check_gate_a(
        out, batch, r_grid,
        n=1, l=0,
        cos_threshold=0.99, pde_threshold=1e6, e_orb_meV_threshold=1e6,
    )
    assert rep1.cos_min < 0.5


def test_manifest_row_n6_differs_from_n1(tmp_path):
    from pinn_art.data.config_parser import encode_config_to_array, parse_config_string
    import pandas as pd

    rows = []
    for n in (1, 6):
        rows.append({
            "Z": 1, "ion_charge": 0, "nele": 1,
            "parent_config": "1s1", "level_config": f"{n}s1",
            "J": 0.5, "parity": 0, "term": "2S", "level_eV": 0.0,
            "has_nist_level": True, "element": "H",
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "h.parquet"
    df.to_parquet(path, index=False)
    ds = ManifestDataset(path, n_orb_max=16)
    b1 = collate_batches([ds[0]], n_csf_max=8)
    b6 = collate_batches([ds[1]], n_csf_max=8)
    assert principal_quantum_from_batch(b1) == (1, 0)
    assert principal_quantum_from_batch(b6) == (6, 0)
