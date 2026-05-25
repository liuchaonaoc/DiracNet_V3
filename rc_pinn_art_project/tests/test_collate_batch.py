from pathlib import Path

import pandas as pd

from pinn_art.data.collate import collate_batches
from pinn_art.data.dataset import ManifestDataset


def test_collate_batch_shapes(tmp_path):
    df = pd.DataFrame(
        [
            {
                "Z": 1, "ion_charge": 0, "nele": 1,
                "parent_config": "1s1", "level_config": "1s1",
                "level_eV": 0.0, "has_nist_level": True,
                "J": 0.5, "parity": 0, "term": "2S",
            },
        ]
    )
    p = tmp_path / "m.parquet"
    df.to_parquet(p)
    ds = ManifestDataset(p, n_orb_max=16, n_csf_max=8)
    batch = collate_batches([ds[0]], n_csf_max=8)
    assert batch["Z"].shape == (1,)
    assert batch["orb_mask"].shape == (1, 16)
