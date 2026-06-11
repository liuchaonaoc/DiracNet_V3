"""R0.1 — configuration enumeration sanity (Z<=26, n<=10)."""

from pinn_art.data.config_enum import (
    config_to_string,
    enumerate_all,
    enumerate_levels,
    ground_config,
)


def test_ground_config_known_atoms():
    assert config_to_string(ground_config(1)) == "1s1"
    assert config_to_string(ground_config(2)) == "1s2"
    assert config_to_string(ground_config(3)) == "1s2 2s1"
    assert config_to_string(ground_config(10)) == "1s2 2s2 2p6"
    assert config_to_string(ground_config(11)) == "1s2 2s2 2p6 3s1"
    # Fe: Aufbau gives 4s filled before 3d
    assert config_to_string(ground_config(26)) == "1s2 2s2 2p6 3s2 3p6 3d6 4s2"


def test_enumerate_levels_hydrogen_rydberg():
    levels = enumerate_levels(1, 0, n_max=10)
    cfgs = {lv.level_config for lv in levels}
    # ground + ns/np/.../ up to n=10
    assert "1s1" in cfgs
    assert "2p1" in cfgs and "10s1" in cfgs
    # every H level is single_valence
    assert all(lv.group == "single_valence" for lv in levels)
    # one ground row tagged
    assert sum(lv.is_ground for lv in levels) >= 1


def test_enumerate_fine_structure_doublet():
    levels = enumerate_levels(3, 0, n_max=4)  # Li
    p_levels = [lv for lv in levels if lv.level_config == "1s2 2p1"]
    twice_js = sorted(lv.twice_j for lv in p_levels)
    assert twice_js == [1, 3]  # 2P_{1/2}, 2P_{3/2}


def test_enumerate_z26_n10_counts():
    levels = enumerate_all(1, 26, n_max=10, ion_charges="all")
    assert len(levels) > 10000
    # all (Z, ion) pairs present: nele in 1..Z
    pairs = {(lv.Z, lv.ion_charge) for lv in levels}
    assert (26, 0) in pairs and (26, 25) in pairs and (1, 0) in pairs
    # n_active never exceeds n_max
    assert max(lv.n_active for lv in levels) <= 10
    # both groups populated
    groups = {lv.group for lv in levels}
    assert groups == {"single_valence", "multi_electron"}
