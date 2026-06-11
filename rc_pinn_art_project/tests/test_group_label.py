"""R0.3 — single_valence / multi_electron grouping rule."""

import pytest

from pinn_art.data.config_enum import (
    classify_group,
    complete_core,
    n_open_electrons,
    parse_config_tuples,
)


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ("1s1", "single_valence"),          # H
        ("1s2 2s1", "single_valence"),      # Li-like
        ("1s2 2s2 2p1", "single_valence"),  # B-like (one e- over closed 2s)
        ("1s2 2s2 2p6 3s1", "single_valence"),  # Na-like
        ("1s2", "multi_electron"),          # closed shell, no valence e-
        ("1s2 2s2 2p2", "multi_electron"),  # C-like open p^2
        ("1s2 2s1 2p1", "multi_electron"),  # two open shells
        ("1s2 2s2 2p6 3s2 3p6 3d6 4s2", "multi_electron"),  # Fe ground
    ],
)
def test_classify_group(cfg, expected):
    assert classify_group(parse_config_tuples(cfg)) == expected


def test_n_open_electrons_counts_only_unfilled():
    # closed 1s2/2s2 contribute 0; only the lone 2p counts
    assert n_open_electrons(parse_config_tuples("1s2 2s2 2p1")) == 1
    assert n_open_electrons(parse_config_tuples("1s2 2s2 2p2")) == 2
    assert n_open_electrons(parse_config_tuples("1s2")) == 0


def test_complete_core_refills_omitted_core():
    # NIST drops the 1s2 core for C I: "2s2 2p2" + nele=6 -> full config
    shells = complete_core(parse_config_tuples("2s2 2p2"), nele=6)
    assert shells == [(1, 0, 2), (2, 0, 2), (2, 1, 2)]
    # already-complete configs are unchanged
    full = parse_config_tuples("1s2 2s1")
    assert complete_core(full, nele=3) == full
