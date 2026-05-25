from pinn_art.data.config_parser import parse_config_string


def test_parse_1s1():
    shells = parse_config_string("1s1")
    assert len(shells) == 1
    assert shells[0].n == 1
    assert shells[0].kappa == -1
