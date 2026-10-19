from openhands_discovery.__main__ import self_check


def test_self_check_passes(capsys):
    self_check()

    assert capsys.readouterr().out.strip() == "self-check passed"