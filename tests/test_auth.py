from openhands_discovery.auth import keys_match, read_api_key


def test_read_api_key_reads_the_matching_line(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("OTHER=1\nOH_SESSION_API_KEYS_0=secret\n", encoding="utf-8")
    assert read_api_key(str(env_file)) == "secret"


def test_read_api_key_keeps_the_value_literal(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text('OH_SESSION_API_KEYS_0="quoted"\n', encoding="utf-8")
    assert read_api_key(str(env_file)) == '"quoted"'


def test_read_api_key_returns_empty_on_a_missing_file(tmp_path):
    assert read_api_key(str(tmp_path / "missing")) == ""


def test_keys_match_rejects_an_empty_expected_key():
    assert keys_match("secret", "secret")
    assert not keys_match("secret", "")
    assert not keys_match("", "secret")
    assert not keys_match("wrong", "secret")