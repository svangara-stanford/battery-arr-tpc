import pytest

from battery_aar.agents.llm_client import llm_startup_summary, load_llm_client_config, make_agent


ENV_KEYS = [
    "OPEN_BATTERY_AGENTS_API_KEY",
    "OPEN_BATTERY_AGENTS_BASE_URL",
    "OPEN_BATTERY_AGENTS_MODEL",
    "OPEN_BATTERY_AGENTS_ALIAS",
    "OPEN_BATTERY_AGENTS_ALIAS_HEADER",
    "OPEN_BATTERY_AGENTS_EXTRA_HEADERS_JSON",
    "STANFORD_AI_API_KEY",
    "STANFORD_AI_BASE_URL",
    "STANFORD_AI_ALIAS",
    "STANFORD_AI_PLAYGROUND_API_KEY",
    "STANFORD_AI_PLAYGROUND_BASE_URL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_llm_client_reads_open_battery_alias_and_header(monkeypatch):
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_API_KEY", "secret-key")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_MODEL", "stanford-model")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_ALIAS", "my-alias")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_ALIAS_HEADER", "X-Stanford-Alias")
    config = load_llm_client_config()

    assert config.api_key == "secret-key"
    assert config.base_url == "https://gateway.example/v1"
    assert config.model == "stanford-model"
    assert config.alias == "my-alias"
    assert config.default_headers == {"X-Stanford-Alias": "my-alias"}
    summary = config.safe_summary()
    assert summary["alias_configured"] is True
    assert summary["alias_header_configured"] is True
    assert "secret-key" not in str(summary)
    assert "my-alias" not in str(summary)


def test_llm_client_falls_back_to_stanford_alias_env(monkeypatch):
    monkeypatch.setenv("STANFORD_AI_API_KEY", "stanford-secret")
    monkeypatch.setenv("STANFORD_AI_BASE_URL", "https://stanford.example/v1")
    monkeypatch.setenv("STANFORD_AI_ALIAS", "stanford-alias")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_ALIAS_HEADER", "X-Alias")
    config = load_llm_client_config()

    assert config.api_key == "stanford-secret"
    assert config.base_url == "https://stanford.example/v1"
    assert config.alias == "stanford-alias"
    assert config.default_headers == {"X-Alias": "stanford-alias"}


def test_llm_client_supports_arbitrary_extra_headers(monkeypatch):
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_API_KEY", "secret")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_ALIAS", "alias")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_ALIAS_HEADER", "X-Alias")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_EXTRA_HEADERS_JSON", '{"X-Team": "battery", "X-Trace": 123}')
    config = load_llm_client_config()

    assert config.default_headers == {"X-Team": "battery", "X-Trace": "123", "X-Alias": "alias"}
    summary = llm_startup_summary()
    assert summary["extra_headers_configured"] is True
    assert summary["default_header_names"] == ["X-Alias", "X-Team", "X-Trace"]


def test_llm_client_alias_without_header_is_safe_noop(monkeypatch):
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_API_KEY", "secret")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_ALIAS", "alias")
    config = load_llm_client_config()
    assert config.alias == "alias"
    assert config.default_headers == {}
    assert config.safe_summary()["alias_configured"] is True
    assert config.safe_summary()["alias_header_configured"] is False


def test_llm_client_invalid_extra_headers_json(monkeypatch):
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_EXTRA_HEADERS_JSON", "not-json")
    with pytest.raises(ValueError):
        load_llm_client_config()


def test_make_agent_uses_new_stanford_api_key_alias(monkeypatch):
    monkeypatch.setenv("STANFORD_AI_API_KEY", "secret")
    monkeypatch.setenv("STANFORD_AI_ALIAS", "alias")
    monkeypatch.setenv("OPEN_BATTERY_AGENTS_ALIAS_HEADER", "X-Alias")
    agent = make_agent("agent", offline=False)
    assert agent.config.alias == "alias"
    assert agent.config.default_headers == {"X-Alias": "alias"}
