from backend.config.llm import DEFAULT_GROQ_MODEL, get_groq_model, has_llm_credentials


def test_groq_model_defaults(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert get_groq_model() == DEFAULT_GROQ_MODEL


def test_has_llm_credentials_uses_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert has_llm_credentials() is False

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert has_llm_credentials() is True
