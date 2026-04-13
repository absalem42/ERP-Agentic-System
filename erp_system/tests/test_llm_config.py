def test_resolve_azure_openai_settings_supports_full_target_uri(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_TARGET_URI",
        "https://ai-la.cognitiveservices.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
    )
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    from backend.config.llm import has_llm_credentials, resolve_azure_openai_settings

    settings = resolve_azure_openai_settings()

    assert settings["endpoint"] == "https://ai-la.cognitiveservices.azure.com"
    assert settings["deployment"] == "gpt-4.1"
    assert settings["api_version"] == "2025-01-01-preview"
    assert has_llm_credentials() is True


def test_resolve_azure_openai_settings_normalizes_full_endpoint_value(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://ai-la.cognitiveservices.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
    )
    monkeypatch.delenv("AZURE_OPENAI_TARGET_URI", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    from backend.config.llm import resolve_azure_openai_settings

    settings = resolve_azure_openai_settings()

    assert settings["endpoint"] == "https://ai-la.cognitiveservices.azure.com"
    assert settings["deployment"] == "gpt-4.1"
    assert settings["api_version"] == "2025-01-01-preview"


def test_resolve_azure_openai_settings_uses_project_default_deployment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ai-la.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    monkeypatch.delenv("AZURE_OPENAI_TARGET_URI", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    from backend.config.llm import has_llm_credentials, resolve_azure_openai_settings

    settings = resolve_azure_openai_settings()

    assert settings["endpoint"] == "https://ai-la.cognitiveservices.azure.com"
    assert settings["deployment"] == "gpt-4.1"
    assert settings["api_version"] == "2024-12-01-preview"
    assert has_llm_credentials() is True


def test_resolve_azure_openai_settings_accepts_model_alias(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://ai-la.cognitiveservices.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.delenv("AZURE_OPENAI_TARGET_URI", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    from backend.config.llm import resolve_azure_openai_settings

    settings = resolve_azure_openai_settings()

    assert settings["deployment"] == "gpt-4.1-mini"
