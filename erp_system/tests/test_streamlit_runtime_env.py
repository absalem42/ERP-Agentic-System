import os


def test_bootstrap_runtime_environment_sets_missing_values_from_secrets(monkeypatch):
    from frontend.runtime_env import bootstrap_runtime_environment

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

    bootstrap_runtime_environment(
        {
            "AZURE_OPENAI_API_KEY": "secret-key",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
        }
    )

    assert os.getenv("AZURE_OPENAI_API_KEY") == "secret-key"
    assert os.getenv("AZURE_OPENAI_ENDPOINT") == "https://example.openai.azure.com"


def test_bootstrap_runtime_environment_does_not_override_existing_env(monkeypatch):
    from frontend.runtime_env import bootstrap_runtime_environment

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "existing-key")

    bootstrap_runtime_environment({"AZURE_OPENAI_API_KEY": "secret-key"})

    assert os.getenv("AZURE_OPENAI_API_KEY") == "existing-key"


def test_bootstrap_runtime_environment_ignores_unavailable_streamlit_secrets(monkeypatch):
    from frontend.runtime_env import bootstrap_runtime_environment

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    class UnavailableSecrets:
        def __contains__(self, key):
            raise RuntimeError("secrets.toml missing")

        def __getitem__(self, key):
            raise RuntimeError("secrets.toml missing")

    bootstrap_runtime_environment(UnavailableSecrets())

    assert os.getenv("AZURE_OPENAI_API_KEY") is None
