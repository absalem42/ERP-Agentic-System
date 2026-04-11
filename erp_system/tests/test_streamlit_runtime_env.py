import os

from frontend.runtime_env import bootstrap_runtime_environment


def test_bootstrap_runtime_environment_sets_missing_values_from_secrets(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    bootstrap_runtime_environment(
        {
            "GROQ_API_KEY": "secret-key",
            "GROQ_MODEL": "llama-3.1-8b-instant",
        }
    )

    assert os.getenv("GROQ_API_KEY") == "secret-key"
    assert os.getenv("GROQ_MODEL") == "llama-3.1-8b-instant"


def test_bootstrap_runtime_environment_does_not_override_existing_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "existing-key")

    bootstrap_runtime_environment({"GROQ_API_KEY": "secret-key"})

    assert os.getenv("GROQ_API_KEY") == "existing-key"
