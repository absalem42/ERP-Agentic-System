import types

from backend.config import llm
from backend.config.llm import DEFAULT_GROQ_MODEL, get_groq_model, has_llm_credentials


def test_groq_model_defaults(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert get_groq_model() == DEFAULT_GROQ_MODEL


def test_has_llm_credentials_uses_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert has_llm_credentials() is False

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert has_llm_credentials() is True


def test_get_llm_uses_http_groq_fallback_when_langchain_package_is_unavailable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setattr(llm, "GROQ_AVAILABLE", False)
    monkeypatch.setattr(llm, "OLLAMA_AVAILABLE", False)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "groq http ok"}}]}

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    client = llm.get_llm()
    response = client.invoke("ping")
    content = response.content if hasattr(response, "content") else response

    assert content == "groq http ok"
