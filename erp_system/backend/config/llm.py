from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse
from typing import Any

import requests

from backend.config.env import load_environment


load_environment()

DEFAULT_AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
PROJECT_DEFAULT_AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"


@dataclass
class LLMResponse:
    content: str


class MockLLM:
    def invoke(self, prompt: str, **_: Any) -> LLMResponse:
        return LLMResponse(content=f"Mock LLM response unavailable for prompt: {prompt[:120]}")


class AzureOpenAILLM:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = DEFAULT_AZURE_OPENAI_API_VERSION,
        temperature: float = 0.1,
        timeout: float = 20.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version
        self.temperature = temperature
        self.timeout = timeout

    def invoke(self, prompt: str, **_: Any) -> LLMResponse:
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        response = requests.post(
            url,
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return LLMResponse(content=data["choices"][0]["message"]["content"].strip())


def _extract_endpoint_parts(value: str) -> tuple[str, str, str]:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        return value.rstrip("/"), "", ""

    api_version = parse_qs(parsed.query).get("api-version", [""])[0]
    deployment = ""
    endpoint = f"{parsed.scheme}://{parsed.netloc}"

    deployment_marker = "/openai/deployments/"
    completion_suffix = "/chat/completions"
    if deployment_marker in parsed.path and completion_suffix in parsed.path:
        _, deployment_path = parsed.path.split(deployment_marker, 1)
        deployment = deployment_path.split(completion_suffix, 1)[0].strip("/")

    return endpoint.rstrip("/"), deployment, api_version


def _resolve_deployment_name() -> str:
    for env_key in ("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_MODEL", "OPENAI_MODEL"):
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return ""


def resolve_azure_openai_settings() -> dict[str, str]:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment = _resolve_deployment_name()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_OPENAI_API_VERSION).strip()

    target_uri = os.getenv("AZURE_OPENAI_TARGET_URI", "").strip()
    source_uri = target_uri or endpoint
    if source_uri:
        parsed_endpoint, parsed_deployment, parsed_version = _extract_endpoint_parts(source_uri)
        endpoint = parsed_endpoint or endpoint
        deployment = deployment or parsed_deployment
        api_version = parsed_version or api_version

    # Azure requires a deployment name. This project's public demo uses `gpt-4.1`
    # unless explicitly overridden in the environment.
    if api_key and endpoint and not deployment:
        deployment = PROJECT_DEFAULT_AZURE_OPENAI_DEPLOYMENT

    return {
        "api_key": api_key,
        "endpoint": endpoint.rstrip("/"),
        "deployment": deployment,
        "api_version": api_version or DEFAULT_AZURE_OPENAI_API_VERSION,
    }


def has_llm_credentials() -> bool:
    settings = resolve_azure_openai_settings()
    return all([settings["api_key"], settings["endpoint"], settings["deployment"]])


def get_provider_mode() -> str:
    return "azure" if has_llm_credentials() else "fallback"


def get_llm():
    if has_llm_credentials():
        settings = resolve_azure_openai_settings()
        return AzureOpenAILLM(
            endpoint=settings["endpoint"],
            api_key=settings["api_key"],
            deployment=settings["deployment"],
            api_version=settings["api_version"],
        )
    return MockLLM()


def coerce_text(response: Any) -> str:
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()
