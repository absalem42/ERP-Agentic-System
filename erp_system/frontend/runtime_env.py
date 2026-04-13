from __future__ import annotations

import os
from typing import Any, Mapping


RUNTIME_ENV_KEYS = (
    "API_URL",
    "DB_PATH",
    "ERP_SAMPLE_DB_PATH",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_TARGET_URI",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_MODEL",
    "OPENAI_MODEL",
    "AZURE_OPENAI_API_VERSION",
)


def bootstrap_runtime_environment(secret_values: Mapping[str, Any] | None = None) -> None:
    if secret_values is None:
        return

    for key in RUNTIME_ENV_KEYS:
        if os.getenv(key):
            continue
        try:
            if key not in secret_values:
                continue
            value = secret_values[key]
        except Exception:
            continue
        if value is not None:
            os.environ[key] = str(value)
