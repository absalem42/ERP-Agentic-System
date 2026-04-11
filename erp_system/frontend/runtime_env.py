import os
from typing import Any, Mapping


RUNTIME_ENV_KEYS = (
    "API_URL",
    "ERP_RUNTIME_MODE",
    "ERP_ENABLE_DIRECT_AI",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "DB_PATH",
)


def bootstrap_runtime_environment(secret_values: Mapping[str, Any] | None = None) -> None:
    """Populate runtime env vars from Streamlit secrets when they are missing."""
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
