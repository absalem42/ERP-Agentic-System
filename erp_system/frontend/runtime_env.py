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
    if not secret_values:
        return

    for key in RUNTIME_ENV_KEYS:
        if os.getenv(key):
            continue

        if key in secret_values and secret_values[key] is not None:
            os.environ[key] = str(secret_values[key])
