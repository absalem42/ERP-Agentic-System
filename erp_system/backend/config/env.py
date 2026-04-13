from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILES = (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local")


@lru_cache(maxsize=1)
def load_environment() -> None:
    if os.getenv("ERP_SKIP_DOTENV") == "1":
        return

    existing_keys = set(os.environ)
    merged_values: dict[str, str] = {}

    for env_file in ENV_FILES:
        if not env_file.exists():
            continue
        for key, value in dotenv_values(env_file).items():
            if value is None:
                continue
            merged_values[key] = str(value)

    for key, value in merged_values.items():
        if key in existing_keys:
            continue
        os.environ[key] = value
