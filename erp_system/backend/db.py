from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from backend.config.env import load_environment


load_environment()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "databases" / "erp.db"


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path:
        return Path(db_path)

    env_db_path = os.getenv("DB_PATH")
    if env_db_path:
        return Path(env_db_path)

    return DEFAULT_DB_PATH


@contextmanager
def get_db(db_path: str | Path | None = None):
    resolved_path = resolve_db_path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(resolved_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(query: str, params: Iterable[Any] = (), db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]


def fetch_one(query: str, params: Iterable[Any] = (), db_path: str | Path | None = None) -> dict[str, Any] | None:
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        row = cursor.fetchone()
        return dict(row) if row else None


def execute(
    query: str,
    params: Iterable[Any] = (),
    db_path: str | Path | None = None,
    *,
    commit: bool = True,
) -> int:
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        if commit:
            conn.commit()
        return cursor.lastrowid
