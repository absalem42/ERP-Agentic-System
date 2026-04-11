import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def get_default_db_path() -> str:
    """Return the default project database path."""
    return str(Path(__file__).resolve().parent.parent / "databases" / "erp.db")


def resolve_db_path() -> str:
    """Resolve the active database path at call time."""
    return os.getenv("DB_PATH", get_default_db_path())


@contextmanager
def get_db():
    db_path = resolve_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
