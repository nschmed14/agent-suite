"""Run a simple local SQLite query."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteQueryTool:
    """Execute SQL against a local SQLite database file."""

    def query(self, db_path: str, sql: str) -> list[tuple]:
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"Database does not exist: {db_path}")
        connection = sqlite3.connect(path)
        try:
            rows = connection.execute(sql).fetchall()
        finally:
            connection.close()
        return rows
