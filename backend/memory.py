"""Persistent local memory for Agent Suite.

The implementation stores lightweight memories in SQLite and, when available,
uses ChromaDB for vector-style search over local data. No cloud service is used.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")

try:  # pragma: no cover - optional dependency
    import chromadb
except ImportError:  # pragma: no cover - fallback path
    chromadb = None


class LocalMemory:
    """Persist agent memory locally with SQLite and optional ChromaDB."""

    def __init__(self, sqlite_path: str, chroma_path: str) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.chroma_path = Path(chroma_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._collection = None
        if chromadb is not None:
            try:
                client = chromadb.PersistentClient(path=str(self.chroma_path))
                self._collection = client.get_or_create_collection("agent_memories")
            except Exception:  # pragma: no cover - local fallback
                self._collection = None

    def _ensure_schema(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.sqlite_path.exists():
            self.sqlite_path.touch()
        connection = sqlite3.connect(self.sqlite_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def add_memory(self, agent_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Store one memory entry in SQLite and optionally ChromaDB."""
        connection = sqlite3.connect(self.sqlite_path)
        payload = json.dumps(metadata or {}, sort_keys=True)
        try:
            cursor = connection.execute(
                "INSERT INTO memories (agent_id, content, metadata, created_at) VALUES (?, ?, ?, datetime('now'))",
                (agent_id, content, payload),
            )
            connection.commit()
            memory_id = cursor.lastrowid
        finally:
            connection.close()

        if self._collection is not None:
            try:
                self._collection.add(
                    documents=[content],
                    metadatas=[{"agent_id": agent_id, **(metadata or {})}],
                    ids=[f"memory-{memory_id}"],
                )
            except Exception:  # pragma: no cover - local fallback
                pass
        return memory_id

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories. Uses ChromaDB when available, otherwise SQLite."""
        if self._collection is not None:
            try:
                results = self._collection.query(query_texts=[query], n_results=limit)
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                return [
                    {"content": doc, "metadata": meta or {}}
                    for doc, meta in zip(docs, metas)
                ]
            except Exception:  # pragma: no cover - local fallback
                pass

        connection = sqlite3.connect(self.sqlite_path)
        try:
            rows = connection.execute(
                "SELECT content, metadata FROM memories WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        finally:
            connection.close()

        return [{"content": content, "metadata": json.loads(metadata)} for content, metadata in rows]
