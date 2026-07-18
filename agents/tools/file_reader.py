"""Read a local file from disk.

This tool is intentionally simple and local-only. It does not contact any service.
"""

from __future__ import annotations

from pathlib import Path


class FileReaderTool:
    """Read the contents of a local file."""

    def read(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        return path.read_text(encoding="utf-8")
