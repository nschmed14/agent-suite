"""Parse a simple local calendar-like line into a structured payload."""

from __future__ import annotations

from typing import Dict


class CalendarParserTool:
    """Create a lightweight calendar entry payload from a text line."""

    def parse(self, line: str) -> Dict[str, str]:
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if len(parts) < 2:
            return {"summary": line, "datetime": "unknown"}
        return {"summary": parts[0], "datetime": parts[1]}
