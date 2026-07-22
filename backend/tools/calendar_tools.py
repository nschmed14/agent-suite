"""Google Calendar helpers for the local Agent Suite assistant.

The tool layer is designed to stay local-first. It loads OAuth credentials from
local files and intentionally does not transmit tokens anywhere else.
"""

from __future__ import annotations

from typing import Any, Dict, List


def get_today_events() -> List[Dict[str, Any]]:
    """Return today's calendar events. Fallback placeholder until OAuth is configured."""
    return [{"summary": "No calendar access configured yet", "start": None, "end": None}]


def get_upcoming_events(days: int) -> List[Dict[str, Any]]:
    return get_today_events()


def create_event(summary: str, start_time: str, end_time: str, description: str | None = None) -> Dict[str, Any]:
    return {"ok": False, "message": "Calendar write support requires local OAuth setup."}


def find_free_slots(days: int, duration_minutes: int) -> List[Dict[str, Any]]:
    return []


def delete_event(event_id: str) -> Dict[str, Any]:
    return {"ok": False, "message": "Delete support requires local OAuth setup."}
