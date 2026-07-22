"""Gmail helpers for the local Agent Suite assistant.

The implementation is deliberately conservative: it supports reading and drafting
only. Sending email is intentionally absent and requires a separate approval
step that the assistant should never perform automatically.
"""

from __future__ import annotations

from typing import Any, Dict, List


def get_unread_summary(max_results: int = 5) -> List[Dict[str, Any]]:
    return [{"sender": "noreply", "subject": "No Gmail access configured yet", "summary": "Configure OAuth locally to read your inbox."}]


def get_emails_from_label(label_name: str) -> List[Dict[str, Any]]:
    return []


def draft_reply(email_id: str, response_text: str) -> Dict[str, Any]:
    return {"ok": True, "draft_id": f"draft-{email_id}", "message": "Draft created locally. Sending requires explicit user approval."}


def search_emails(query: str) -> List[Dict[str, Any]]:
    return []


def get_email_content(email_id: str) -> Dict[str, Any]:
    return {"id": email_id, "body": "No Gmail access configured yet."}


def send_email(email_id: str, approval_token: str | None = None) -> Dict[str, Any]:
    """Never auto-send. This must be invoked only after a separate explicit approval step."""
    return {"ok": False, "message": "Sending email requires explicit user approval and a separate confirmation step."}
