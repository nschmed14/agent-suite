"""Google Calendar helpers for local OAuth-based calendar access."""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class CalendarAuthenticationError(RuntimeError):
    """Raised when Google Calendar authentication cannot be completed."""

SCOPES = ["https://www.googleapis.com/auth/calendar"]
BASE_DIR = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = BASE_DIR / "credentials" / "calendar_credentials.json"
TOKEN_PATH = BASE_DIR / "credentials" / "token.pickle"


def _load_client_config() -> dict[str, Any]:
    """Load Google OAuth client configuration from the environment or a local file."""
    env_json = os.getenv("GOOGLE_OAUTH_CREDENTIALS_JSON")
    if env_json:
        return json.loads(env_json)

    env_path = os.getenv("GOOGLE_OAUTH_CREDENTIALS_FILE")
    if env_path:
        credentials_path = Path(env_path).expanduser()
        if not credentials_path.exists():
            raise FileNotFoundError(
                f"Google Calendar credentials file configured at {credentials_path} does not exist."
            )
        return json.loads(credentials_path.read_text())

    if CREDENTIALS_PATH.exists():
        return json.loads(CREDENTIALS_PATH.read_text())

    raise FileNotFoundError(
        "Google Calendar credentials were not provided. Set GOOGLE_OAUTH_CREDENTIALS_JSON or GOOGLE_OAUTH_CREDENTIALS_FILE."
    )


def _serialize_datetime(value: datetime | str) -> str:
    """Convert a datetime-like value to an RFC3339 string for Google Calendar."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        return value
    raise TypeError("start_time and end_time must be datetime objects or ISO strings")


def authenticate_google_calendar():
    """Authenticate to Google Calendar using a local loopback OAuth flow and persist the token locally."""
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    client_config = _load_client_config()

    creds: Credentials | None = None
    if TOKEN_PATH.exists():
        try:
            with TOKEN_PATH.open("rb") as token_file:
                creds = pickle.load(token_file)
        except Exception as exc:
            print(f"Existing token could not be loaded: {exc}. Starting a fresh OAuth flow.")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                print(f"Token refresh failed: {exc}. Starting a fresh OAuth flow.")
                creds = None

        if not creds or not creds.valid:
            try:
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                print("Starting Google Calendar authorization flow...")
                auth_url, _ = flow.authorization_url(prompt="consent")
                print("Please visit this URL to authorize access:")
                print(auth_url)
                try:
                    creds = flow.run_local_server(port=0)
                except Exception as exc:
                    raise CalendarAuthenticationError(
                        "Google Calendar authentication requires manual authorization because the local callback flow failed. "
                        f"Please complete sign-in at: {auth_url}"
                    ) from exc
            except CalendarAuthenticationError:
                raise
            except Exception as exc:
                raise CalendarAuthenticationError(f"Google Calendar authentication failed: {exc}") from exc

            TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            with TOKEN_PATH.open("wb") as token_file:
                pickle.dump(creds, token_file)
            print(f"Saved Google Calendar token to {TOKEN_PATH}")

    return build("calendar", "v3", credentials=creds)


def get_events(service: Any | None = None, days: int = 7) -> List[dict[str, Any]]:
    """Return upcoming calendar events for the next number of days."""
    if service is None:
        service = authenticate_google_calendar()
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(days=days)

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now.isoformat().replace("+00:00", "Z"),
            timeMax=window_end.isoformat().replace("+00:00", "Z"),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = events_result.get("items", [])
    return [
        {
            "summary": event.get("summary", "(No title)"),
            "start": event.get("start", {}).get("dateTime") or event.get("start", {}).get("date"),
            "end": event.get("end", {}).get("dateTime") or event.get("end", {}).get("date"),
        }
        for event in events
    ]


def create_event(summary: str, start_time: datetime | str, end_time: datetime | str, description: str | None = None) -> str:
    """Create a new event in the user's primary calendar and return its ID."""
    try:
        service = authenticate_google_calendar()
        event_body = {
            "summary": summary,
            "description": description or "",
            "start": {"dateTime": _serialize_datetime(start_time)},
            "end": {"dateTime": _serialize_datetime(end_time)},
        }
        created_event = service.events().insert(calendarId="primary", body=event_body).execute()
        return created_event.get("id", "")
    except Exception as exc:
        raise RuntimeError(f"Google Calendar event creation failed: {exc}") from exc
