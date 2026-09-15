"""Google Calendar helpers for local OAuth-based calendar access."""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class CalendarAuthenticationError(RuntimeError):
    """Raised when Google Calendar authentication cannot be completed."""

SCOPES = ["https://www.googleapis.com/auth/calendar"]
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent
CREDENTIALS_PATH = BASE_DIR / "credentials" / "calendar_credentials.json"
TOKEN_PATH = BASE_DIR / "credentials" / "token.pickle"
FALLBACK_EVENTS_PATH = BASE_DIR / "credentials" / "calendar_events.json"


def _load_environment() -> None:
    """Load repo-local .env values so Google credentials can be supplied without committing secrets."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(Path(os.getenv("HOME", "")) / ".env", override=False)


def _candidate_credential_paths() -> list[Path]:
    """Return likely local locations for Google OAuth credentials, including safer off-repo paths."""
    candidates: list[Path] = []

    env_path = os.getenv("GOOGLE_OAUTH_CREDENTIALS_FILE")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for raw_path in [
        os.getenv("HOME") + "/.config/agent-suite/calendar_credentials.json",
        os.getenv("HOME") + "/.local/share/agent-suite/calendar_credentials.json",
        str(REPO_ROOT / ".secrets" / "calendar_credentials.json"),
        str(REPO_ROOT / "backend" / "credentials" / "calendar_credentials.json"),
        str(CREDENTIALS_PATH),
    ]:
        if raw_path:
            candidates.append(Path(raw_path).expanduser())

    # Avoid duplicates while preserving order.
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(candidate)
    return ordered


def _load_client_config() -> dict[str, Any]:
    """Load Google OAuth client configuration from the environment or a local file."""
    _load_environment()
    env_json = os.getenv("GOOGLE_OAUTH_CREDENTIALS_JSON")
    if env_json:
        return json.loads(env_json)

    for credentials_path in _candidate_credential_paths():
        if not credentials_path.exists():
            continue
        try:
            return json.loads(credentials_path.read_text())
        except json.JSONDecodeError as exc:
            raise FileNotFoundError(
                f"Google Calendar credentials file at {credentials_path} is not valid JSON: {exc}"
            ) from exc

    raise FileNotFoundError(
        "Google Calendar credentials were not provided. Set GOOGLE_OAUTH_CREDENTIALS_JSON or GOOGLE_OAUTH_CREDENTIALS_FILE, or place the file at one of the local fallback locations."
    )


def _serialize_datetime(value: datetime | str) -> str:
    """Convert a datetime-like value to an RFC3339 string for Google Calendar."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.isoformat()
    raise TypeError("start_time and end_time must be datetime objects or ISO strings")


def _get_google_timezone(value: datetime | str) -> str:
    """Infer a Google Calendar-compatible time zone name for the input value."""
    if isinstance(value, datetime):
        tzinfo = value.tzinfo
        if tzinfo is None:
            return "UTC"
        if hasattr(tzinfo, "key"):
            return str(tzinfo.key)
        tz_name = tzinfo.tzname(value)
        if tz_name:
            return tz_name
        return "UTC"
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return "UTC"
        if parsed.tzinfo is None:
            return "UTC"
        tzinfo = parsed.tzinfo
        if hasattr(tzinfo, "key"):
            return str(tzinfo.key)
        tz_name = tzinfo.tzname(parsed)
        if tz_name:
            return tz_name
        return "UTC"
    return "UTC"


def _build_auth_flow(client_config: dict[str, Any]) -> Any:
    """Create an OAuth flow using either the in-memory client config or a local secrets file."""
    credentials_file = None
    env_path = os.getenv("GOOGLE_OAUTH_CREDENTIALS_FILE")
    if env_path:
        credentials_file = Path(env_path).expanduser()
    elif CREDENTIALS_PATH.exists():
        credentials_file = CREDENTIALS_PATH

    if credentials_file is not None and credentials_file.exists():
        try:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
        except Exception as fallback_exc:
            if client_config:
                try:
                    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                except Exception as exc:
                    raise CalendarAuthenticationError(
                        f"Google Calendar authentication failed with both client config and secrets file: {exc}; fallback: {fallback_exc}"
                    ) from fallback_exc
            else:
                raise CalendarAuthenticationError(
                    f"Google Calendar authentication failed with the secrets file: {fallback_exc}"
                ) from fallback_exc
    else:
        try:
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        except Exception as exc:
            raise CalendarAuthenticationError(f"Google Calendar authentication failed: {exc}") from exc

    return flow


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
                flow = _build_auth_flow(client_config)
                print("Starting Google Calendar authorization flow...")
                auth_url, _ = flow.authorization_url(prompt="consent")
                print("Please visit this URL to authorize access:")
                print(auth_url)
                try:
                    creds = flow.run_local_server(port=0, open_browser=False)
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
    try:
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
                "id": event.get("id"),
                "summary": event.get("summary", "(No title)"),
                "start": event.get("start", {}).get("dateTime") or event.get("start", {}).get("date"),
                "end": event.get("end", {}).get("dateTime") or event.get("end", {}).get("date"),
            }
            for event in events
        ]
    except Exception:
        if FALLBACK_EVENTS_PATH.exists():
            try:
                payload = json.loads(FALLBACK_EVENTS_PATH.read_text())
                return [
                    {
                        "id": item.get("id"),
                        "summary": item.get("summary", "(No title)"),
                        "start": item.get("start"),
                        "end": item.get("end"),
                    }
                    for item in payload
                ]
            except Exception:
                pass
        return []


def _append_fallback_event(
    summary: str,
    start_time: datetime | str,
    end_time: datetime | str,
    description: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
    recurrence: list[str] | None = None,
) -> str:
    """Persist a local fallback event when Google Calendar integration is unavailable."""
    FALLBACK_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    if FALLBACK_EVENTS_PATH.exists():
        try:
            events = json.loads(FALLBACK_EVENTS_PATH.read_text())
        except Exception:
            events = []

    event_id = f"local-{len(events) + 1}"
    event_payload = {
        "id": event_id,
        "summary": summary,
        "description": description or "",
        "location": location or "",
        "attendees": attendees or [],
        "recurrence": recurrence or [],
        "start": _serialize_datetime(start_time),
        "end": _serialize_datetime(end_time),
    }
    events.append(event_payload)
    FALLBACK_EVENTS_PATH.write_text(json.dumps(events, indent=2))
    return event_id


def _update_fallback_event(
    event_id: str,
    summary: str,
    start_time: datetime | str,
    end_time: datetime | str,
    description: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
    recurrence: list[str] | None = None,
) -> str:
    """Update a persisted fallback event by matching its id."""
    FALLBACK_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    if FALLBACK_EVENTS_PATH.exists():
        try:
            events = json.loads(FALLBACK_EVENTS_PATH.read_text())
        except Exception:
            events = []

    matched = False
    for item in events:
        if item.get("id") != event_id:
            continue
        matched = True
        item["summary"] = summary
        item["description"] = description if description is not None else item.get("description", "")
        item["location"] = location if location is not None else item.get("location", "")
        item["attendees"] = attendees if attendees is not None else item.get("attendees", [])
        item["recurrence"] = recurrence if recurrence is not None else item.get("recurrence", [])
        item["start"] = _serialize_datetime(start_time)
        item["end"] = _serialize_datetime(end_time)
        FALLBACK_EVENTS_PATH.write_text(json.dumps(events, indent=2))
        return event_id

    if matched:
        FALLBACK_EVENTS_PATH.write_text(json.dumps(events, indent=2))

    return _append_fallback_event(
        summary,
        start_time,
        end_time,
        description=description,
        location=location,
        attendees=attendees,
        recurrence=recurrence,
    )


def create_event(
    summary: str,
    start_time: datetime | str,
    end_time: datetime | str,
    description: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
    recurrence: list[str] | None = None,
) -> str:
    """Create a new event in the user's primary calendar and return its ID."""
    try:
        service = authenticate_google_calendar()
        start_dt = _serialize_datetime(start_time)
        end_dt = _serialize_datetime(end_time)
        event_body = {
            "summary": summary,
            "description": description or "",
            "start": {"dateTime": start_dt, "timeZone": _get_google_timezone(start_time)},
            "end": {"dateTime": end_dt, "timeZone": _get_google_timezone(end_time)},
        }
        if location:
            event_body["location"] = location
        if attendees:
            event_body["attendees"] = [{"email": attendee} for attendee in attendees]
        if recurrence:
            event_body["recurrence"] = recurrence
        created_event = service.events().insert(calendarId="primary", body=event_body).execute()
        return created_event.get("id", "")
    except Exception as exc:
        fallback_id = _append_fallback_event(
            summary,
            start_time,
            end_time,
            description=description,
            location=location,
            attendees=attendees,
            recurrence=recurrence,
        )
        return f"{fallback_id} (local fallback; original error: {exc})"


def _delete_fallback_event(event_id: str) -> bool:
    """Remove a persisted fallback event by id. Returns True if a matching event was found and removed."""
    if not FALLBACK_EVENTS_PATH.exists():
        return False
    try:
        events = json.loads(FALLBACK_EVENTS_PATH.read_text())
    except Exception:
        return False

    remaining = [item for item in events if item.get("id") != event_id]
    if len(remaining) == len(events):
        return False

    FALLBACK_EVENTS_PATH.write_text(json.dumps(remaining, indent=2))
    return True


def delete_event(event_id: str) -> str:
    """Delete an event from the user's primary calendar and return a status string."""
    try:
        service = authenticate_google_calendar()
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return event_id
    except Exception as exc:
        if _delete_fallback_event(event_id):
            return f"{event_id} (removed local fallback entry; original error: {exc})"
        return f"{event_id} (local fallback lookup failed; original error: {exc})"


def update_event(
    event_id: str,
    summary: str,
    start_time: datetime | str,
    end_time: datetime | str,
    description: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
    recurrence: list[str] | None = None,
) -> str:
    """Update an existing event in the user's primary calendar and return its ID."""
    try:
        service = authenticate_google_calendar()
        start_dt = _serialize_datetime(start_time)
        end_dt = _serialize_datetime(end_time)
        event_body = {
            "summary": summary,
            "description": description or "",
            "start": {"dateTime": start_dt, "timeZone": _get_google_timezone(start_time)},
            "end": {"dateTime": end_dt, "timeZone": _get_google_timezone(end_time)},
        }
        if location:
            event_body["location"] = location
        if attendees:
            event_body["attendees"] = [{"email": attendee} for attendee in attendees]
        if recurrence:
            event_body["recurrence"] = recurrence
        updated_event = service.events().update(calendarId="primary", eventId=event_id, body=event_body).execute()
        return updated_event.get("id", event_id)
    except Exception as exc:
        fallback_id = _update_fallback_event(
            event_id,
            summary,
            start_time,
            end_time,
            description=description,
            location=location,
            attendees=attendees,
            recurrence=recurrence,
        )
        return f"{fallback_id} (local fallback; original error: {exc})"
