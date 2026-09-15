import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.tools import calendar_tool


class FakeCredentials:
    def __init__(self) -> None:
        self.valid = True
        self.expired = False
        self.refresh_token = None

    def __getstate__(self) -> dict:
        return {"valid": self.valid, "expired": self.expired, "refresh_token": self.refresh_token}

    def __setstate__(self, state: dict) -> None:
        self.valid = state.get("valid", True)
        self.expired = state.get("expired", False)
        self.refresh_token = state.get("refresh_token")


class CalendarToolTests(unittest.TestCase):
    def test_authenticate_google_calendar_uses_local_server_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "token.pickle"
            credentials_path = Path(tmpdir) / "calendar_credentials.json"
            credentials_path.write_text("{}")

            class FakeFlow:
                def __init__(self) -> None:
                    self.authorization_url_call_count = 0
                    self.run_local_server_calls = []

                def authorization_url(self, prompt=None) -> tuple:
                    self.authorization_url_call_count += 1
                    return ("https://example.test/auth", "state")

                def run_local_server(self, *args, **kwargs) -> FakeCredentials:
                    self.run_local_server_calls.append((args, kwargs))
                    return FakeCredentials()

            fake_flow = FakeFlow()

            with (
                patch.object(calendar_tool, "TOKEN_PATH", token_path),
                patch.object(calendar_tool, "CREDENTIALS_PATH", credentials_path),
                patch.object(calendar_tool.InstalledAppFlow, "from_client_secrets_file", return_value=fake_flow) as from_client_secrets_file,
                patch.object(calendar_tool, "build", return_value="service") as build_mock,
            ):
                service = calendar_tool.authenticate_google_calendar()

            self.assertEqual(service, "service")
            from_client_secrets_file.assert_called_once()
            self.assertEqual(fake_flow.authorization_url_call_count, 1)
            self.assertEqual(fake_flow.run_local_server_calls[0][1].get("port"), 0)
            self.assertTrue(token_path.exists())
            build_mock.assert_called_once()

    def test_authenticate_google_calendar_surfaces_manual_auth_url_when_local_server_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "token.pickle"
            credentials_path = Path(tmpdir) / "calendar_credentials.json"
            credentials_path.write_text("{}")

            class FakeFlow:
                def authorization_url(self, prompt=None) -> tuple:
                    return ("https://example.test/auth", "state")

                def run_local_server(self, *args, **kwargs) -> FakeCredentials:
                    raise RuntimeError("localhost callback failed")

            with (
                patch.object(calendar_tool, "TOKEN_PATH", token_path),
                patch.object(calendar_tool, "CREDENTIALS_PATH", credentials_path),
                patch.object(calendar_tool.InstalledAppFlow, "from_client_secrets_file", return_value=FakeFlow()),
            ):
                with self.assertRaises(calendar_tool.CalendarAuthenticationError) as context:
                    calendar_tool.authenticate_google_calendar()

            self.assertIn("https://example.test/auth", str(context.exception))
            self.assertIn("manual authorization", str(context.exception).lower())

    def test_create_event_falls_back_to_local_file_when_google_auth_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "calendar_events.json"
            with (
                patch.object(calendar_tool, "FALLBACK_EVENTS_PATH", fallback_path),
                patch.object(calendar_tool, "authenticate_google_calendar", side_effect=FileNotFoundError("missing creds")),
            ):
                event_id = calendar_tool.create_event(
                    "Demo meeting",
                    "2026-07-31T15:00:00+00:00",
                    "2026-07-31T15:30:00+00:00",
                    "Test",
                )

            self.assertTrue(fallback_path.exists())
            payload = json.loads(fallback_path.read_text())
            self.assertEqual(payload[0]["summary"], "Demo meeting")
            self.assertTrue(event_id.startswith("local-"))

    def test_create_event_passes_location_attendees_and_recurrence_to_google_calendar(self) -> None:
        class FakeEvents:
            def __init__(self) -> None:
                self.insert_calls = []

            def insert(self, calendarId: str, body: dict) -> "FakeInsert":
                self.insert_calls.append((calendarId, body))
                return FakeInsert()

        class FakeInsert:
            def execute(self) -> dict:
                return {"id": "evt-123"}

        fake_service = Mock()
        fake_service.events.return_value = FakeEvents()

        with patch.object(calendar_tool, "authenticate_google_calendar", return_value=fake_service):
            event_id = calendar_tool.create_event(
                summary="Weekly sync",
                start_time="2026-07-31T16:00:00+00:00",
                end_time="2026-07-31T16:45:00+00:00",
                description="Roadmap review",
                location="Conference Room A",
                attendees=["will@example.com", "jane@example.com"],
                recurrence=["RRULE:FREQ=WEEKLY;BYDAY=FR"],
            )

        self.assertEqual(event_id, "evt-123")
        calendar_id, body = fake_service.events.return_value.insert_calls[0]
        self.assertEqual(calendar_id, "primary")
        self.assertEqual(body["location"], "Conference Room A")
        self.assertEqual(body["description"], "Roadmap review")
        self.assertEqual(body["attendees"], [{"email": "will@example.com"}, {"email": "jane@example.com"}])
        self.assertEqual(body["recurrence"], ["RRULE:FREQ=WEEKLY;BYDAY=FR"])

    def test_create_event_adds_timezone_to_naive_datetime_strings(self) -> None:
        class FakeEvents:
            def __init__(self) -> None:
                self.insert_calls = []

            def insert(self, calendarId: str, body: dict) -> "FakeInsert":
                self.insert_calls.append((calendarId, body))
                return FakeInsert()

        class FakeInsert:
            def execute(self) -> dict:
                return {"id": "evt-456"}

        fake_service = Mock()
        fake_service.events.return_value = FakeEvents()

        with patch.object(calendar_tool, "authenticate_google_calendar", return_value=fake_service):
            event_id = calendar_tool.create_event(
                summary="Timezone test",
                start_time="2026-07-31T16:00:00",
                end_time="2026-07-31T16:30:00",
            )

        self.assertEqual(event_id, "evt-456")
        calendar_id, body = fake_service.events.return_value.insert_calls[0]
        self.assertEqual(calendar_id, "primary")
        self.assertEqual(body["start"]["timeZone"], "UTC")
        self.assertEqual(body["start"]["dateTime"], "2026-07-31T16:00:00+00:00")
        self.assertEqual(body["end"]["timeZone"], "UTC")
        self.assertEqual(body["end"]["dateTime"], "2026-07-31T16:30:00+00:00")

    def test_delete_event_calls_google_calendar_delete(self) -> None:
        class FakeEvents:
            def __init__(self) -> None:
                self.delete_calls = []

            def delete(self, calendarId: str, eventId: str) -> "FakeDelete":
                self.delete_calls.append((calendarId, eventId))
                return FakeDelete()

        class FakeDelete:
            def execute(self) -> dict:
                return {}

        fake_service = Mock()
        fake_service.events.return_value = FakeEvents()

        with patch.object(calendar_tool, "authenticate_google_calendar", return_value=fake_service):
            result = calendar_tool.delete_event("evt-123")

        self.assertEqual(result, "evt-123")
        self.assertEqual(fake_service.events.return_value.delete_calls, [("primary", "evt-123")])

    def test_delete_event_removes_matching_local_fallback_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "calendar_events.json"
            fallback_path.write_text(
                json.dumps(
                    [
                        {"id": "local-1", "summary": "Keep me"},
                        {"id": "local-2", "summary": "Delete me"},
                    ]
                )
            )

            with (
                patch.object(calendar_tool, "FALLBACK_EVENTS_PATH", fallback_path),
                patch.object(calendar_tool, "authenticate_google_calendar", side_effect=FileNotFoundError("missing creds")),
            ):
                result = calendar_tool.delete_event("local-2")

            self.assertIn("local-2", result)
            remaining = json.loads(fallback_path.read_text())
            self.assertEqual([item["id"] for item in remaining], ["local-1"])

    def test_load_client_config_reads_google_credentials_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "calendar_credentials.json"
            credentials_path.write_text('{"installed": {"client_id": "abc"}}')
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text(f"GOOGLE_OAUTH_CREDENTIALS_FILE={credentials_path}\n")

            def fake_load_dotenv(path, override=False):
                os.environ["GOOGLE_OAUTH_CREDENTIALS_FILE"] = str(credentials_path)

            with (
                patch.object(calendar_tool, "REPO_ROOT", Path(tmpdir)),
                patch.object(calendar_tool, "load_dotenv", side_effect=fake_load_dotenv),
            ):
                config = calendar_tool._load_client_config()

            self.assertEqual(config["installed"]["client_id"], "abc")
