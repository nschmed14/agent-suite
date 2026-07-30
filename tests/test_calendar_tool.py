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
