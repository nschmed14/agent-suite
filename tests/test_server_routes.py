import unittest

from fastapi.testclient import TestClient

from backend.server import app


class ServerRouteTests(unittest.TestCase):
    def test_root_serves_frontend_index(self) -> None:
        with TestClient(app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Agent Suite", response.text)

    def test_api_assistant_alias_returns_response_payload(self) -> None:
        with TestClient(app) as client:
            response = client.post("/api/assistant", json={"request": "hello"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("response", payload)
        self.assertIn("model_status", payload)

    def test_frontend_javascript_bundle_is_served(self) -> None:
        with TestClient(app) as client:
            response = client.get("/js/ui_overlay.js")

        self.assertEqual(response.status_code, 200)
        self.assertIn("class UIOverlay", response.text)


if __name__ == "__main__":
    unittest.main()
