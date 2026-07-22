import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.assistant import Assistant
from backend.config import Settings
from backend.memory import LocalMemory
from backend.server import app


class AssistantTests(unittest.TestCase):
    def _build_assistant(self) -> Assistant:
        workspace_root = Path(__file__).resolve().parents[1]
        tempdir = workspace_root / ".tmp" / "assistant-tests"
        tempdir.mkdir(parents=True, exist_ok=True)
        return Assistant(
            memory=LocalMemory(str(tempdir / "agent-suite-test.db"), str(tempdir / "agent-suite-chroma")),
            settings=Settings(),
        )

    def test_plans_calendar_and_email_tasks(self) -> None:
        assistant = self._build_assistant()

        plan = assistant.plan_request("Morning briefing")

        self.assertIn("calendar", plan)
        self.assertIn("email", plan)
        self.assertIn("manager", plan)

    def test_plans_finance_and_research_tasks(self) -> None:
        assistant = self._build_assistant()

        plan = assistant.plan_request("Review my subscriptions and compare travel options")

        self.assertIn("financial", plan)
        self.assertIn("researcher", plan)
        self.assertIn("manager", plan)

    def test_fallback_response_is_manager_friendly(self) -> None:
        assistant = self._build_assistant()

        response = assistant._fallback_response("hello there", [])

        self.assertIn("unable", response.lower())
        self.assertIn("fallback", response.lower())
        self.assertIn("response", response.lower())

    def test_greeting_returns_local_fallback_message(self) -> None:
        assistant = self._build_assistant()

        payload = asyncio.run(assistant.run("hi"))

        self.assertIn("calendar", payload["response"].lower())
        self.assertIn("ollama", payload["response"].lower())
        self.assertEqual(payload["model_status"], "fallback")

    def test_assistant_endpoint_returns_response_payload(self) -> None:
        with TestClient(app) as client:
            response = client.post("/assistant", json={"request": "Give me a brief local status update"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("response", payload)
        self.assertIn("model_status", payload)

    def test_assistant_pulls_missing_model_before_chat(self) -> None:
        assistant = self._build_assistant()

        class FakeClient:
            instances = []

            def __init__(self, *args, **kwargs) -> None:
                self.pull_calls = []
                self.chat_calls = []
                FakeClient.instances.append(self)

            def list(self) -> dict:
                return {"models": []}

            def pull(self, model: str) -> dict:
                self.pull_calls.append(model)
                return {"status": "success"}

            def chat(self, model: str, messages: list) -> dict:
                self.chat_calls.append((model, messages))
                return {"message": {"content": "Hello from the locally pulled model"}}

        class FakeOllamaModule:
            @staticmethod
            def Client(*args, **kwargs) -> FakeClient:
                return FakeClient(*args, **kwargs)

        with patch("backend.assistant._load_ollama_client", return_value=FakeOllamaModule):
            payload = asyncio.run(assistant.run("hello there"))

        self.assertEqual(payload["model_status"], "ollama")
        self.assertIn("locally pulled model", payload["response"].lower())
        self.assertEqual(FakeClient.instances[0].pull_calls, [assistant.settings.ollama_model])


if __name__ == "__main__":
    unittest.main()
