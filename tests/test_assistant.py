import asyncio
import os
import tempfile
import unittest

from backend.assistant import Assistant
from backend.config import Settings
from backend.memory import LocalMemory


class AssistantTests(unittest.TestCase):
    def _build_assistant(self) -> Assistant:
        tempdir = tempfile.mkdtemp(prefix="agent-suite-test-", dir="/tmp")
        return Assistant(
            memory=LocalMemory(os.path.join(tempdir, "agent-suite-test.db"), os.path.join(tempdir, "agent-suite-chroma")),
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


if __name__ == "__main__":
    unittest.main()
