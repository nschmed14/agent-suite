import unittest

from backend.assistant import Assistant
from backend.config import Settings
from backend.memory import LocalMemory


class AssistantTests(unittest.TestCase):
    def test_plans_calendar_and_email_tasks(self) -> None:
        assistant = Assistant(
            memory=LocalMemory("/tmp/agent-suite-test.db", "/tmp/agent-suite-chroma"),
            settings=Settings(),
        )

        plan = assistant.plan_request("Morning briefing")

        self.assertIn("calendar", plan)
        self.assertIn("email", plan)
        self.assertIn("manager", plan)

    def test_plans_finance_and_research_tasks(self) -> None:
        assistant = Assistant(
            memory=LocalMemory("/tmp/agent-suite-test.db", "/tmp/agent-suite-chroma"),
            settings=Settings(),
        )

        plan = assistant.plan_request("Review my subscriptions and compare travel options")

        self.assertIn("financial", plan)
        self.assertIn("researcher", plan)
        self.assertIn("manager", plan)


if __name__ == "__main__":
    unittest.main()
