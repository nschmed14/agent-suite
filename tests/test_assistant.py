import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

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

    def test_prefers_small_chat_model_when_available(self) -> None:
        assistant = self._build_assistant()

        class FakeClient:
            instances = []

            def __init__(self, *args, **kwargs) -> None:
                self.pull_calls = []
                self.chat_calls = []
                FakeClient.instances.append(self)

            def list(self) -> dict:
                return {"models": [{"name": "phi3.5:3.5-mini-instruct-q4_K_M"}, {"name": "llama3.1:8b"}]}

            def pull(self, model: str) -> dict:
                self.pull_calls.append(model)
                return {"status": "success"}

            def chat(self, model: str, messages: list) -> dict:
                self.chat_calls.append((model, messages))
                return {"message": {"content": "Hello from the fast chat model"}}

        class FakeOllamaModule:
            @staticmethod
            def Client(*args, **kwargs) -> FakeClient:
                return FakeClient(*args, **kwargs)

        with patch("backend.assistant._load_ollama_client", return_value=FakeOllamaModule):
            payload = asyncio.run(assistant.run("hello there"))

        self.assertEqual(payload["model_status"], "ollama")
        self.assertIn("fast chat model", payload["response"].lower())
        self.assertEqual(FakeClient.instances[0].chat_calls[0][0], "phi3.5:3.5-mini-instruct-q4_K_M")

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

    def test_uses_available_model_name_from_model_objects(self) -> None:
        assistant = self._build_assistant()

        class FakeModel:
            def __init__(self, name: str) -> None:
                self.name = name

            def __str__(self) -> str:
                return self.name

        class FakeClient:
            instances = []

            def __init__(self, *args, **kwargs) -> None:
                self.pull_calls = []
                self.chat_calls = []
                FakeClient.instances.append(self)

            def list(self) -> dict:
                return {"models": [FakeModel("llama3.1:8b")]}

            def pull(self, model: str) -> dict:
                self.pull_calls.append(model)
                return {"status": "success"}

            def chat(self, model: str, messages: list) -> dict:
                self.chat_calls.append((model, messages))
                return {"message": {"content": "Hello from the discovered model"}}

        class FakeOllamaModule:
            @staticmethod
            def Client(*args, **kwargs) -> FakeClient:
                return FakeClient(*args, **kwargs)

        with patch("backend.assistant._load_ollama_client", return_value=FakeOllamaModule):
            payload = asyncio.run(assistant.run("hello there"))

        self.assertEqual(payload["model_status"], "ollama")
        self.assertIn("discovered model", payload["response"].lower())
        self.assertEqual(FakeClient.instances[0].chat_calls[0][0], "llama3.1:8b")

    def test_deterministic_routing_handles_greetings_and_specialists(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None):
            greeting_route = asyncio.run(assistant._route_request("hello"))
            schedule_route = asyncio.run(assistant._route_request("schedule a meeting"))
            finance_route = asyncio.run(assistant._route_request("check my finances"))

        self.assertEqual(greeting_route["agent"], "manager")
        self.assertEqual(schedule_route["agent"], "scheduler")
        self.assertEqual(finance_route["agent"], "finance")

    def test_route_request_sends_create_calendar_requests_to_scheduler(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None):
            route = asyncio.run(assistant._route_request("Create a team sync"))

        self.assertEqual(route["agent"], "scheduler")
        self.assertIn("team sync", route["task"].lower())

    def test_route_request_does_not_treat_this_friday_as_greeting(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None):
            route = asyncio.run(assistant._route_request("Schedule a meeting with John for this friday at 3:45 pm for 1 hour"))

        self.assertEqual(route["agent"], "scheduler")
        self.assertIn("john", route["task"].lower())

    def test_create_calendar_event_parses_llm_schedule_payload(self) -> None:
        assistant = self._build_assistant()

        class FakeClient:
            def list(self) -> dict:
                return {"models": [{"name": "llama3.2:3b"}]}

            def chat(self, model: str, messages: list) -> dict:
                return {"message": {"content": '{"title": "Team sync", "date": "2026-07-30", "start_time": "14:30", "duration_minutes": 45}'}}

        class FakeOllamaModule:
            @staticmethod
            def Client(*args, **kwargs) -> FakeClient:
                return FakeClient()

        with patch("backend.assistant._load_ollama_client", return_value=FakeOllamaModule), patch("backend.assistant.create_event", return_value="evt-123") as create_event_mock:
            result = asyncio.run(assistant._create_calendar_event("Schedule a team sync for tomorrow at 2:30 pm"))

        self.assertIn("Team sync", result)
        self.assertIn("evt-123", result)
        create_event_mock.assert_called_once_with(
            summary="Team sync",
            start_time=unittest.mock.ANY,
            end_time=unittest.mock.ANY,
        )
        called_kwargs = create_event_mock.call_args.kwargs
        start_dt = datetime.fromisoformat(called_kwargs["start_time"])
        end_dt = datetime.fromisoformat(called_kwargs["end_time"])
        self.assertEqual(start_dt.hour, 14)
        self.assertEqual(end_dt.hour, 15)
        self.assertIsNotNone(start_dt.tzinfo)

    def test_create_calendar_event_parses_tomorrow_time_without_llm(self) -> None:
        assistant = self._build_assistant()
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", return_value="evt-456") as create_event_mock:
            result = asyncio.run(assistant._create_calendar_event("schedule a meeting with john for tomorrow at 2pm"))

        self.assertIn("Meeting with John", result)
        self.assertIn("evt-456", result)
        create_event_mock.assert_called_once_with(
            summary="Meeting with John",
            start_time=unittest.mock.ANY,
            end_time=unittest.mock.ANY,
        )
        called_kwargs = create_event_mock.call_args.kwargs
        start_dt = datetime.fromisoformat(called_kwargs["start_time"])
        end_dt = datetime.fromisoformat(called_kwargs["end_time"])
        self.assertEqual(start_dt.day, (datetime.now() + timedelta(days=1)).day)
        self.assertEqual(start_dt.hour, 14)
        self.assertEqual(end_dt.hour, 15)
        self.assertIsNotNone(start_dt.tzinfo)

    def test_create_calendar_event_parses_noon_without_treating_duration_as_time(self) -> None:
        assistant = self._build_assistant()
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", return_value="evt-789") as create_event_mock:
            result = asyncio.run(assistant._create_calendar_event("schedule a meeting with Stacy for tomorrow at noon for 30 minutes"))

        self.assertIn("Meeting with Stacy", result)
        self.assertIn("evt-789", result)
        create_event_mock.assert_called_once_with(
            summary="Meeting with Stacy",
            start_time=unittest.mock.ANY,
            end_time=unittest.mock.ANY,
        )
        called_kwargs = create_event_mock.call_args.kwargs
        start_dt = datetime.fromisoformat(called_kwargs["start_time"])
        end_dt = datetime.fromisoformat(called_kwargs["end_time"])
        self.assertEqual(start_dt.day, (datetime.now() + timedelta(days=1)).day)
        self.assertEqual(start_dt.hour, 12)
        self.assertEqual(start_dt.minute, 0)
        self.assertEqual(end_dt.minute, 30)
        self.assertEqual(end_dt.hour, 12)
        self.assertIsNotNone(start_dt.tzinfo)

    def test_create_calendar_event_parses_multiple_back_to_back_meetings(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", side_effect=["evt-aaa", "evt-bbb"]) as create_event_mock:
            result = asyncio.run(assistant._create_calendar_event("schedule meetings with both stacy and carol tomorrow, both half an hour, stacy at 4 pm and carol right afterwards"))

        self.assertIn("Stacy", result)
        self.assertIn("Carol", result)
        self.assertEqual(create_event_mock.call_count, 2)

        first_kwargs = create_event_mock.call_args_list[0].kwargs
        second_kwargs = create_event_mock.call_args_list[1].kwargs
        self.assertEqual(first_kwargs["summary"], "Meeting with Stacy")
        self.assertEqual(second_kwargs["summary"], "Meeting with Carol")

        first_start = datetime.fromisoformat(first_kwargs["start_time"])
        first_end = datetime.fromisoformat(first_kwargs["end_time"])
        second_start = datetime.fromisoformat(second_kwargs["start_time"])
        second_end = datetime.fromisoformat(second_kwargs["end_time"])
        self.assertEqual(first_start.hour, 16)
        self.assertEqual(first_end.hour, 16)
        self.assertEqual(first_end.minute, 30)
        self.assertEqual(second_start.hour, 16)
        self.assertEqual(second_start.minute, 30)
        self.assertEqual(second_end.hour, 17)
        self.assertEqual(second_end.minute, 0)

    def test_create_calendar_event_handles_sequential_pair_meetings(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", side_effect=["evt-1", "evt-2", "evt-3", "evt-4"]) as create_event_mock:
            result = asyncio.run(
                assistant._create_calendar_event(
                    "first schedule a meeting with john for tomorrow at 4pm for 1 hour, then a meeting with hannah for 7 for 30 minutes, next I have meetings with both stacy and carol to add to the schedule, both of those meetings should be 30 minutes, happening right after the other at 6pm with stacy first"
                )
            )

        self.assertIn("John", result)
        self.assertIn("Hannah", result)
        self.assertIn("Stacy", result)
        self.assertIn("Carol", result)
        self.assertEqual(create_event_mock.call_count, 4)

    def test_scheduler_stub_routes_create_requests_to_calendar_helper(self) -> None:
        assistant = self._build_assistant()

        create_event_mock = AsyncMock(return_value="created-event-id")
        get_events_mock = AsyncMock(return_value=[{"summary": "Existing meeting"}])

        with patch.object(assistant, "_create_calendar_event", create_event_mock), patch.object(assistant, "_get_calendar_events", get_events_mock):
            result = asyncio.run(assistant._scheduler_stub("Create a team sync"))

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["message"], "created-event-id")
        create_event_mock.assert_awaited_once_with("Create a team sync")
        get_events_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
