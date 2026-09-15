import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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

    def test_route_request_sends_availability_follow_up_to_scheduler(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None):
            route = asyncio.run(assistant._route_request("It should be 1 hour, based on my schedule do you have a suggested time?"))

        self.assertEqual(route["agent"], "scheduler")
        self.assertIn("suggested time", route["task"].lower())

    def test_route_request_sends_reschedule_requests_to_scheduler(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "schedule a meeting for tomorrow"},
            {"role": "assistant", "content": "I scheduled 'Meeting' for 2026-07-31 at 09:00 for 30 minutes. Event ID: abc123"},
        ]

        with patch("backend.assistant._load_ollama_client", return_value=None):
            route = asyncio.run(assistant._route_request("reschedule that for 1pm"))

        self.assertEqual(route["agent"], "scheduler")
        self.assertIn("reschedule", route["task"].lower())

    def test_new_scheduling_request_is_not_treated_as_follow_up_to_reschedule(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "reschedule that meeting for noon"},
            {"role": "assistant", "content": "I rescheduled your meeting."},
        ]

        self.assertFalse(assistant._looks_like_follow_up_to_scheduling_request("schedule a meeting for tomorrow", "reschedule that meeting for noon"))

    def test_correction_phrase_is_treated_as_reschedule_request(self) -> None:
        assistant = self._build_assistant()

        self.assertTrue(assistant._looks_like_reschedule_request("Nope, that is friday, I said saturday"))

    def test_first_turn_schedule_request_requests_missing_details(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = []

        self.assertTrue(assistant._needs_clarification("schedule a meeting for tomorrow"))

    def test_first_turn_schedule_request_does_not_inherit_prior_context(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "schedule a meeting with Rosa tomorrow at 1pm for 30 minutes"},
            {"role": "assistant", "content": "I scheduled 'Meeting with Rosa' for 2026-07-31 at 13:00 for 30 minutes. Event ID: abc123"},
        ]

        self.assertTrue(assistant._needs_clarification("schedule a meeting for tomorrow"))
        self.assertEqual(assistant._build_contextual_request("schedule a meeting for tomorrow"), "schedule a meeting for tomorrow")

    def test_follow_up_with_shorthand_duration_uses_prior_context(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "schedule a meeting for tomorrow"},
            {"role": "assistant", "content": "I can help with that. I need to know who the meeting is with or what it is about before I schedule it."},
            {"role": "user", "content": "it's with rosa, 1pm"},
        ]

        self.assertFalse(assistant._needs_clarification("for 30 min"))
        self.assertEqual(assistant._extract_duration_minutes("for 30 min"), 30)

    def test_clarification_uses_conversation_context_for_missing_details(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "schedule a meeting for tomorrow"},
            {"role": "assistant", "content": "I can help with that. I need to know who the meeting is with or what it is about before I schedule it."},
        ]

        response = assistant._build_clarification_response("it's with rosa")

        self.assertIn("time", response.lower())
        self.assertIn("tomorrow", response.lower())

    def test_follow_up_with_participant_and_time_keeps_context_for_duration(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "schedule a meeting for tomorrow"},
            {"role": "assistant", "content": "I can help with that. I need to know who the meeting is with or what it is about before I schedule it."},
            {"role": "user", "content": "it's with rosa, 1pm"},
        ]

        contextual_request = assistant._build_contextual_request("30 min")

        self.assertIn("rosa", contextual_request.lower())
        self.assertIn("tomorrow", contextual_request.lower())
        self.assertIn("13:00", contextual_request)
        self.assertIn("30", contextual_request)

    def test_explicit_new_schedule_request_is_not_treated_as_follow_up(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "schedule a meeting tomorrow with john"},
            {"role": "assistant", "content": "I can help with that. I have John in mind, but I still need the meeting time and how long it should last."},
        ]

        self.assertFalse(assistant._looks_like_follow_up_to_scheduling_request("schedule a meeting for tomorrow", "schedule a meeting tomorrow with john"))

    def test_build_contextual_request_uses_previous_schedule_context(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "Schedule a meeting with tom tomorrow"},
            {"role": "assistant", "content": "I can help with that. I have Tom in mind, but I still need the meeting time and how long it should last."},
        ]

        contextual_request = assistant._build_contextual_request("the meeting should be 1 hour, do you have a time to suggest based on my schedule?")

        self.assertIn("tom", contextual_request.lower())
        self.assertIn("tomorrow", contextual_request.lower())
        self.assertIn("60", contextual_request)

    def test_build_contextual_request_combines_short_follow_up_with_previous_schedule_context(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "Schedule a meeting tomorrow with William for 1 hour"},
            {"role": "assistant", "content": "I can help with that. I have William in mind, but I still need the meeting time and how long it should last."},
        ]

        contextual_request = assistant._build_contextual_request("let's do noon")

        self.assertIn("william", contextual_request.lower())
        self.assertIn("tomorrow", contextual_request.lower())
        self.assertIn("noon", contextual_request.lower())
        self.assertIn("60", contextual_request)

    def test_parse_single_calendar_instruction_uses_prior_participant_for_title(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "Schedule a meeting tomorrow with William for 1 hour"},
            {"role": "assistant", "content": "I can help with that. I have William in mind, but I still need the meeting time and how long it should last."},
        ]

        parsed_events = assistant._parse_single_calendar_instruction(
            "let's do noon",
            now=datetime.now(),
            local_tz=timezone.utc,
            default_date=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            default_start_time="09:00",
            default_duration_minutes=60,
        )

        self.assertEqual(parsed_events[0]["summary"], "Meeting with William")

    def test_run_uses_prior_schedule_context_for_follow_up_requests(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "schedule a meeting tomorrow with will"},
            {"role": "assistant", "content": "I can help with that. I have Will in mind, but I still need the meeting time and how long it should last."},
        ]

        async def fake_route_request(request: str) -> dict:
            self.assertIn("schedule a meeting tomorrow with will", request.lower())
            self.assertIn("one hour", request.lower())
            return {"agent": "scheduler", "task": request}

        with patch.object(assistant, "_route_request", side_effect=fake_route_request):
            with patch.object(assistant, "_execute_agent", new=AsyncMock(return_value={"status": "done", "message": "ok"})):
                with patch.object(assistant, "_generate_final_response", new=AsyncMock(return_value="ok")):
                    asyncio.run(assistant.run("it should last one hour and be at noon"))

    def test_build_contextual_request_preserves_reschedule_intent(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "schedule a meeting with Milo tomorrow at noon for 30 minutes"},
            {"role": "assistant", "content": "✅ I scheduled 'Meeting with Milo' for 2026-07-31 at 12:00 for 30 minutes. Event ID: google-event-1"},
        ]

        contextual_request = assistant._build_contextual_request("reschedule that to be this saturday at noon")

        self.assertIn("reschedule", contextual_request.lower())
        self.assertIn("milo", contextual_request.lower())

    def test_create_calendar_event_reschedules_latest_meeting_from_history(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "Schedule a meeting tomorrow with Riley for 1 hour"},
            {"role": "assistant", "content": "✅ I scheduled 'Meeting with Riley' for 2026-07-31 at 12:00 for 60 minutes. Event ID: google-event-1"},
        ]

        with patch("backend.assistant.update_event", return_value="google-event-1") as update_event_mock:
            result = asyncio.run(assistant._create_calendar_event("actually reschedule that to 1:00 pm"))

        self.assertIn("reschedul", result.lower())
        update_event_mock.assert_called_once()

    def test_create_calendar_event_reschedules_latest_meeting_from_history_even_when_latest_event_is_local_fallback(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "Schedule a meeting tomorrow with Riley for 1 hour"},
            {"role": "assistant", "content": "⚠️ I prepared a local backup entry for 'Meeting with Riley' for 2026-07-31 at 12:00 for 60 minutes. Backup ID: local-7 (local fallback; original error: test)"},
        ]

        with patch("backend.assistant.update_event", return_value="local-7") as update_event_mock:
            result = asyncio.run(assistant._create_calendar_event("reschedule that meeting for noon"))

        self.assertIn("reschedul", result.lower())
        update_event_mock.assert_called_once()
        self.assertEqual(update_event_mock.call_args.kwargs["event_id"], "local-7")

    def test_extract_last_event_context_prefers_real_google_event_id_over_local_fallback(self) -> None:
        assistant = self._build_assistant()
        assistant._conversation_history = [
            {"role": "user", "content": "Schedule a meeting tomorrow with Riley for 1 hour"},
            {"role": "assistant", "content": "⚠️ I prepared a local backup entry for 'Meeting with Riley' for 2026-07-31 at 12:00 for 60 minutes. Backup ID: local-32 (local fallback; original error: test)"},
            {"role": "assistant", "content": "✅ I scheduled 'Meeting with Riley' for 2026-07-31 at 12:00 for 60 minutes. Event ID: real-event-123"},
        ]

        context = assistant._extract_last_event_context()

        self.assertEqual(context["event_id"], "real-event-123")

    def test_suggest_available_time_avoids_busy_slots(self) -> None:
        assistant = self._build_assistant()
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        fake_events = [{"summary": "Existing event", "start": f"{tomorrow}T09:00:00", "end": f"{tomorrow}T10:00:00"}]

        with patch("backend.assistant.get_events", return_value=fake_events):
            suggestion = asyncio.run(assistant._suggest_available_time("suggest an available time for a meeting with tom tomorrow for 60 minutes based on my calendar"))

        self.assertIn("10:00", suggestion)

    def test_route_request_does_not_treat_this_friday_as_greeting(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None):
            route = asyncio.run(assistant._route_request("Schedule a meeting with John for this friday at 3:45 pm for 1 hour"))

        self.assertEqual(route["agent"], "scheduler")
        self.assertIn("john", route["task"].lower())

    def test_scheduler_treats_schedule_lookup_as_calendar_read(self) -> None:
        assistant = self._build_assistant()

        with patch.object(assistant, "_get_calendar_events", new=AsyncMock(return_value=[{"summary": "Existing event"}])) as get_events_mock, \
             patch.object(assistant, "_create_calendar_event", new=AsyncMock(return_value="should not run")) as create_event_mock:
            result = asyncio.run(assistant._scheduler_stub("what do I have on the schedule for tomorrow?"))

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["message"], [{"summary": "Existing event"}])
        get_events_mock.assert_awaited_once_with("what do I have on the schedule for tomorrow?")
        create_event_mock.assert_not_called()

    def test_clean_response_text_formats_lists_of_events(self) -> None:
        assistant = self._build_assistant()

        cleaned = assistant._clean_response_text([{"summary": "Team sync", "start": "2026-07-31T09:00:00Z"}])

        self.assertIn("Team sync", cleaned)
        self.assertIn("2026-07-31", cleaned)

    def test_create_calendar_event_requests_duration_when_missing(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event") as create_event_mock:
            result = asyncio.run(assistant._create_calendar_event("schedule a meeting with john tomorrow"))

        self.assertIn("how long", result.lower())
        self.assertIn("john", result.lower())
        create_event_mock.assert_not_called()

    def test_get_calendar_events_filters_to_requested_day(self) -> None:
        assistant = self._build_assistant()
        today = datetime.now()
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
        saturday_str = saturday.strftime("%Y-%m-%d")
        other_str = (saturday + timedelta(days=1)).strftime("%Y-%m-%d")

        fake_events = [
            {"summary": "Saturday meetup", "start": f"{saturday_str}T10:00:00Z"},
            {"summary": "Sunday meetup", "start": f"{other_str}T10:00:00Z"},
        ]

        with patch("backend.assistant.get_events", return_value=fake_events):
            result = asyncio.run(assistant._get_calendar_events("what do I have on my schedule for this saturday?"))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["summary"], "Saturday meetup")

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

    def test_create_calendar_event_parses_explicit_weekday_date_without_defaults(self) -> None:
        assistant = self._build_assistant()
        today = datetime.now()
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
        saturday_str = saturday.strftime("%Y-%m-%d")

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", return_value="evt-456") as create_event_mock:
            result = asyncio.run(assistant._create_calendar_event("schedule a meeting with Milo for saturday at noon for 30 minutes"))

        self.assertIn("Meeting with Milo", result)
        self.assertIn("evt-456", result)
        create_event_mock.assert_called_once_with(
            summary="Meeting with Milo",
            start_time=unittest.mock.ANY,
            end_time=unittest.mock.ANY,
        )
        called_kwargs = create_event_mock.call_args.kwargs
        start_dt = datetime.fromisoformat(called_kwargs["start_time"])
        self.assertEqual(start_dt.strftime("%Y-%m-%d"), saturday_str)

    def test_create_calendar_event_parses_tomorrow_time_without_llm(self) -> None:
        assistant = self._build_assistant()
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", return_value="evt-456") as create_event_mock:
            result = asyncio.run(assistant._create_calendar_event("schedule a meeting with john for tomorrow at 2pm"))

        self.assertIn("how long", result.lower())
        self.assertIn("john", result.lower())
        create_event_mock.assert_not_called()

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

    def test_create_calendar_event_parses_recurring_details_and_guests(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", return_value="evt-recurring") as create_event_mock:
            result = asyncio.run(
                assistant._create_calendar_event(
                    "schedule a weekly recurring meeting with will for fridays at 4pm for 45 minutes starting tomorrow at the conference room, invite jane@example.com, description: roadmap review"
                )
            )

        self.assertIn("Weekly", result)
        self.assertIn("evt-recurring", result)
        create_event_mock.assert_called_once()
        kwargs = create_event_mock.call_args.kwargs
        self.assertEqual(kwargs["summary"], "Meeting with Will")
        self.assertEqual(kwargs["location"], "conference room")
        self.assertEqual(kwargs["attendees"], ["jane@example.com"])
        self.assertEqual(kwargs["description"], "roadmap review")
        self.assertEqual(kwargs["recurrence"], ["RRULE:FREQ=WEEKLY;BYDAY=FR"])

    def test_create_calendar_event_parses_weekly_recurring_without_weekday(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", return_value="evt-weekly") as create_event_mock:
            result = asyncio.run(
                assistant._create_calendar_event(
                    "schedule a weekly recurring meeting with will starting tomorrow at 4pm for 30 minutes"
                )
            )

        self.assertIn("Will", result)
        self.assertIn("evt-weekly", result)
        create_event_mock.assert_called_once()
        kwargs = create_event_mock.call_args.kwargs
        self.assertEqual(kwargs["recurrence"], ["RRULE:FREQ=WEEKLY"])

    def test_create_calendar_event_parses_address_phrase_as_location(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", return_value="evt-address") as create_event_mock:
            result = asyncio.run(
                assistant._create_calendar_event(
                    "schedule a weekly recurring meeting with will starting tomorrow at 4pm for 30 minutes, set the address to the empire state building"
                )
            )

        self.assertIn("Will", result)
        self.assertIn("evt-address", result)
        create_event_mock.assert_called_once()
        kwargs = create_event_mock.call_args.kwargs
        self.assertEqual(kwargs["location"], "Empire State Building")

    def test_create_calendar_event_mentions_location_in_response(self) -> None:
        assistant = self._build_assistant()

        with patch("backend.assistant._load_ollama_client", return_value=None), patch("backend.assistant.create_event", return_value="local-1 (local fallback; original error: test)") as create_event_mock:
            result = asyncio.run(
                assistant._create_calendar_event(
                    "schedule a weekly recurring meeting with will starting tomorrow at 4pm for 30 minutes, set the address to the empire state building"
                )
            )

        self.assertIn("Empire State Building", result)
        self.assertIn("local backup entry", result.lower())
        create_event_mock.assert_called_once()

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

    def test_scheduler_stub_routes_reschedule_requests_to_calendar_helper(self) -> None:
        assistant = self._build_assistant()

        create_event_mock = AsyncMock(return_value="rescheduled-event-id")
        get_events_mock = AsyncMock(return_value=[{"summary": "Existing meeting"}])

        with patch.object(assistant, "_create_calendar_event", create_event_mock), patch.object(assistant, "_get_calendar_events", get_events_mock):
            result = asyncio.run(assistant._scheduler_stub("reschedule that for 1pm"))

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["message"], "rescheduled-event-id")
        create_event_mock.assert_awaited_once_with("reschedule that for 1pm")
        get_events_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
