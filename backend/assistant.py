"""Single-brain assistant orchestration for Agent Suite.

The assistant coordinates local-only tools and streams progress updates over a
callback interface so the frontend can show the office characters as a warm,
privacy-first progress indicator.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from backend.config import Settings
from backend.memory import LocalMemory
from backend.tools.calendar_tool import create_event, get_events


def _load_ollama_client():
    """Import Ollama lazily so tests and local falls back work even without a proxy setup."""
    try:
        import ollama  # type: ignore
    except Exception:
        return None

    # Avoid proxy-related import failures when the environment exposes odd proxy settings.
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(proxy_var, None)
    return ollama


class Assistant:
    """A single local AI brain that can reason over a few private tools."""

    def __init__(self, memory: LocalMemory, settings: Settings) -> None:
        self.memory = memory
        self.settings = settings
        self._callbacks: List[Callable[[Dict[str, Any]], Awaitable[None] | None]] = []

    def register_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None] | None]) -> None:
        self._callbacks.append(callback)

    async def run(self, request: str) -> Dict[str, Any]:
        """Route the request to the right agent and execute."""
        await self._emit_status("working", "manager", "Routing your request...", 0.1)

        # Step 1: Route with the LLM
        route = await self._route_request(request)
        agent = route["agent"]
        task = route["task"]

        await self._emit_status("working", agent, f"Processing {agent} task...", 0.3)

        # Step 2: Execute the right agent
        result = await self._execute_agent(agent, task)

        # Step 3: Generate final response
        response = await self._generate_final_response(request, agent, result)
        response = self._clean_response_text(response)

        self.memory.add_memory(
            "assistant",
            f"User: {request} | Routed to: {agent} | Response: {response}",
            metadata={"kind": "assistant"}
        )

        await self._emit_status("idle", "manager", "Ready.", 1.0, final_response=response)
        return {"response": response, "agent": agent, "task": task, "result": result}

    async def _route_request(self, request: str) -> Dict[str, Any]:
        """Route the request to the right agent using local heuristics first."""
        normalized = (request or "").strip()
        if not normalized:
            return {"agent": "manager", "task": request}

        lowered = normalized.lower()

        greeting_terms = [
            "hello",
            "hi",
            "hey",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "how's it going",
            "how do you do",
            "what's up",
        ]
        if lowered in greeting_terms or any(re.fullmatch(rf"\b{re.escape(term)}\b", lowered) for term in greeting_terms):
            return {"agent": "manager", "task": normalized}

        briefing_terms = ["morning briefing", "daily briefing", "daily brief", "briefing"]
        if any(term in lowered for term in briefing_terms):
            return {"agent": "manager", "task": normalized}

        schedule_terms = ["schedule", "meeting", "calendar", "appointment", "book", "reserve", "conference room", "create a", "create an", "create"]
        if any(term in lowered for term in schedule_terms):
            return {"agent": "scheduler", "task": normalized}

        mail_terms = ["email", "mail", "draft", "reply", "inbox"]
        if any(term in lowered for term in mail_terms):
            return {"agent": "mail", "task": normalized}

        finance_terms = ["finance", "finances", "money", "budget", "spending", "subscription", "subscriptions", "rocket money", "check my finances"]
        if any(term in lowered for term in finance_terms):
            return {"agent": "finance", "task": normalized}

        research_terms = ["research", "search", "find", "compare", "look up", "internet", "web", "news"]
        if any(term in lowered for term in research_terms):
            return {"agent": "researcher", "task": normalized}

        ollama = _load_ollama_client()
        if ollama is None:
            return {"agent": "manager", "task": normalized}

        client = ollama.Client(host=self.settings.ollama_base_url)
        model = self._get_fastest_model(client)

        route_prompt = f"""Route this user request to exactly one specialist agent.

Agents:
- Scheduler: Calendar events, scheduling, meetings, time management
- Mail: Email drafts, inbox management, replies (NEVER sends)
- Finance: Rocket Money, budgets, spending, subscriptions
- Researcher: Internet search, research, fact-finding, comparisons
- Manager: General conversation, morning briefing, coordination

User request: "{request}"

Respond ONLY with this format:
AGENT: [Scheduler|Mail|Finance|Researcher|Manager]
TASK: [One sentence description of what needs to be done]

If unclear, default to Manager with the task as-is.
"""

        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": route_prompt}],
        )

        content = response.get("message", {}).get("content", "")

        # Parse the response
        agent_match = re.search(r"AGENT:\s*(\w+)", content, re.IGNORECASE)
        task_match = re.search(r"TASK:\s*(.+)", content, re.IGNORECASE)

        agent = agent_match.group(1).lower() if agent_match else "manager"
        task = task_match.group(1).strip() if task_match else normalized

        return {"agent": agent, "task": task}

    def _get_fastest_model(self, client) -> str:
        """Return the fastest available quantized model."""
        available = client.list().get("models", [])
        model_names = [self._extract_model_name(m) for m in available]

        # Priority: fastest first
        preferences = [
            "phi3.5:3.5-mini-instruct-q4_K_M",
            "llama3.2:3b",
            "phi3:mini",
            "tinyllama",
            self.settings.ollama_model,
        ]

        for pref in preferences:
            if pref in model_names:
                return pref

        return model_names[0] if model_names else self.settings.ollama_model

    @staticmethod
    def _extract_model_name(model: Any) -> str:
        if isinstance(model, dict):
            for key in ("name", "model"):
                value = model.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        for attr in ("name", "model"):
            value = getattr(model, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return str(model).strip()

    async def _execute_agent(self, agent: str, task: str) -> Dict[str, Any]:
        """Execute the specialist agent."""
        agent_map = {
            "scheduler": self._scheduler_stub,
            "mail": self._mail_stub,
            "finance": self._finance_stub,
            "researcher": self._researcher_stub,
            "manager": self._manager_stub,
        }
        handler = agent_map.get(agent, self._manager_stub)
        return await handler(task)

    async def _scheduler_stub(self, task: str) -> Dict[str, Any]:
        await self._emit_status("working", "calendar", "Accessing your calendar...", 0.5)

        try:
            lowered = task.lower()
            if "schedule" in lowered or "create" in lowered:
                result = await self._create_calendar_event(task)
                return {"status": "done", "message": result}

            events = await self._get_calendar_events(task)
            return {"status": "done", "message": events}
        except Exception as exc:
            return {"status": "done", "message": f"Calendar error: {str(exc)}"}

    def _split_calendar_steps(self, task: str) -> List[str]:
        """Split a multi-step scheduling request into individual instructions."""
        normalized = (task or "").strip()
        if not normalized:
            return []

        if not re.search(r"\b(?:first|then|next|after(?:\s+that)?|afterwards|also)\b", normalized, re.IGNORECASE):
            return [normalized]

        segments = [segment.strip() for segment in re.split(r"[,;]\s*", normalized) if segment and segment.strip()]
        if len(segments) <= 1:
            return [normalized]

        steps = []
        for segment in segments:
            lowered = segment.lower()
            if "with both" in lowered or ("both" in lowered and "and" in lowered):
                steps.append(segment)
            elif re.search(r"\b(?:meeting|schedule|book|create)\b", lowered):
                if steps and re.search(r"\b(?:john|hannah|stacy|carol)\b", lowered):
                    steps.append(segment)
                else:
                    steps.append(segment)
            else:
                if steps:
                    steps[-1] = f"{steps[-1]} {segment}".strip()
                else:
                    steps.append(segment)

        if not steps:
            return [normalized]
        return steps

    def _parse_single_calendar_instruction(
        self,
        task: str,
        now: datetime,
        local_tz: Any,
        default_date: str,
        default_start_time: str,
        default_duration_minutes: int,
        base_date: str | None = None,
        base_time: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Parse one scheduling instruction into one or more calendar events."""
        normalized_task = (task or "").strip()
        lowered = normalized_task.lower()
        title = "Meeting"

        if "with " in lowered:
            match = re.search(r"\bwith\s+([a-z0-9._-]+)", lowered)
            if match:
                participant = match.group(1).strip()
                participant = re.sub(r"\b(for|at|on|tomorrow|today|next|this)\b.*$", "", participant)
                participant = re.sub(r"\s+", " ", participant).strip()
                if participant:
                    title = f"Meeting with {participant.title()}"

        attendees: List[str] = []
        multi_meeting = False
        multi_match = re.search(r"\bwith\b(?:\s+both)?\s+([a-z0-9._-]+)(?:\s+and\s+([a-z0-9._-]+))", lowered)
        if multi_match:
            first_name = multi_match.group(1).strip()
            second_name = (multi_match.group(2) or "").strip()
            if first_name and second_name:
                attendees = [first_name, second_name]
                multi_meeting = True

        date = base_date or default_date
        if "tomorrow" in lowered:
            date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "today" in lowered:
            date = now.strftime("%Y-%m-%d")

        start_time = base_time or default_start_time
        time_match = None
        if "noon" in lowered:
            start_time = "12:00"
            time_match = True
        else:
            time_match = re.search(r"\b(\d{1,2})(?::?(\d{2}))?\s*(am|pm)?\b(?!\s*(minute|minutes|hour|hours)\b)", lowered)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                meridiem = (time_match.group(3) or "").lower()
                if hour > 12 and not meridiem:
                    time_match = None
                else:
                    if meridiem == "pm" and hour < 12:
                        hour += 12
                    if meridiem == "am" and hour == 12:
                        hour = 0
                    start_time = f"{hour:02d}:{minute:02d}"

        duration = default_duration_minutes
        duration_match = None
        half_hour_match = re.search(r"\bhalf(?:\s+an)?\s+hour\b|\bhalf-hour\b", lowered)
        if half_hour_match:
            duration = 30
        else:
            duration_match = re.search(r"\b(\d+)\s*(minute|minutes|hour|hours)\b", lowered)
            if duration_match:
                amount = int(duration_match.group(1))
                unit = duration_match.group(2).lower()
                duration = amount if unit.startswith("minute") else amount * 60

        should_use_llm = not (
            time_match is not None
            or duration_match is not None
            or "tomorrow" in lowered
            or "today" in lowered
            or "with " in lowered
        )

        if should_use_llm:
            try:
                ollama = _load_ollama_client()
                if ollama:
                    client = ollama.Client(host=self.settings.ollama_base_url)
                    model = self._get_fastest_model(client)

                    parse_prompt = f"""Extract the date, time, duration, and title from this scheduling request.

Request: "{task}"

Respond ONLY with this JSON format:
{{
    "title": "Meeting title",
    "date": "YYYY-MM-DD",
    "start_time": "HH:MM",
    "duration_minutes": 60
}}

If any field is unclear, use reasonable defaults:
- Date: tomorrow if not specified
- Start time: 9:00 AM if not specified
- Duration: 60 minutes if not specified
- Title: "Meeting" if not specified

Use the date and time from the user's request. Today is {now.strftime("%Y-%m-%d")}.
"""

                    response = client.chat(
                        model=model,
                        messages=[{"role": "user", "content": parse_prompt}],
                    )
                    content = response.get("message", {}).get("content", "")
                    data = json.loads(content)

                    if data.get("title"):
                        title = str(data["title"])
                    if data.get("date"):
                        date = str(data["date"])
                    if data.get("start_time"):
                        start_time = str(data["start_time"])
                    if data.get("duration_minutes"):
                        duration = int(data["duration_minutes"])
            except Exception:
                pass

        start_dt = datetime.strptime(f"{date}T{start_time}", "%Y-%m-%dT%H:%M").replace(tzinfo=local_tz)

        if multi_meeting and attendees:
            first_start = start_dt
            first_end = first_start + timedelta(minutes=duration)
            second_start = first_end
            second_end = second_start + timedelta(minutes=duration)
            return [
                {
                    "summary": f"Meeting with {attendees[0].title()}",
                    "start_dt": first_start,
                    "end_dt": first_end,
                    "date": date,
                    "start_time": first_start.strftime("%H:%M"),
                },
                {
                    "summary": f"Meeting with {attendees[1].title()}",
                    "start_dt": second_start,
                    "end_dt": second_end,
                    "date": date,
                    "start_time": second_start.strftime("%H:%M"),
                },
            ]

        return [{
            "summary": title,
            "start_dt": start_dt,
            "end_dt": start_dt + timedelta(minutes=duration),
            "date": date,
            "start_time": start_time,
        }]

    async def _create_calendar_event(self, task: str) -> str:
        """Create one or more calendar events from a scheduling request."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now()
        local_tz = now.astimezone().tzinfo or timezone.utc
        default_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        default_start_time = "09:00"
        default_duration_minutes = 60

        normalized_task = (task or "").strip()
        steps = self._split_calendar_steps(normalized_task)
        if not steps:
            steps = [normalized_task]

        created_events = []
        context_date = None
        context_time = None

        for step in steps:
            parsed_events = self._parse_single_calendar_instruction(
                task=step,
                now=now,
                local_tz=local_tz,
                default_date=default_date,
                default_start_time=default_start_time,
                default_duration_minutes=default_duration_minutes,
                base_date=context_date,
                base_time=context_time,
            )
            for event in parsed_events:
                event_id = create_event(
                    summary=event["summary"],
                    start_time=event["start_dt"].isoformat(),
                    end_time=event["end_dt"].isoformat(),
                )
                created_events.append((event, event_id))
                context_date = event["date"]
                context_time = event["start_time"]

        if not created_events:
            return "No calendar events were created."

        lines = []
        for event, event_id in created_events:
            lines.append(
                f"✅ Scheduled '{event['summary']}' for {event['date']} at {event['start_time']} for {int((event['end_dt'] - event['start_dt']).total_seconds() // 60)} minutes. Event ID: {event_id}"
            )
        return "\n".join(lines)

    async def _get_calendar_events(self, task: str) -> List[Dict[str, Any]]:
        """Fetch upcoming calendar events for the requested window."""
        del task
        return get_events(days=7)

    async def _mail_stub(self, task: str) -> Dict[str, Any]:
        await self._emit_status("working", "email", "Drafting email...", 0.5)
        cleaned_task = self._clean_task_text(task)
        return {"status": "stub", "message": f"I'll draft that email: {cleaned_task}"}

    async def _finance_stub(self, task: str) -> Dict[str, Any]:
        await self._emit_status("working", "financial", "Checking finances...", 0.5)
        cleaned_task = self._clean_task_text(task)
        return {"status": "stub", "message": f"Let me check your finances for: {cleaned_task}"}

    async def _researcher_stub(self, task: str) -> Dict[str, Any]:
        await self._emit_status("working", "researcher", "Searching...", 0.5)
        cleaned_task = self._clean_task_text(task)
        return {"status": "stub", "message": f"I'll research that for you: {cleaned_task}"}

    async def _manager_stub(self, task: str) -> Dict[str, Any]:
        """Handle general conversation and morning briefings."""
        await self._emit_status("working", "manager", "Thinking...", 0.5)

        lowered = task.lower().strip()

        greetings = {"hello", "hi", "hey", "greetings"}
        if lowered in greetings:
            return {"status": "done", "message": "Hello! How can I help you today?"}

        wellbeing_greetings = ["how are you", "how's it going", "how do you do", "what's up"]
        if any(term in lowered for term in wellbeing_greetings):
            return {"status": "done", "message": "I'm doing well! How can I assist you?"}

        polite_greetings = ["good morning", "good afternoon", "good evening"]
        if any(term in lowered for term in polite_greetings):
            return {"status": "done", "message": "Good morning! How can I help you today?"}

        briefing_keywords = ["morning briefing", "daily brief", "daily briefing", "briefing"]
        if lowered in briefing_keywords or any(keyword in lowered for keyword in briefing_keywords):
            briefing = self._get_briefing_cache()
            if briefing:
                return {"status": "done", "message": briefing.get("message", "No briefing available.")}
            return {"status": "done", "message": "I don't have a morning briefing cached. You can create one in data/briefing.json."}

        # For everything else, use the LLM when available
        ollama = _load_ollama_client()
        if ollama:
            client = ollama.Client(host=self.settings.ollama_base_url)
            model = self._get_fastest_model(client)
            try:
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": task}],
                )
                content = response.get("message", {}).get("content", "")
                if content:
                    return {"status": "done", "message": content}
            except Exception:
                pass

        # Fallback if LLM fails
        return {"status": "done", "message": "How can I help you today?"}

    def _get_briefing_cache(self) -> Dict[str, Any]:
        """Read the morning briefing from local cache."""
        try:
            cache_path = Path(__file__).resolve().parent.parent / "data" / "briefing.json"
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    async def _generate_final_response(self, request: str, agent: str, result: Dict[str, Any]) -> str:
        """Generate a natural language response from the agent result."""
        if result.get("status") == "stub":
            return self._clean_response_text(result.get("message", "Done."))
        return self._clean_response_text(result.get("message", "Done."))

    def _clean_response_text(self, text: str) -> str:
        """Remove routing prefixes and return a clean user-facing message."""
        if not text:
            return "How can I help you today?"

        text = text.strip()
        text = re.sub(r"^AGENT:\s*\w+\s*TASK:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\[[^\]]+\]\s*", "", text)
        text = re.sub(r"^\s+", "", text)
        return text or "How can I help you today?"

    def _clean_task_text(self, task: str) -> str:
        """Clean routing-style task text into a natural task description."""
        cleaned = self._clean_response_text(task)
        if cleaned.startswith("Task:"):
            cleaned = cleaned[len("Task:"):].strip()
        return cleaned or "your request"

    async def _emit_status(
        self,
        status: str,
        task: str,
        message: str,
        progress: float,
        final_response: str | None = None,
    ) -> None:
        payload = {
            "type": "status_update",
            "current_task": task,
            "message": message,
            "agent_states": {
                "manager": {"status": "idle", "thoughts": ""},
                "financial": {"status": "idle", "thoughts": ""},
                "calendar": {"status": "idle", "thoughts": ""},
                "email": {"status": "idle", "thoughts": ""},
                "researcher": {"status": "idle", "thoughts": ""},
            },
            "progress": progress,
            "final_response": final_response,
        }
        payload["agent_states"][task] = {"status": status, "thoughts": message}
        for callback in list(self._callbacks):
            result = callback(payload)
            if Awaitable is not None and hasattr(result, "__await__"):
                await result

    def _greeting_response(self) -> str:
        return "Hello! I'm your office manager. How can I help you today?"

    def _system_prompt(self) -> str:
        return (
            "You are the Axiom Office Manager. You have no direct access to calendars, email, or the web.\n"
            "Your only job is to triage user requests to one of four specialists:\n\n"
            "- Scheduler (calendar events)\n"
            "- Mail (draft emails only)\n"
            "- Finance (Rocket Money data)\n"
            "- Researcher (internet search)\n\n"
            "Output must be strictly parseable:\n"
            "AGENT: [Scheduler|Mail|Finance|Researcher|Manager]\n"
            "TASK: [Extracted task description]\n\n"
            "If the request is a morning greeting or general check-in, respond with a \n"
            "brief greeting and the morning briefing from the local cache. Do not \n"
            "invent data.\n\n"
            "Do not let the model free-associate. If it can't route, it should respond with: AGENT: Manager\nTASK: Please clarify your request."
        )
