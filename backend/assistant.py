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
from backend.tools.calendar_tool import create_event, delete_event, get_events, update_event

# Marks the start of a new meeting/appointment clause within a scheduling request: either the
# very start of the request, or a comma/semicolon/connector word ("and", "then", "next", "also",
# "first", "after that", "afterwards") immediately followed (allowing an optional verb like
# "schedule" or "book", an optional "I have", and an optional article) by "meeting(s)" or
# "appointment(s)". Requiring the noun to appear right after the boundary is what keeps this from
# firing on an "and" that is merely joining two attendees of one meeting (e.g. "with both Stacy
# and Carol") or on unrelated commas elsewhere in the sentence (attendee emails, a location, a
# description) -- those are never immediately followed by a fresh "a meeting"/"an appointment".
_CLAUSE_START_RE = re.compile(
    r"(?:^|(?:,\s*|;\s*)?\b(?:and|then|next|also|first|after(?:\s+that)?|afterwards)\b\s*|,\s*|;\s*)"
    r"(?:(?:schedule|book|create|set up)\s+)?"
    r"(?:i have\s+)?"
    r"(?:a|an|another|one)?\s*"
    r"(?:meetings?|appointments?)\b",
    re.IGNORECASE,
)


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
        self._conversation_history: List[Dict[str, Any]] = []

    def register_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None] | None]) -> None:
        self._callbacks.append(callback)

    def plan_request(self, request: str) -> List[str]:
        """Return a simple plan for a request so the UI can show the intended agent flow."""
        normalized = (request or "").strip().lower()
        if not normalized:
            return ["manager"]

        if any(term in normalized for term in ["morning briefing", "daily briefing", "briefing"]):
            return ["calendar", "email", "manager"]

        if any(term in normalized for term in ["finance", "subscription", "budget", "money", "spending"]):
            return ["financial", "researcher", "manager"]

        if any(term in normalized for term in ["schedule", "meeting", "calendar", "appointment", "book", "reserve"]):
            return ["calendar", "manager"]

        return ["manager"]

    def _fallback_response(self, request: str, history: List[Dict[str, Any]]) -> str:
        """Return a friendly fallback response when local processing cannot be completed."""
        del history
        return f"I’m unable to produce a reliable response for '{request}', but I can still help with local calendar scheduling and fallback planning."

    def _is_greeting(self, request: str) -> bool:
        """Return True for simple greeting-style requests."""
        normalized = (request or "").strip().lower()
        if not normalized:
            return False
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
        if normalized in greeting_terms:
            return True
        if normalized.startswith(tuple(greeting_terms)):
            return normalized.split()[0] in {"hello", "hi", "hey", "greetings"}
        return False

    def _looks_like_scheduling_request(self, task: str) -> bool:
        """Return True when a request looks like a scheduling or calendar task."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False
        return any(term in lowered for term in ["schedule", "meeting", "calendar", "appointment", "book", "reserve"])

    def _contains_reschedule_terms(self, task: str) -> bool:
        """Return True when a request should update an existing meeting rather than create a new one."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False

        reschedule_terms = [
            "reschedule",
            "reschedule that",
            "move it",
            "move that",
            "change that",
            "change it",
            "change the time",
            "change time",
            "push back",
            "shift it",
            "shift that",
            "update it",
            "update that",
            "please change",
        ]
        if any(term in lowered for term in reschedule_terms):
            return True

        correction_patterns = [
            r"\b(?:nope|actually|sorry|wait|instead|i meant|i said|make it|change it to|change that to|move it to|shift it to)\b",
            r"\b(?:that is|that's|this is|that was|this was)\s+(?:friday|saturday|sunday|monday|tuesday|wednesday|thursday|tomorrow|today|next|this)\b",
            r"\b(?:not|wrong|different)\b.*\b(?:friday|saturday|sunday|monday|tuesday|wednesday|thursday|tomorrow|today|next|this)\b",
        ]
        return any(re.search(pattern, lowered) for pattern in correction_patterns)

    def _get_previous_user_request(self, current_request: str | None = None) -> str:
        """Return the most recent user request from earlier in the conversation, excluding the current turn."""
        current_text = (current_request or "").strip()
        for entry in reversed(self._conversation_history):
            if entry.get("role") != "user":
                continue
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            if current_text and content == current_text:
                continue
            return content
        return ""

    def _has_prior_scheduling_context(self, task: str) -> bool:
        """Return True when the conversation history already contains scheduling context relevant to the current turn."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False
        for entry in reversed(self._conversation_history):
            if entry.get("role") != "user":
                continue
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            if content.lower() == lowered:
                continue
            if self._looks_like_scheduling_request(content):
                return True
            content_lower = content.lower()
            if re.search(r"\b(?:with|tomorrow|today|noon|midday|am|pm|at|for)\b", content_lower):
                return True
        return False

    def _looks_like_follow_up_to_scheduling_request(self, task: str, prior_task: str) -> bool:
        """Return True when a short follow-up should inherit the previous scheduling context."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False
        if not prior_task or not str(prior_task).strip():
            return False
        if str(prior_task).strip().lower() == lowered:
            return False
        if re.search(r"^\s*(?:schedule|create|book|set)\b", lowered):
            return False
        if self._contains_reschedule_terms(prior_task):
            return False
        if self._looks_like_availability_request(task):
            return True
        if self._looks_like_scheduling_request(prior_task) and re.search(r"\b(?:noon|midday|today|tomorrow|am|pm|[0-9]{1,2}(?::[0-9]{2})?|for|with|at)\b", lowered):
            return True
        if self._looks_like_scheduling_request(prior_task) and re.search(r"\b(?:half(?:\s+an)?\s+hour|half-hour|\d+\s*(?:min|mins|minute|minutes|hr|hrs|hour|hours)|[0-9]{1,2}(?::[0-9]{2})?)\b", lowered):
            return True
        if self._looks_like_scheduling_request(prior_task) and re.search(r"\b(?:let's|let us|make it|change it|do it|set it|shift it|actually|instead)\b", lowered):
            return True
        return False

    def _should_use_prior_scheduling_context(self, task: str, prior_task: str | None = None) -> bool:
        """Return True when a current turn should inherit earlier scheduling context."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False
        if re.search(r"^\s*(?:schedule|create|book|set)\b", lowered):
            return False

        previous_request = prior_task if prior_task is not None else self._get_previous_user_request(task)
        if not previous_request or not str(previous_request).strip():
            return False

        if self._looks_like_availability_request(task):
            return True
        if self._looks_like_follow_up_to_scheduling_request(task, previous_request):
            return True

        if self._has_prior_scheduling_context(task) and re.search(r"\b(?:noon|midday|today|tomorrow|am|pm|for|with|at|let's|let us|make it|change it|do it|set it|shift it|actually|instead|half(?:\s+an)?\s+hour|half-hour|\d+\s*(?:min|mins|minute|minutes|hr|hrs|hour|hours)|[0-9]{1,2}(?::[0-9]{2})?)\b", lowered):
            return True
        return False

    _TIME_OF_DAY_RE = re.compile(
        r"\b(\d{1,2})(?::?(\d{2}))?\s*(am|pm)?\b(?!\s*(?:minute|minutes|hour|hours)\b)",
        re.IGNORECASE,
    )

    def _parse_time_of_day(self, text: str) -> str | None:
        """Parse a clock time like "2pm", "14:30", or "noon" from text into "HH:MM", or None.

        Accepts an explicit am/pm hour (1-12) or an unambiguous 24-hour hour (13-23 or 0) without
        needing am/pm. Previously any hour above 12 was rejected outright even when unambiguous
        (e.g. "17:00"), which incorrectly triggered a request for clarification on a time the user
        had already given. A bare hour from 1-12 with no am/pm ("at 5") is genuinely ambiguous
        between morning and afternoon, so it is treated as no time given rather than silently
        assumed to be AM -- callers should ask the user to say which they meant (see
        _detect_ambiguous_hour).
        """
        lowered = (text or "").strip().lower()
        if not lowered:
            return None
        if "noon" in lowered:
            return "12:00"

        match = self._TIME_OF_DAY_RE.search(lowered)
        if not match:
            return None

        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = (match.group(3) or "").lower()

        if hour > 23 or minute > 59:
            return None
        if hour > 12 and meridiem:
            return None
        if 1 <= hour <= 12 and not meridiem:
            return None

        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    def _detect_ambiguous_hour(self, text: str) -> int | None:
        """Return the bare 1-12 hour in text that has no am/pm marker (e.g. the 5 in "at 5"), or
        None if there is no such ambiguity. Used to ask "did you mean 5 AM or PM?" instead of the
        generic "I still need the time" when the user did give a time but it was ambiguous.
        """
        lowered = (text or "").strip().lower()
        if not lowered or "noon" in lowered:
            return None

        match = self._TIME_OF_DAY_RE.search(lowered)
        if not match:
            return None

        hour = int(match.group(1))
        meridiem = (match.group(3) or "").lower()
        if 1 <= hour <= 12 and not meridiem:
            return hour
        return None

    _PARTICIPANT_STOP_WORDS_RE = re.compile(
        r"\b(?:for|at|on|to|tomorrow|today|next|this|starting|invite|description|location|in|from)\b",
        re.IGNORECASE,
    )

    def _extract_participant_phrase(self, lowered_text: str) -> str | None:
        """Extract who a meeting is with from the phrase following "with".

        Captures up to a few words rather than a single token, so a collective reference such as
        "the whole team" is kept together instead of only grabbing the stopword "the" (which
        previously produced a nonsense title like "Meeting with The").
        """
        match = re.search(r"\bwith\s+((?:[a-z0-9._-]+\s*){1,4})", lowered_text)
        if not match:
            return None

        phrase = match.group(1)
        stop_match = self._PARTICIPANT_STOP_WORDS_RE.search(phrase)
        if stop_match:
            phrase = phrase[: stop_match.start()]
        phrase = re.sub(r"\s+", " ", phrase).strip()
        return phrase or None

    def _extract_location_phrase(self, lowered_text: str) -> str | None:
        """Extract a meeting location from phrasing like "conference room", "location: X",
        "set the address to X", or "at the X". Shared by event creation and by updating an
        existing event's location so both recognize the same phrasings consistently.
        """
        if "conference room" in lowered_text:
            return "conference room"
        if "location:" in lowered_text:
            match = re.search(r"location\s*:\s*([a-z0-9 ._\-]+)", lowered_text)
            if match:
                return match.group(1).strip().title()
        elif re.search(r"\b(?:address|location)\b", lowered_text):
            match = re.search(r"\b(?:address|location)\b(?:\s+(?:is|to))?\s*(?:the\s+)?([a-z0-9 ._\-]+)", lowered_text)
            if match:
                return match.group(1).strip().title()
        elif " at the " in lowered_text:
            match = re.search(r"\bat the ([a-z0-9 ._\-]+)", lowered_text)
            if match:
                return match.group(1).strip().title()
        return None

    def _extract_scheduling_details(self, task: str, use_history: bool = True) -> Dict[str, Any]:
        """Extract the most relevant scheduling fields from the current request and, for true follow-ups, recent conversation history.

        Pass use_history=False to parse the given text in isolation, ignoring conversation
        follow-up context entirely. This is used when evaluating one clause of a compound
        scheduling request (e.g. the second meeting in "schedule X, and schedule Y"), where the
        clause is itself a substring of the current turn and must not be treated as a follow-up
        to its own parent request.
        """
        details: Dict[str, Any] = {"participant": None, "title": None, "date": None, "time": None, "duration": None}
        texts: List[str] = []

        normalized_task = str(task or "").strip()
        if normalized_task:
            texts.append(normalized_task)

        if use_history:
            if self._looks_like_reschedule_request(normalized_task):
                last_event_context = self._extract_last_event_context()
                summary = str(last_event_context.get("summary") or "").strip()
                if summary:
                    texts.append(summary)
            else:
                previous_request = self._get_previous_user_request(normalized_task)
                if self._should_use_prior_scheduling_context(normalized_task, previous_request):
                    for entry in self._conversation_history:
                        if entry.get("role") != "user":
                            continue
                        content = str(entry.get("content", "")).strip()
                        if content:
                            texts.append(content)

        for text in texts:
            lowered = (text or "").strip().lower()
            if details["participant"] is None:
                participant = self._extract_participant_phrase(lowered)
                if participant:
                    details["participant"] = participant

            if details["title"] is None:
                title_match = re.search(
                    r"\b(?:schedule|create|book|set)\b(?:\s+(?:a|an))?\s+([a-z0-9 ._\-]+?)(?=\s+(?:for|at|on|tomorrow|today|with|starting|next|this|invite|description|location|in|from))",
                    text,
                    re.IGNORECASE,
                )
                if title_match:
                    inferred_title = title_match.group(1).strip()
                    if inferred_title and inferred_title.lower() not in {"meeting", "event", "an event", "a meeting"}:
                        details["title"] = inferred_title

            if details["date"] is None:
                if "tomorrow" in lowered:
                    details["date"] = "tomorrow"
                elif "today" in lowered:
                    details["date"] = "today"
                else:
                    weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                    for day_name in weekday_names:
                        if day_name in lowered:
                            details["date"] = day_name
                            break

            if details["time"] is None:
                parsed_time = self._parse_time_of_day(lowered)
                if parsed_time is not None:
                    details["time"] = parsed_time

            if details["duration"] is None:
                duration_minutes = self._extract_duration_minutes(text)
                if duration_minutes is not None:
                    details["duration"] = duration_minutes

        return details

    def _format_duration_label(self, duration_minutes: int) -> str:
        """Return a human-readable duration label for a scheduling request."""
        if duration_minutes >= 60:
            hours = duration_minutes // 60
            if hours == 1:
                return f"one hour ({duration_minutes} minutes)"
            return f"{hours} hours ({duration_minutes} minutes)"
        if duration_minutes == 30:
            return "30 minutes"
        return f"{duration_minutes} minutes"

    def _compose_contextual_request(self, details: Dict[str, Any], fallback_request: str, source_request: str | None = None) -> str:
        """Create a fuller scheduling request from the inferred conversation context."""
        if not any(value is not None for value in details.values()):
            return fallback_request

        parts: List[str] = ["schedule a meeting"]
        if details.get("date"):
            parts.append(str(details["date"]))
        if details.get("title"):
            parts.append(str(details["title"]))
        if details.get("participant"):
            parts.append(f"with {details['participant']}")
        if details.get("time"):
            time_value = details["time"]
            if str(time_value) == "12:00" and source_request and "noon" in str(source_request).lower():
                time_value = "noon"
            parts.append(f"at {time_value}")
        if details.get("duration") is not None:
            duration_minutes = int(details["duration"])
            parts.append(f"for {self._format_duration_label(duration_minutes)}")
        composed = " ".join(parts).strip()
        if composed == "schedule a meeting":
            return fallback_request
        return composed

    def _build_contextual_request(self, request: str) -> str:
        """Build a richer scheduling request from the latest follow-up and prior conversation context."""
        normalized = (request or "").strip()
        if not normalized:
            return request

        prior_request = self._get_previous_user_request(normalized)
        if not prior_request:
            return normalized

        contextual_request = normalized
        details = self._extract_scheduling_details(normalized)
        if self._should_use_prior_scheduling_context(normalized, prior_request):
            base_request = f"{prior_request} {normalized}".strip()
            if self._looks_like_availability_request(normalized):
                contextual_request = self._compose_contextual_request(details, base_request, normalized)
                contextual_request = f"{contextual_request} based on my schedule".strip()
            else:
                contextual_request = self._compose_contextual_request(details, base_request, normalized)

        if self._looks_like_reschedule_request(normalized):
            if contextual_request.lower().startswith("schedule"):
                contextual_request = contextual_request.replace("schedule", "reschedule", 1)
            elif not contextual_request.lower().startswith("reschedule"):
                contextual_request = f"reschedule {contextual_request}".strip()

        is_follow_up = self._looks_like_follow_up_to_scheduling_request(normalized, prior_request)
        duration_minutes = self._extract_duration_minutes(normalized)
        if duration_minutes is None and is_follow_up:
            duration_minutes = self._extract_duration_minutes(prior_request)
        if duration_minutes is not None and is_follow_up:
            duration_phrase = self._format_duration_label(duration_minutes)
            has_duration = re.search(r"\b(?:minute|minutes|mins|min|hour|hours|hr|hrs|h)\b", contextual_request.lower())
            if not has_duration:
                contextual_request = f"{contextual_request} for {duration_phrase}".strip()

        return contextual_request

    def _extract_duration_minutes(self, task: str) -> int | None:
        """Extract a duration in minutes from the request, if present."""
        lowered = (task or "").strip().lower()
        half_hour_match = re.search(r"\bhalf(?:\s+an)?\s+hour\b|\bhalf-hour\b", lowered)
        if half_hour_match:
            return 30

        word_to_number = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "a": 1,
            "an": 1,
        }

        duration_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s*(minute|minutes|mins|min|hour|hours|hr|hrs|h)\b", lowered)
        if not duration_match:
            return None

        amount_token = duration_match.group(1).lower()
        unit = duration_match.group(2).lower()
        if amount_token in word_to_number:
            amount = word_to_number[amount_token]
        else:
            try:
                amount = int(amount_token)
            except ValueError:
                return None
        return amount if unit.startswith("minute") or unit in {"min", "mins", "m"} else amount * 60

    def _looks_like_cancel_request(self, task: str) -> bool:
        """Return True when the request is asking to cancel or delete an existing meeting."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False
        if not re.search(r"\b(?:cancel|delete|remove)\b", lowered):
            return False
        return bool(re.search(r"\b(?:meeting|event|appointment|call)\b", lowered))

    async def _cancel_calendar_event(self, task: str) -> str:
        """Find and delete the calendar event matching a cancel/delete request.

        Matches on the requested day (if any) and on the participant/title mentioned in the
        request, since the user refers to meetings by who they are with or what they are about,
        not by an internal event ID.
        """
        requested_day = self._extract_requested_day(task, datetime.now())
        details = self._extract_scheduling_details(task, use_history=False)
        search_term = (details.get("participant") or details.get("title") or "").strip().lower()

        events = get_events(days=60)
        candidates = []
        for event in events:
            if requested_day and self._get_event_date(event) != requested_day:
                continue
            summary = str(event.get("summary") or "")
            if search_term and search_term not in summary.lower():
                continue
            candidates.append(event)

        if not candidates:
            return "I couldn't find a matching meeting on your calendar to cancel."

        if len(candidates) > 1:
            options = "; ".join(f"'{c.get('summary')}' at {c.get('start')}" for c in candidates)
            return f"I found more than one matching meeting ({options}). Which one should I cancel?"

        event = candidates[0]
        event_id = event.get("id")
        summary = event.get("summary") or "that meeting"
        if not event_id:
            return f"I found '{summary}' but couldn't determine its event ID to cancel it."

        delete_event(event_id)
        return f"🗑️ I canceled '{summary}'."

    def _looks_like_detail_update_request(self, task: str) -> bool:
        """Return True when the request asks to change a meeting's location or attendees.

        Deliberately distinct from a reschedule request, which changes the meeting's time --
        "change the location of my meeting with Hannah to the conference room" should not go
        through the time-focused reschedule flow.
        """
        lowered = (task or "").strip().lower()
        if not lowered:
            return False
        if not re.search(r"\b(?:meeting|event|appointment)\b", lowered):
            return False

        wants_location_change = bool(re.search(r"\b(?:location|address|venue)\b", lowered)) and bool(
            re.search(r"\b(?:change|update|set)\b", lowered)
        )
        wants_attendee_change = bool(re.search(r"\b(?:invite|add)\b", lowered)) and bool(re.search(r"\bto\b", lowered))
        return wants_location_change or wants_attendee_change

    async def _update_calendar_event_details(self, task: str) -> str:
        """Change an existing meeting's location and/or attendees without touching its time.

        Looks the meeting up directly on the real calendar (rather than this conversation's
        history) because updating it means resending its whole event body -- update_event has no
        way to patch a single field -- so the current start/end time must come from an
        authoritative source, not a coarse HH:MM string that reschedule's session-context tracking
        keeps.
        """
        details = self._extract_scheduling_details(task, use_history=False)
        search_term = (details.get("participant") or details.get("title") or "").strip().lower()
        if not search_term:
            return "I don't have a calendar event to update -- who is the meeting with, or what is it called?"

        events = get_events(days=60)
        candidates = [event for event in events if search_term in str(event.get("summary") or "").lower()]
        if not candidates:
            return "I couldn't find a matching meeting on your calendar to update."
        if len(candidates) > 1:
            options = "; ".join(f"'{c.get('summary')}' at {c.get('start')}" for c in candidates)
            return f"I found more than one matching meeting ({options}). Which one did you mean?"

        event = candidates[0]
        event_id = event.get("id")
        summary = event.get("summary") or "that meeting"
        if not event_id:
            return f"I found '{summary}' but couldn't determine its event ID to update it."

        lowered = task.lower()
        new_location = self._extract_location_phrase(lowered)
        new_attendees = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", task)

        if not new_location and not new_attendees:
            return f"I found '{summary}' but couldn't tell what to change -- what should the new location or attendee be?"

        update_kwargs: Dict[str, Any] = {
            "event_id": event_id,
            "summary": summary,
            "start_time": event.get("start"),
            "end_time": event.get("end"),
        }
        if new_location:
            update_kwargs["location"] = new_location
        if new_attendees:
            update_kwargs["attendees"] = new_attendees

        updated_event_id = update_event(**update_kwargs)

        changes = []
        if new_location:
            changes.append(f"location to {new_location}")
        if new_attendees:
            changes.append(f"attendees to include {', '.join(new_attendees)}")
        change_summary = " and ".join(changes)

        if "local fallback" in str(updated_event_id).lower():
            return f"⚠️ I updated the {change_summary} for '{summary}' and saved a local backup entry. Backup ID: {updated_event_id}"
        return f"✅ I updated the {change_summary} for '{summary}'."

    def _looks_like_availability_request(self, task: str) -> bool:
        """Return True when the request is asking for a suggested or available meeting slot."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False

        if "suggest" in lowered and ("time" in lowered or "slot" in lowered or "available" in lowered):
            return True
        if "available" in lowered and ("time" in lowered or "slot" in lowered):
            return True
        if "do you have" in lowered and ("time" in lowered or "slot" in lowered or "suggest" in lowered):
            return True
        if "based on my schedule" in lowered or "based on my calendar" in lowered:
            return True
        if "what time works" in lowered or "what works" in lowered:
            return True
        return False

    def _event_time_range(self, event: Dict[str, Any], local_tz: Any) -> "tuple[datetime, datetime] | None":
        """Return an event's (start, end) as timezone-aware datetimes in local_tz, or None if unparseable."""
        start_value = event.get("start")
        end_value = event.get("end")
        if not start_value or not end_value:
            return None
        try:
            event_start = datetime.fromisoformat(str(start_value).replace("Z", "+00:00"))
            event_end = datetime.fromisoformat(str(end_value).replace("Z", "+00:00"))
        except ValueError:
            return None

        if event_start.tzinfo is None:
            event_start = event_start.replace(tzinfo=local_tz)
        if event_end.tzinfo is None:
            event_end = event_end.replace(tzinfo=local_tz)

        return event_start.astimezone(local_tz), event_end.astimezone(local_tz)

    def _find_conflicting_events(
        self, events: List[Dict[str, Any]], start_dt: datetime, end_dt: datetime, local_tz: Any
    ) -> List[Dict[str, Any]]:
        """Return events from the given list that overlap the [start_dt, end_dt) window."""
        conflicts = []
        for event in events:
            time_range = self._event_time_range(event, local_tz)
            if time_range is None:
                continue
            event_start, event_end = time_range
            if start_dt < event_end and end_dt > event_start:
                conflicts.append(event)
        return conflicts

    async def _suggest_available_time(self, task: str) -> str:
        """Suggest the earliest free slot for a meeting on the requested day."""
        now = datetime.now()
        requested_day = self._extract_requested_day(task, now)
        duration_minutes = self._extract_duration_minutes(task) or 60
        if requested_day is None:
            requested_day = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        events = get_events(days=7)
        requested_events = []
        for event in events:
            event_date = self._get_event_date(event)
            if event_date == requested_day:
                requested_events.append(event)

        requested_date = datetime.strptime(requested_day, "%Y-%m-%d")
        local_tz = now.astimezone().tzinfo or timezone.utc

        candidate_starts = []
        for hour in range(9, 18):
            candidate_dt = datetime.combine(requested_date.date(), datetime.min.time().replace(hour=hour), tzinfo=local_tz)
            candidate_starts.append(candidate_dt)

        for candidate_start in candidate_starts:
            candidate_end = candidate_start + timedelta(minutes=duration_minutes)
            if not self._find_conflicting_events(requested_events, candidate_start, candidate_end, local_tz):
                return f"I suggest {candidate_start.strftime('%H:%M')} for {requested_day}."

        return f"I don't see a clear opening on {requested_day}."

    async def run(self, request: str) -> Dict[str, Any]:
        """Route the request to the right agent and execute."""
        if self._is_greeting(request):
            # Preserve the older assistant behavior for greetings by returning the
            # model-driven fallback response text when ollama is available.
            ollama = _load_ollama_client()
            model_status = "ollama" if ollama is not None else "fallback"
            if ollama is not None:
                try:
                    client = ollama.Client(host=self.settings.ollama_base_url)
                    response = client.chat(
                        model=self._get_fastest_model(client),
                        messages=[{"role": "user", "content": request}],
                    )
                    response_text = response.get("message", {}).get("content", "")
                    if response_text:
                        return {
                            "response": response_text,
                            "agent": "manager",
                            "task": request,
                            "result": {"status": "done", "message": response_text},
                            "plan": self.plan_request(request),
                            "model_status": model_status,
                        }
                except Exception:
                    pass

            response = self._greeting_response()
            return {
                "response": response,
                "agent": "manager",
                "task": request,
                "result": {"status": "done", "message": response},
                "plan": self.plan_request(request),
                "model_status": model_status,
            }

        await self._emit_status("working", "manager", "Routing your request...", 0.1)

        contextual_request = self._build_contextual_request(request)
        self._conversation_history.append({"role": "user", "content": request})

        # Step 1: Route with the LLM
        route = await self._route_request(contextual_request)
        agent = route["agent"]
        task = route["task"]

        await self._emit_status("working", agent, f"Processing {agent} task...", 0.3)

        # Step 2: Execute the right agent
        result = await self._execute_agent(agent, task)

        # Step 3: Generate final response
        response = await self._generate_final_response(request, agent, result)
        response = self._clean_response_text(response)

        model_status = "ollama" if _load_ollama_client() is not None else "fallback"

        self.memory.add_memory(
            "assistant",
            f"User: {request} | Routed to: {agent} | Response: {response}",
            metadata={"kind": "assistant"}
        )
        self._conversation_history.append({"role": "assistant", "content": response})

        await self._emit_status("idle", "manager", "Ready.", 1.0, final_response=response)
        return {
            "response": response,
            "agent": agent,
            "task": task,
            "result": result,
            "plan": self.plan_request(request),
            "model_status": model_status,
        }

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

        if self._looks_like_reschedule_request(normalized):
            return {"agent": "scheduler", "task": normalized}

        if self._looks_like_cancel_request(normalized):
            return {"agent": "scheduler", "task": normalized}

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

        if not model_names:
            try:
                client.pull(self.settings.ollama_model)
            except Exception:
                pass
            return self.settings.ollama_model

        return model_names[0]

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

    def _looks_like_calendar_query(self, task: str) -> bool:
        """Return True when a request is asking to read calendar information instead of create an event."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False

        if "?" in lowered:
            return True

        query_starters = [
            "what ",
            "what's ",
            "what is ",
            "show me",
            "show my",
            "list",
            "tell me",
            "when",
            "where",
            "do i have",
            "do i",
            "am i",
            "can you",
            "can i",
            "any meetings",
            "upcoming",
            "suggested time",
            "suggest a time",
            "suggested slot",
            "available time",
            "available slot",
            "based on my schedule",
            "do you have",
        ]
        if any(lowered.startswith(starter) for starter in query_starters):
            return True

        if "on my calendar" in lowered or "on the schedule" in lowered or "in my calendar" in lowered:
            return True

        if "suggested time" in lowered or "suggest a time" in lowered or "suggested slot" in lowered:
            return True

        if "available time" in lowered or "available slot" in lowered or "based on my schedule" in lowered:
            return True

        if lowered.startswith("what") and ("schedule" in lowered or "calendar" in lowered):
            return True

        return False

    async def _scheduler_stub(self, task: str) -> Dict[str, Any]:
        await self._emit_status("working", "calendar", "Accessing your calendar...", 0.5)

        try:
            if self._looks_like_cancel_request(task):
                result = await self._cancel_calendar_event(task)
                return {"status": "done", "message": result}

            if self._looks_like_detail_update_request(task):
                result = await self._update_calendar_event_details(task)
                return {"status": "done", "message": result}

            if self._looks_like_availability_request(task):
                result = await self._suggest_available_time(task)
                return {"status": "done", "message": result}

            if self._looks_like_calendar_query(task):
                events = await self._get_calendar_events(task)
                return {"status": "done", "message": events}

            lowered = task.lower()
            if self._looks_like_reschedule_request(task):
                result = await self._create_calendar_event(task)
                return {"status": "done", "message": result}
            if "schedule" in lowered or "create" in lowered or "book" in lowered or "reserve" in lowered:
                result = await self._create_calendar_event(task)
                return {"status": "done", "message": result}

            events = await self._get_calendar_events(task)
            return {"status": "done", "message": events}
        except Exception as exc:
            return {"status": "done", "message": f"Calendar error: {str(exc)}"}

    def _split_calendar_steps(self, task: str) -> List[str]:
        """Split a multi-meeting scheduling request into individual meeting instructions.

        Splits at each point _CLAUSE_START_RE finds a new meeting/appointment being introduced,
        so qualifiers that follow a clause (a time, a duration, "both half an hour") stay attached
        to it up until the next such boundary, instead of relying on a comma plus a specific
        trigger word both being present (which missed plain phrasings like "..., and a meeting
        with Will on Friday").
        """
        normalized = (task or "").strip()
        if not normalized:
            return []

        boundaries = sorted({0, *(match.start() for match in _CLAUSE_START_RE.finditer(normalized))})
        if len(boundaries) <= 1:
            return [normalized]

        steps = []
        for start, end in zip(boundaries, boundaries[1:] + [len(normalized)]):
            clause = normalized[start:end].strip()
            if clause:
                steps.append(clause)
        return steps or [normalized]

    def _extract_contextual_title(self, task: str) -> str | None:
        """Infer a meeting title from the current request and earlier scheduling context when it is a short follow-up."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return None

        prior_request = ""
        for entry in reversed(self._conversation_history):
            if entry.get("role") == "user":
                prior_request = str(entry.get("content", "")).strip()
                break
        if not prior_request or not (
            self._looks_like_follow_up_to_scheduling_request(task, prior_request)
            or self._has_prior_scheduling_context(task)
        ):
            return None

        for entry in reversed(self._conversation_history):
            content = str(entry.get("content", ""))
            if not content:
                continue
            content_lower = content.lower()
            if entry.get("role") == "user":
                participant = self._extract_participant_phrase(content_lower)
                if participant:
                    return f"Meeting with {participant.title()}"

                if any(term in content_lower for term in ["schedule", "meeting", "book", "reserve", "calendar"]):
                    summary_match = re.search(r"(?:schedule|create|book|set)\s+(?:a|an)?\s+([a-z0-9 ._\-]+)", content, re.IGNORECASE)
                    if summary_match:
                        inferred = summary_match.group(1).strip()
                        if inferred.lower() not in {"meeting", "event", "an event", "a meeting"}:
                            return inferred[0].upper() + inferred[1:] if inferred else inferred

            if entry.get("role") == "assistant":
                summary_match = re.search(r"(?:scheduled|prepared a local backup entry for|updated)\s+['\"]([^'\"]+)['\"]", content, re.IGNORECASE)
                if summary_match:
                    inferred = summary_match.group(1).strip()
                    if inferred.lower() not in {"meeting", "event", "meeting or event"}:
                        return inferred

        return None

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

        title_match = re.search(
            r"\b(?:schedule|create|book|set)\b(?:\s+(?:a|an))?\s+([a-z0-9 ._\-]+?)(?=\s+(?:for|at|on|tomorrow|today|with|starting|next|this|invite|description|location|in|from))",
            normalized_task,
            re.IGNORECASE,
        )
        if title_match:
            inferred_title = title_match.group(1).strip()
            if inferred_title and inferred_title.lower() != "meeting":
                title = inferred_title[0].upper() + inferred_title[1:] if inferred_title else inferred_title

        if "with " in lowered:
            participant = self._extract_participant_phrase(lowered)
            if participant:
                title = f"Meeting with {participant.title()}"

        contextual_title = self._extract_contextual_title(normalized_task)
        if contextual_title and title.lower() in {"meeting", "event", "meeting or event"}:
            title = contextual_title

        attendees: List[str] = []
        participant_names: List[str] = []
        location: str | None = None
        recurrence: List[str] | None = None
        description: str | None = None
        multi_meeting = False
        multi_match = re.search(r"\bwith\b(?:\s+both)?\s+([a-z0-9._-]+)(?:\s+and\s+([a-z0-9._-]+))", lowered)
        if multi_match:
            first_name = multi_match.group(1).strip()
            second_name = (multi_match.group(2) or "").strip()
            if first_name and second_name:
                participant_names = [first_name, second_name]
                multi_meeting = True

        attendees = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", normalized_task)

        location = self._extract_location_phrase(lowered)

        if "weekly recurring" in lowered or "every week" in lowered:
            weekday_match = re.search(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b", lowered)
            if weekday_match:
                weekday_map = {
                    "monday": "MO",
                    "tuesday": "TU",
                    "wednesday": "WE",
                    "thursday": "TH",
                    "friday": "FR",
                    "saturday": "SA",
                    "sunday": "SU",
                }
                recurrence = [f"RRULE:FREQ=WEEKLY;BYDAY={weekday_map[weekday_match.group(1)]}"]
            else:
                recurrence = ["RRULE:FREQ=WEEKLY"]

        description_match = re.search(r"(?:description|details|notes?)\s*[:\-]\s*(.+)", normalized_task, re.IGNORECASE)
        if description_match:
            description = description_match.group(1).strip()

        requested_date = self._extract_requested_day(normalized_task, now)
        date = requested_date or base_date or default_date

        start_time = base_time or default_start_time
        parsed_time = self._parse_time_of_day(lowered)
        time_match = parsed_time is not None
        if parsed_time is not None:
            start_time = parsed_time

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

        if multi_meeting and participant_names:
            first_start = start_dt
            first_end = first_start + timedelta(minutes=duration)
            second_start = first_end
            second_end = second_start + timedelta(minutes=duration)
            return [
                {
                    "summary": f"Meeting with {participant_names[0].title()}",
                    "start_dt": first_start,
                    "end_dt": first_end,
                    "date": date,
                    "start_time": first_start.strftime("%H:%M"),
                    "location": location,
                    "attendees": attendees,
                    "recurrence": recurrence,
                    "description": description,
                },
                {
                    "summary": f"Meeting with {participant_names[1].title()}",
                    "start_dt": second_start,
                    "end_dt": second_end,
                    "date": date,
                    "start_time": second_start.strftime("%H:%M"),
                    "location": location,
                    "attendees": attendees,
                    "recurrence": recurrence,
                    "description": description,
                },
            ]

        return [{
            "summary": title,
            "start_dt": start_dt,
            "end_dt": start_dt + timedelta(minutes=duration),
            "date": date,
            "start_time": start_time,
            "location": location,
            "attendees": attendees,
            "recurrence": recurrence,
            "description": description,
        }]

    def _extract_requested_day(self, task: str, now: datetime) -> str | None:
        """Return a YYYY-MM-DD date when the request refers to a specific day such as tomorrow or this Saturday."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return None

        if "tomorrow" in lowered:
            return (now + timedelta(days=1)).strftime("%Y-%m-%d")
        if "today" in lowered:
            return now.strftime("%Y-%m-%d")

        weekday_names = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        for day_name, weekday in weekday_names.items():
            if day_name in lowered:
                days_ahead = (weekday - now.weekday()) % 7
                # "next <day>" means the day after the nearest upcoming one, not the nearest one
                # itself -- e.g. said on a Monday, "this Friday" is 4 days out but "next Friday"
                # is 11 days out. Previously "next" only pushed the date out when the nearest
                # occurrence was today, so "next Friday" and "this Friday" resolved to the same
                # date on every other day of the week.
                if re.search(rf"\bnext\s+{day_name}\b", lowered):
                    days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        return None

    def _get_event_date(self, event: Dict[str, Any]) -> str | None:
        """Extract the event date from the start value, if present."""
        start_value = event.get("start")
        if isinstance(start_value, str) and start_value:
            if "T" in start_value:
                try:
                    return datetime.fromisoformat(start_value.replace("Z", "+00:00")).date().strftime("%Y-%m-%d")
                except ValueError:
                    return None
            try:
                return datetime.fromisoformat(start_value).date().strftime("%Y-%m-%d")
            except ValueError:
                return None
        return None

    def _build_clarification_response(self, task: str) -> str:
        """Create a clarification prompt for missing scheduling details using the current conversation context."""
        details = self._extract_scheduling_details(task)
        missing: List[str] = []

        if not details.get("participant") and not details.get("title"):
            missing.append("who the meeting is with or what it is about")

        if not details.get("date"):
            missing.append("the date")

        ambiguous_hour = self._detect_ambiguous_hour(task)
        if not details.get("time"):
            if ambiguous_hour is not None:
                missing.append(f"whether {ambiguous_hour} is AM or PM")
            else:
                missing.append("the time")

        if details.get("duration") is None:
            missing.append("how long it should last")

        if not missing:
            return "I can help with that, but I need a bit more detail before I schedule it."

        detail_list = ", ".join(missing[:-1]) + f" and {missing[-1]}" if len(missing) > 1 else missing[0]
        context_hint = ""
        if details.get("participant"):
            context_hint = f" for {details['participant'].title()}"
        elif details.get("title"):
            context_hint = f" for {details['title']}"

        if details.get("date") and not details.get("time"):
            return f"I can help with that, but I still need {detail_list} before I schedule it{context_hint} for {details['date']}."

        if details.get("date") and details.get("participant") and not details.get("duration"):
            return f"I can help with that, but I still need {detail_list} before I schedule it{context_hint} for {details['date']}."

        return f"I can help with that, but I need to know {detail_list} before I schedule it{context_hint}."

    def _step_needs_clarification(self, step: str, context_date: str | None = None, context_time: str | None = None) -> bool:
        """Return True when one clause of a compound scheduling request lacks enough detail.

        Unlike _needs_clarification, this evaluates the clause on its own (use_history=False)
        so that a later clause such as "and a meeting with Will next Friday" is not mistaken for
        a conversational follow-up to the very request it is part of. context_date/context_time
        let a clause without its own date or time inherit one from an earlier clause in the same
        compound request, mirroring how _parse_single_calendar_instruction chains base_date and
        base_time across sequential steps when it actually creates the events.
        """
        lowered = (step or "").strip().lower()
        if not lowered:
            return False
        if not self._looks_like_scheduling_request(step):
            return False

        details = self._extract_scheduling_details(step, use_history=False)
        has_multi_participants = bool(re.search(r"\bwith\b(?:\s+both)?\s+[a-z0-9._-]+\s+and\s+[a-z0-9._-]+", lowered))
        has_topic = bool(details.get("participant") or details.get("title")) or has_multi_participants
        has_date = details.get("date") is not None or context_date is not None
        has_time = details.get("time") is not None or context_time is not None
        has_duration = details.get("duration") is not None

        if not has_topic or not has_date or not has_time:
            return True

        if not has_duration and not (details.get("title") and details.get("title").lower() not in {"meeting", "event", "an event", "a meeting"}):
            return True

        return False

    def _build_multi_step_clarification_response(self, incomplete_steps: List[tuple]) -> str:
        """Build one combined clarification message covering every incomplete meeting in a compound request."""
        clauses: List[str] = []
        for step, details in incomplete_steps:
            missing: List[str] = []
            if not details.get("participant") and not details.get("title"):
                missing.append("who it's with or what it's about")
            if not details.get("date"):
                missing.append("the date")
            if not details.get("time"):
                ambiguous_hour = self._detect_ambiguous_hour(step)
                if ambiguous_hour is not None:
                    missing.append(f"whether {ambiguous_hour} is AM or PM")
                else:
                    missing.append("the time")
            if details.get("duration") is None:
                missing.append("how long it should last")
            if not missing:
                continue

            detail_list = ", ".join(missing[:-1]) + f" and {missing[-1]}" if len(missing) > 1 else missing[0]

            if details.get("participant"):
                label = f"the meeting with {details['participant'].title()}"
            elif details.get("title"):
                label = f"the {details['title']} meeting"
            else:
                label = "one of the meetings"
            if details.get("date"):
                label = f"{label} on {details['date']}"

            clauses.append(f"{detail_list} for {label}")

        if not clauses:
            return "I can help with that, but I need a bit more detail before I schedule these."
        if len(clauses) == 1:
            return f"I can help with that, but I still need {clauses[0]}."
        return "I can help with that, but I still need " + ", and ".join(clauses) + "."

    def _needs_clarification(self, task: str) -> bool:
        """Return True when a scheduling request lacks enough details to act without guessing."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False
        if not re.search(r"\b(?:schedule|create|book|set)\b", lowered):
            return False

        if self._looks_like_reschedule_request(task):
            return False

        previous_request = self._get_previous_user_request(task)
        if self._should_use_prior_scheduling_context(task, previous_request):
            return False

        if self._looks_like_scheduling_request(task) and not self._extract_scheduling_details(task).get("participant") and not self._extract_scheduling_details(task).get("title"):
            return True

        details = self._extract_scheduling_details(task)
        has_date = details.get("date") is not None
        has_time = details.get("time") is not None
        has_duration = details.get("duration") is not None
        has_topic = bool(details.get("participant") or details.get("title"))

        if not has_topic or not has_date or not has_time:
            return True

        if not has_duration and not (details.get("title") and details.get("title").lower() not in {"meeting", "event", "an event", "a meeting"}):
            return True

        return False

    def _looks_like_reschedule_request(self, task: str) -> bool:
        """Return True when a request should update an existing meeting rather than create a new one."""
        lowered = (task or "").strip().lower()
        if not lowered:
            return False

        if self._contains_reschedule_terms(task):
            return True

        if not any(entry.get("role") == "assistant" for entry in self._conversation_history):
            return False

        return False

    def _extract_requested_time(self, task: str, default_time: str = "09:00") -> str:
        """Extract an explicit time from the request, if present."""
        parsed_time = self._parse_time_of_day(task)
        return parsed_time if parsed_time is not None else default_time

    def _extract_requested_date(self, task: str, fallback_date: str | None = None) -> str | None:
        """Extract a date from the request or fall back to the last known event date."""
        now = datetime.now()
        requested_day = self._extract_requested_day(task, now)
        if requested_day:
            return requested_day
        if fallback_date:
            return fallback_date
        return None

    def _extract_last_event_context(self) -> Dict[str, Any]:
        """Look for the most recent real created event context in the conversation history."""
        fallback_context: Dict[str, Any] = {}
        for entry in reversed(self._conversation_history):
            if entry.get("role") != "assistant":
                continue
            content = str(entry.get("content", ""))
            summary_match = re.search(r"(?:scheduled|prepared a local backup entry for|updated)\s+['\"]([^'\"]+)['\"]", content, re.IGNORECASE)
            event_id_match = re.search(r"(?:Event ID|Backup ID):\s*([A-Za-z0-9._-]+)", content, re.IGNORECASE)
            if not event_id_match:
                event_id_match = re.search(r"id:\s*([A-Za-z0-9._-]+)", content, re.IGNORECASE)
            date_match = re.search(r"for\s+(\d{4}-\d{2}-\d{2})\s+at\s+(\d{1,2}:\d{2})", content)
            event_id = event_id_match.group(1).strip() if event_id_match else None
            is_local_fallback = bool(re.search(r"local fallback", content, re.IGNORECASE)) or bool(event_id and event_id.lower().startswith("local-"))

            if summary_match or event_id:
                context = {
                    "summary": summary_match.group(1).strip() if summary_match else None,
                    "event_id": event_id,
                    "date": date_match.group(1).strip() if date_match else None,
                    "time": date_match.group(2).strip() if date_match else None,
                }
                if not is_local_fallback and context.get("event_id"):
                    return context
                if not fallback_context and (context.get("event_id") or context.get("summary")):
                    fallback_context = context

        if fallback_context:
            return fallback_context

        prior_request = ""
        for entry in reversed(self._conversation_history):
            if entry.get("role") == "user":
                prior_request = str(entry.get("content", ""))
                break
        if prior_request:
            participant = self._extract_participant_phrase(prior_request.lower())
            if participant:
                return {
                    "summary": f"Meeting with {participant.title()}",
                    "event_id": None,
                    "date": None,
                    "time": None,
                }
        return {}

    async def _reschedule_calendar_event(self, task: str) -> str:
        """Update an existing meeting's time.

        Prefers an event created or mentioned earlier in this conversation. If none exists -- for
        example, this is the first message of a new session -- falls back to searching the real
        calendar for a meeting matching who/what the request names. Unlike cancellation, this
        search does not filter by the date parsed from the request, since a reschedule request's
        date/time normally describes the NEW slot the user wants, not the meeting's current one.
        """
        context = self._extract_last_event_context()
        event_id = context.get("event_id")
        summary = context.get("summary")

        if not event_id:
            details = self._extract_scheduling_details(task, use_history=False)
            search_term = (details.get("participant") or details.get("title") or summary or "").strip().lower()
            candidates = []
            if search_term:
                events = get_events(days=60)
                candidates = [event for event in events if search_term in str(event.get("summary") or "").lower()]

            if not candidates:
                return "I don't have a calendar event to reschedule in this conversation yet."
            if len(candidates) > 1:
                options = "; ".join(f"'{c.get('summary')}' at {c.get('start')}" for c in candidates)
                return f"I found more than one matching meeting ({options}). Which one should I reschedule?"

            found_event = candidates[0]
            event_id = found_event.get("id")
            summary = found_event.get("summary") or summary
            if not event_id:
                return f"I found '{summary}' but couldn't determine its event ID to reschedule it."

        now = datetime.now()
        local_tz = now.astimezone().tzinfo or timezone.utc
        fallback_date = context.get("date")
        requested_date = self._extract_requested_date(task, fallback_date)
        requested_time = self._extract_requested_time(task, context.get("time") or "09:00")
        duration_minutes = self._extract_duration_minutes(task)
        if duration_minutes is None:
            duration_minutes = self._extract_duration_minutes(" ".join(entry.get("content", "") for entry in self._conversation_history if entry.get("role") == "user"))
        if duration_minutes is None:
            duration_minutes = 60

        if requested_date is None:
            requested_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        start_dt = datetime.strptime(f"{requested_date}T{requested_time}", "%Y-%m-%dT%H:%M").replace(tzinfo=local_tz)
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        summary = summary or "Meeting"

        updated_event_id = update_event(
            event_id=event_id,
            summary=summary,
            start_time=start_dt,
            end_time=end_dt,
        )
        if "local fallback" in str(updated_event_id).lower():
            return f"⚠️ I rescheduled '{summary}' for {requested_date} at {requested_time} and saved a local backup entry. Backup ID: {updated_event_id}"
        return f"✅ I rescheduled '{summary}' for {requested_date} at {requested_time}. Event ID: {updated_event_id}"

    async def _create_calendar_event(self, task: str) -> str:
        """Create one or more calendar events from a scheduling request."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now()
        local_tz = now.astimezone().tzinfo or timezone.utc
        default_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        default_start_time = "09:00"
        default_duration_minutes = 60

        normalized_task = (task or "").strip()
        if self._looks_like_reschedule_request(normalized_task):
            return await self._reschedule_calendar_event(normalized_task)

        contextual_task = self._build_contextual_request(normalized_task)
        steps = self._split_calendar_steps(contextual_task)
        if not steps:
            steps = [contextual_task]

        if len(steps) == 1:
            if self._needs_clarification(normalized_task):
                return self._build_clarification_response(normalized_task)
        else:
            incomplete_steps = []
            context_date = None
            context_time = None
            for step in steps:
                details = self._extract_scheduling_details(step, use_history=False)
                effective_date = details.get("date") or context_date
                effective_time = details.get("time") or context_time
                if self._step_needs_clarification(step, context_date=context_date, context_time=context_time):
                    merged_details = dict(details)
                    merged_details["date"] = effective_date
                    merged_details["time"] = effective_time
                    incomplete_steps.append((step, merged_details))
                if effective_date:
                    context_date = effective_date
                if effective_time:
                    context_time = effective_time
            if incomplete_steps:
                return self._build_multi_step_clarification_response(incomplete_steps)

        created_events = []
        context_date = None
        context_time = None
        existing_events = get_events(days=45)

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
                event_kwargs: Dict[str, Any] = {
                    "summary": event["summary"],
                    "start_time": event["start_dt"].isoformat(),
                    "end_time": event["end_dt"].isoformat(),
                }
                description = event.get("description")
                location = event.get("location")
                attendees = event.get("attendees") or []
                recurrence = event.get("recurrence")

                if description:
                    event_kwargs["description"] = description
                if location:
                    event_kwargs["location"] = location
                if attendees:
                    event_kwargs["attendees"] = attendees
                if recurrence:
                    event_kwargs["recurrence"] = recurrence

                event["conflicts"] = self._find_conflicting_events(
                    existing_events, event["start_dt"], event["end_dt"], local_tz
                )
                event_id = create_event(**event_kwargs)
                created_events.append((event, event_id))
                context_date = event["date"]
                context_time = event["start_time"]

        if not created_events:
            return "No calendar events were created."

        lines = []
        for event, event_id in created_events:
            recurrence_note = ""
            if event.get("recurrence"):
                recurrence_note = " Weekly recurring."
            location_note = f" at {event.get('location')}" if event.get("location") else ""
            conflicts = event.get("conflicts") or []
            conflict_note = ""
            if conflicts:
                conflict_summaries = ", ".join(f"'{c.get('summary') or 'an existing event'}'" for c in conflicts)
                conflict_note = f" ⚠️ Heads up, this overlaps with {conflict_summaries} already on your calendar."
            if "local fallback" in str(event_id).lower():
                lines.append(
                    f"⚠️ I prepared a local backup entry for '{event['summary']}' for {event['date']} at {event['start_time']} for {int((event['end_dt'] - event['start_dt']).total_seconds() // 60)} minutes{location_note}.{recurrence_note} Backup ID: {event_id}{conflict_note}"
                )
            else:
                lines.append(
                    f"✅ I scheduled '{event['summary']}' for {event['date']} at {event['start_time']} for {int((event['end_dt'] - event['start_dt']).total_seconds() // 60)} minutes{location_note}.{recurrence_note} Event ID: {event_id}{conflict_note}"
                )
        return "\n".join(lines)

    async def _get_calendar_events(self, task: str) -> List[Dict[str, Any]]:
        """Fetch upcoming calendar events for the requested window."""
        requested_day = self._extract_requested_day(task, datetime.now())
        events = get_events(days=7)
        if not requested_day:
            return events

        return [event for event in events if self._get_event_date(event) == requested_day]

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

    def _clean_response_text(self, text: Any) -> str:
        """Remove routing prefixes and return a clean user-facing message."""
        if text is None:
            return "How can I help you today?"

        if isinstance(text, list):
            if not text:
                return "I don't see any calendar events for that window."
            formatted_items = []
            for item in text:
                if isinstance(item, dict):
                    summary = item.get("summary") or item.get("title") or item.get("name")
                    start = item.get("start")
                    if summary and start:
                        formatted_items.append(f"{summary} at {start}")
                    elif summary:
                        formatted_items.append(str(summary))
                    else:
                        formatted_items.append(str(item))
                else:
                    formatted_items.append(str(item))
            text = "; ".join(formatted_items)
        elif isinstance(text, dict):
            summary = text.get("summary") or text.get("title") or text.get("name")
            if summary:
                text = str(summary)
            else:
                text = json.dumps(text, sort_keys=True)
        else:
            text = str(text)

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
        return "Hello! I'm your office manager and can help with calendar scheduling and ollama-backed planning. How can I help you today?"

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