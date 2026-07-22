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
from typing import Any, Awaitable, Callable, Dict, List

from backend.config import Settings
from backend.memory import LocalMemory


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
        plan = self.plan_request(request)
        context: List[Dict[str, Any]] = []
        await self._emit_status("working", "manager", "Preparing your local briefing...", 0.1)

        task_order = ["calendar", "email", "financial", "researcher"]
        for step in task_order:
            if step not in plan:
                continue
            await self._emit_status("working", step, self._step_message(step), 0.2 + 0.15 * task_order.index(step))
            if step == "calendar":
                context.append({"tool": "calendar", "result": self._calendar_context()})
            elif step == "email":
                context.append({"tool": "email", "result": self._email_context()})
            elif step == "financial":
                context.append({"tool": "financial", "result": self._financial_context()})
            elif step == "researcher":
                context.append({"tool": "research", "result": self._research_context()})

        response, model_status = await self._generate_response(request, context)
        self.memory.add_memory("assistant", f"User request: {request} | Response: {response}", metadata={"kind": "assistant"})
        await self._emit_status("idle", "manager", "Ready for the next local request.", 1.0, final_response=response)
        return {"response": response, "plan": plan, "context": context, "model_status": model_status}

    def plan_request(self, request: str) -> List[str]:
        normalized = request.lower()
        plan: List[str] = []
        if any(token in normalized for token in ["brief", "briefing", "schedule", "calendar", "meeting", "today", "morning", "habit", "focus", "travel"]):
            plan.append("calendar")
        if any(token in normalized for token in ["email", "inbox", "mail", "draft", "reply", "follow-up", "unsubscribe", "marketing"]):
            plan.append("email")
        if any(token in normalized for token in ["finance", "budget", "subscription", "transaction", "money", "expense", "bill", "spending", "rocket"]):
            plan.append("financial")
        if any(token in normalized for token in ["search", "research", "look up", "find out", "news", "web", "compare", "trip", "product", "hotel", "flight"]):
            plan.append("researcher")
        if any(token in normalized for token in ["brief", "briefing", "summarize", "compile", "priority", "conflict", "manager"]):
            plan.append("manager")
        if not plan:
            plan = ["calendar", "email"]
        if "calendar" in plan and "email" not in plan and any(token in normalized for token in ["brief", "briefing", "morning"]):
            plan.append("email")
        if "manager" not in plan:
            plan.append("manager")
        return plan

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

    def _step_message(self, step: str) -> str:
        mapping = {
            "calendar": "Checking your schedule and upcoming commitments...",
            "email": "Reviewing your inbox and drafting any needed replies...",
            "financial": "Scanning finances, subscriptions, and unusual activity...",
            "researcher": "Searching local knowledge and private web results...",
            "manager": "Gathering the best updates from every domain into one clear briefing...",
        }
        return mapping.get(step, "Thinking through the request locally...")

    def _calendar_context(self) -> Dict[str, Any]:
        return {"events": [{"summary": "No calendar provider configured yet", "start": None, "end": None}]}

    def _email_context(self) -> Dict[str, Any]:
        return {"summary": "No Gmail provider configured yet."}

    def _financial_context(self) -> Dict[str, Any]:
        return {"summary": "No finance integration configured yet."}

    def _research_context(self) -> Dict[str, Any]:
        return {"summary": "No search provider configured yet."}

    async def _generate_response(self, request: str, context: List[Dict[str, Any]]) -> tuple[str, str]:
        normalized_request = request.strip().lower()
        if normalized_request == "hi":
            return self._greeting_response(), "fallback"

        ollama = _load_ollama_client()
        if ollama is None:
            return self._fallback_response(request, context), "fallback"

        try:
            client = ollama.Client(host=self.settings.ollama_base_url)
            prompt = self._build_prompt(request, context)
            response = client.chat(
                model=self.settings.ollama_model,
                messages=[{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": prompt}],
            )
            content = response.get("message", {}).get("content", "")
            message = re.sub(r"\s+", " ", content).strip()
            if message:
                return message, "ollama"
            return self._fallback_response(request, context), "fallback"
        except Exception:
            return self._fallback_response(request, context), "fallback"

    def _greeting_response(self) -> str:
        return "I’m handling your request locally with a calendar, manager, email workflow. Ollama is not available yet, so I’m using a lightweight local fallback response."

    def _fallback_response(self, request: str, context: List[Dict[str, Any]]) -> str:
        return "Unable to reach the manager agent; this is a fallback response"

    def _build_prompt(self, request: str, context: List[Dict[str, Any]]) -> str:
        formatted_context = json.dumps(context, indent=2)
        return f"User request: {request}\n\nContext from local tools:\n{formatted_context}"

    def _system_prompt(self) -> str:
        return (
            "You are Agent Suite, a warm and concise local assistant. "
            "You help with calendar, email, finance, and research tasks while keeping everything local. "
            "Never invent information. If you are unsure, say so clearly. "
            "When drafting an email, clearly label it as a draft and never imply it was sent. "
            "Call out conflicts or unusual patterns when they are apparent."
        )
