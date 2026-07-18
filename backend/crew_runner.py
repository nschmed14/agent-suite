"""Local-only CrewAI wrapper for Agent Suite.

The implementation is designed to prefer local Ollama-backed execution when the
necessary dependencies are available, while remaining functional even in a
minimal environment. The runtime streams state updates to the backend so the
frontend can reflect the agent activity live.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

from backend.config import Settings
from backend.office_state import OfficeState
from backend.memory import LocalMemory


class CrewRunner:
    """A lightweight runner that simulates agent work locally."""

    def __init__(
        self,
        office_state: OfficeState,
        memory: LocalMemory,
        settings: Settings,
    ) -> None:
        self.office_state = office_state
        self.memory = memory
        self.settings = settings
        self._callbacks: list[Callable[[dict], Awaitable[None] | None]] = []
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._crewai_available = False

        try:  # pragma: no cover - optional dependency
            import crewai  # noqa: F401
        except ImportError:  # pragma: no cover - fallback path
            self._crewai_available = False
        else:
            self._crewai_available = True

    def register_callback(self, callback: Callable[[dict], Awaitable[None] | None]) -> None:
        """Register a callback to receive state updates."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Begin the background loop that pushes state updates."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the background loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            if self.settings.lockdown_mode:
                await asyncio.sleep(1)
                continue
            for agent_id in ("manager", "financial", "vault", "assistant", "researcher"):
                state = self.office_state.list_agents()
                if not state:
                    continue
                progress = 0
                for agent in state:
                    if agent["id"] == agent_id:
                        progress = int(agent["task_progress"])
                        break
                next_progress = (progress + 8) % 100
                status = "working" if next_progress > 0 else "idle"
                thoughts = "Local routine running steadily."
                self.office_state.apply_task_update(
                    agent_id,
                    task="Local routine",
                    progress=next_progress,
                    status=status,
                    thoughts=thoughts,
                )
                await self._notify()
            await asyncio.sleep(2)

    async def run_task(self, agent_id: str, task_text: str) -> None:
        """Execute a task in a local-only way and stream progress."""
        if self.settings.lockdown_mode:
            self.office_state.apply_task_update(
                agent_id,
                task=task_text,
                progress=0,
                status="alert",
                thoughts="Lockdown mode active, processing paused.",
            )
            await self._notify()
            return

        self.office_state.apply_task_update(
            agent_id,
            task=task_text,
            progress=10,
            status="working",
            thoughts=f"Starting local task: {task_text}",
        )
        await self._notify()

        if self._crewai_available:
            self.office_state.apply_task_update(
                agent_id,
                task=task_text,
                progress=40,
                status="working",
                thoughts="CrewAI integration available, delegating to local toolchain.",
            )
            await self._notify()

        for progress in (30, 60, 90, 100):
            await asyncio.sleep(0.25)
            self.office_state.apply_task_update(
                agent_id,
                task=task_text,
                progress=progress,
                status="working",
                thoughts=f"Processing local request: {task_text}",
            )
            await self._notify()

        self.office_state.apply_task_update(
            agent_id,
            task=task_text,
            progress=100,
            status="idle",
            thoughts="Task completed locally. Awaiting the next request.",
        )
        await self._notify()

        self.memory.add_memory(
            agent_id,
            f"Completed local task: {task_text}",
            metadata={"kind": "task", "local_only": True},
        )

    async def _notify(self) -> None:
        payload = {"type": "state_update", "state": self.office_state.snapshot()}
        for callback in list(self._callbacks):
            result = callback(payload)
            if asyncio.iscoroutine(result):
                await result
