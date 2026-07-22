"""Office state management for the local Agent Suite.

This module keeps the agent state that is streamed to the frontend over WebSocket.
The schema is intentionally simple so that it is easy to audit and inspect.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


class OfficeState:
    """Manage the in-memory representation of the office agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, Dict[str, object]] = self._initial_agents()

    def _initial_agents(self) -> Dict[str, Dict[str, object]]:
        return {
            "manager": {
                "id": "manager",
                "name": "Manager",
                "position": {"x": 0, "y": 0},
                "status": "idle",
                "thoughts": "Preparing the local workspace.",
                "task_progress": 0,
                "task": "",
                "desk": 1,
            },
            "financial": {
                "id": "financial",
                "name": "Finance",
                "position": {"x": 1, "y": 0},
                "status": "idle",
                "thoughts": "Reviewing local ledger updates.",
                "task_progress": 0,
                "task": "",
                "desk": 2,
            },
            "calendar": {
                "id": "calendar",
                "name": "Calendar",
                "position": {"x": 2, "y": 0},
                "status": "idle",
                "thoughts": "Waiting for the next local schedule update.",
                "task_progress": 0,
                "task": "",
                "desk": 3,
            },
            "email": {
                "id": "email",
                "name": "Email",
                "position": {"x": 3, "y": 0},
                "status": "idle",
                "thoughts": "Listening for inbox triage requests.",
                "task_progress": 0,
                "task": "",
                "desk": 4,
            },
            "researcher": {
                "id": "researcher",
                "name": "Research",
                "position": {"x": 4, "y": 0},
                "status": "idle",
                "thoughts": "Scanning local knowledge quietly.",
                "task_progress": 0,
                "task": "",
                "desk": 5,
            },
        }

    def snapshot(self) -> Dict[str, object]:
        """Return a copy of the state that is safe to stream."""
        return {"agents": [deepcopy(agent) for agent in self._agents.values()]}

    def update_agent(self, agent_id: str, **updates: object) -> Dict[str, object]:
        """Update a single agent with validated fields."""
        agent = self._agents.get(agent_id)
        if agent is None:
            raise KeyError(f"Unknown agent id: {agent_id}")
        agent.update(updates)
        return deepcopy(agent)

    def apply_task_update(
        self,
        agent_id: str,
        task: str,
        progress: int,
        status: str,
        thoughts: str,
    ) -> Dict[str, object]:
        """Apply task progress updates and keep the state coherent."""
        agent = self._agents[agent_id]
        desk_index = int(agent.get("desk", 1)) - 1
        progress_value = max(0, min(100, progress))
        offset = progress_value / 100.0
        position = {
            "x": desk_index * 0.9 + offset * 0.2,
            "y": 0.0 + offset * 0.1 if status == "working" else 0.0,
        }
        if status == "sleeping":
            position = {"x": desk_index * 0.9 + 0.05, "y": 0.0}
        if status == "alert":
            position = {"x": desk_index * 0.9 + 0.15, "y": 0.02}

        payload = {
            "task": task,
            "task_progress": progress_value,
            "status": status,
            "thoughts": thoughts,
            "position": position,
        }
        return self.update_agent(agent_id, **payload)

    def list_agents(self) -> List[Dict[str, object]]:
        """Return the current list of agents."""
        return [deepcopy(agent) for agent in self._agents.values()]

    def reset(self) -> None:
        """Reset the office state to the default layout."""
        self._agents = self._initial_agents()
