"""Agent definitions for the local-only Agent Suite.

Each agent is intentionally configured with a local Ollama connection. CrewAI is
used where available, but the environment is designed to remain functional even
if the framework is not installed in the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AgentDefinition:
    """A simple descriptor for a local office agent."""

    agent_id: str
    name: str
    role: str
    goal: str
    backstory: str
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"


def build_agent_definitions(base_url: str = "http://localhost:11434") -> List[AgentDefinition]:
    """Create the five local agents used by the project."""
    return [
        AgentDefinition(
            agent_id="manager",
            name="Manager",
            role="Coordinator",
            goal="Coordinate local office operations and keep the room calm.",
            backstory="A discreet coordinator who keeps the local desk network aligned.",
            base_url=base_url,
        ),
        AgentDefinition(
            agent_id="financial",
            name="Financial",
            role="Budget Specialist",
            goal="Track local expenses and summarize task outcomes.",
            backstory="A careful operator who values clear records and private planning.",
            base_url=base_url,
        ),
        AgentDefinition(
            agent_id="vault",
            name="Work Vault",
            role="Security Steward",
            goal="Protect local work artifacts with minimal exposure.",
            backstory="A careful archivist who preserves local knowledge ethically.",
            base_url=base_url,
        ),
        AgentDefinition(
            agent_id="assistant",
            name="Personal Assistant",
            role="Support Agent",
            goal="Handle direct requests and keep the office responsive.",
            backstory="A warm helper who favors local execution and transparent actions.",
            base_url=base_url,
        ),
        AgentDefinition(
            agent_id="researcher",
            name="Researcher",
            role="Knowledge Worker",
            goal="Gather and summarize information from local sources.",
            backstory="A curious investigator who never depends on external services.",
            base_url=base_url,
        ),
    ]


def as_dict(definitions: List[AgentDefinition]) -> List[Dict[str, str]]:
    """Convert definitions to a serializable form."""
    return [
        {
            "id": definition.agent_id,
            "name": definition.name,
            "role": definition.role,
            "goal": definition.goal,
            "backstory": definition.backstory,
            "model": definition.model,
            "base_url": definition.base_url,
        }
        for definition in definitions
    ]
