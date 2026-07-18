"""Configuration helpers for the local-only Agent Suite backend.

The settings are intentionally explicit and auditable. All telemetry and network
behaviour is disabled by default unless the operator opts in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Parse a boolean-like environment value."""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Application configuration used by the backend and agents."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    tailscale_enabled: bool = _as_bool(os.getenv("TAILSCALE_ENABLED"), False)
    tailscale_network: str = os.getenv("TAILSCALE_NETWORK", "100.64.0.0/10")
    lockdown_mode: bool = _as_bool(os.getenv("LOCKDOWN_MODE"), False)
    disable_telemetry: bool = _as_bool(os.getenv("DISABLE_TELEMETRY"), True)
    data_dir: str = os.getenv("DATA_DIR", "./data")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/agent_suite.db")
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    cors_origins: Tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000,http://localhost:8080,http://127.0.0.1:8080",
        ).split(",")
        if item.strip()
    )
    ws_host: str = os.getenv("WS_HOST", "0.0.0.0")
    ws_port: int = int(os.getenv("WS_PORT", "8000"))


def get_settings() -> Settings:
    """Return the resolved settings object."""
    return Settings()
