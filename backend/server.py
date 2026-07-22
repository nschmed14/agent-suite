"""FastAPI server for the local Agent Suite.

The service exposes a WebSocket endpoint that streams office state updates to the
frontend. It also exposes basic health and task endpoints so the project can be
used locally without any cloud service.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.assistant import Assistant
from backend.config import get_settings
from backend.crew_runner import CrewRunner
from backend.memory import LocalMemory
from backend.office_state import OfficeState

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the backend services when the server starts."""
    os.makedirs(settings.data_dir, exist_ok=True)
    app.state.office_state = OfficeState()
    app.state.memory = LocalMemory(settings.sqlite_path, settings.chroma_path)
    app.state.crew_runner = CrewRunner(app.state.office_state, app.state.memory, settings)
    app.state.assistant = Assistant(app.state.memory, settings)
    app.state.connections: List[WebSocket] = []
    app.state.assistant.register_callback(lambda payload: manager.broadcast(payload))
    await app.state.crew_runner.start()
    yield
    await app.state.crew_runner.stop()


app = FastAPI(title="Agent Suite", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def _is_allowed_origin(origin: str) -> bool:
    """Allow localhost origins and the optional Tailscale IP range only."""
    if not origin:
        return False
    try:
        parsed = origin.split("//", 1)[1].split("/", 1)[0].split(":", 1)[0]
    except IndexError:
        return False

    if parsed in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return True

    try:
        parsed_ip = ip_address(parsed)
    except ValueError:
        return False

    try:
        return parsed_ip in ip_network(settings.tailscale_network)
    except ValueError:
        return False


@app.middleware("http")
async def restrict_cors(request: Request, call_next):
    """Reject non-local origins while preserving a local-only experience."""
    origin = request.headers.get("origin")
    response = None
    if origin and _is_allowed_origin(origin):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    if origin:
        response = await call_next(request)
        response.status_code = 403
        return response
    return await call_next(request)


class ConnectionManager:
    """Track connected WebSocket clients."""

    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(message)
        for connection in list(self.active):
            try:
                await connection.send_text(payload)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Return a concise health payload."""
    return {
        "status": "ok",
        "telemetry_disabled": settings.disable_telemetry,
        "lockdown_mode": settings.lockdown_mode,
        "ollama_base_url": settings.ollama_base_url,
    }


@app.post("/task/{agent_id}")
async def submit_task(agent_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a local task to a named agent and stream progress."""
    if settings.lockdown_mode:
        raise HTTPException(status_code=423, detail="Lockdown mode active")

    task_text = str(payload.get("task", ""))
    if not task_text:
        raise HTTPException(status_code=400, detail="Task is required")

    runner = app.state.crew_runner
    asyncio.create_task(runner.run_task(agent_id, task_text))
    return {"status": "accepted", "agent_id": agent_id, "task": task_text}


@app.post("/assistant")
async def submit_assistant_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the single-brain assistant for a user request and stream status updates."""
    if settings.lockdown_mode:
        raise HTTPException(status_code=423, detail="Lockdown mode active")

    request_text = str(payload.get("request", "")).strip()
    if not request_text:
        raise HTTPException(status_code=400, detail="Request is required")

    asyncio.create_task(app.state.assistant.run(request_text))
    return {"status": "accepted", "request": request_text}


@app.websocket("/office")
async def office_socket(websocket: WebSocket) -> None:
    """Stream office state updates to frontend clients."""
    await manager.connect(websocket)
    try:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "state_update",
                    "state": app.state.office_state.snapshot(),
                }
            )
        )

        while True:
            if settings.lockdown_mode:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "status",
                            "message": "Lockdown mode active; processing paused.",
                        }
                    )
                )
                await asyncio.sleep(2)
                continue
            data = await websocket.receive_text()
            if data.strip():
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@app.on_event("startup")
async def register_runner_callback() -> None:
    """Connect the runner to the broadcast function."""
    async def push_state(payload: Dict[str, Any]) -> None:
        await manager.broadcast(payload)

    app.state.crew_runner.register_callback(push_state)


if __name__ == "__main__":
    uvicorn.run("backend.server:app", host=settings.ws_host, port=settings.ws_port, reload=False)
