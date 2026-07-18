# Agent Suite

[![Zero Cloud](https://img.shields.io/badge/Zero%20Cloud-Local%20Only-success)](https://github.com/nschmed14/agent-suite)

Privacy-first, fully local AI agent suite for an office simulation.

> No data ever leaves your machine. No telemetry. No analytics. No accounts.

## Architecture

```text
+------------------+      +----------------------+      +----------------------+
| Phaser Frontend  | <--> | FastAPI WebSocket    | <--> | CrewAI / Ollama      |
| (localhost)      |      | (localhost:8000)    |      | (localhost:11434)    |
+------------------+      +----------------------+      +----------------------+
         |                           |                                |
         |                           |                                |
         v                           v                                v
+------------------+      +----------------------+      +----------------------+
| Local UI / Room |      | SQLite + ChromaDB   |      | Local-only tools     |
| assets, audio   |      | private memory      |      | file/sql/calc/...    |
+------------------+      +----------------------+      +----------------------+
```

No external network calls are made by the backend or agents. Tailscale is optional for encrypted remote access only.

## Hardware Requirements

| Component | Minimum |
| --- | --- |
| CPU | 4 core modern CPU |
| RAM | 8 GB |
| Storage | 20 GB free SSD |
| GPU | Optional, not required |

## Quick Start

1. Clone the repository.
2. Copy .env.example to .env and review the defaults.
3. Run docker compose up.
4. Open http://localhost:8080 in your browser.

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/server.py
```

Then open http://localhost:8080 for the frontend.

## Privacy Notes

- Telemetry is disabled by default.
- CORS is restricted to localhost and the Tailscale network range.
- LOCKDOWN_MODE pauses all agent processing and closes connections.
- All memory is stored locally in SQLite and ChromaDB.
