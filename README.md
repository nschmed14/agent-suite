# Agent Suite

[![Zero Cloud](https://img.shields.io/badge/Zero%20Cloud-Local%20Only-success)](https://github.com/nschmed14/agent-suite)

Privacy-first, fully local AI agent suite for an office simulation.

> No data ever leaves your machine. No telemetry. No analytics. No accounts.

## Architecture

```text
+------------------+      +----------------------+      +----------------------+
| Phaser Frontend  | <--> | FastAPI WebSocket    | <--> | Single local brain   |
| (localhost)      |      | (localhost:8000)    |      | (Ollama / local)     |
+------------------+      +----------------------+      +----------------------+
         |                           |                                |
         |                           |                                |
         v                           v                                v
+------------------+      +----------------------+      +----------------------+
| Office scene    |      | SQLite + ChromaDB   |      | Calendar / Email /   |
| and character   |      | private memory      |      | Finance / Search     |
| status UI       |      |                     |      | tools                |
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
5. Try a local request such as “Morning briefing” or “Draft an email reply”.

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
