# Transub Desktop

The desktop app is an Electron + React + TypeScript shell that communicates
with the Python backend over HTTP. It is the primary GUI in this repository.

## Architecture

```
Electron (React frontend)
    │
    │ HTTP localhost:18789, with automatic fallback ports
    ▼
Python FastAPI server (transub/server.py)
    │
    ▼
faster-whisper / LLM / subtitle pipeline
```

The Electron main process starts the Python server as a background process on
app launch and stops it on quit. The frontend calls REST endpoints and listens
to SSE events for real-time progress. The app chooses port `18789` when it is
available and falls back through nearby ports if another Transub server is
already running.

The run screen should stay user-facing: it shows stage status, output location,
and a saved debug log path. Detailed backend messages are written to
`~/.cache/transub/logs/*.log` instead of being shown as a black terminal panel.

## Run During Development

```bash
cd desktop
npm install
npm run electron:dev
```

This starts both the Vite dev server (React) and the Electron main process.
The main process automatically starts the Python backend via `uv run transub
serve`. If you want to run the Python server separately:

```bash
# Terminal 1: Python backend
uv sync --extra server
uv run transub serve

# Terminal 2: Electron frontend
cd desktop
npm run electron:dev
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/status` | Current pipeline state |
| GET | `/api/config` | Read config |
| PUT | `/api/config` | Update config |
| GET | `/api/auth` | List providers (no keys exposed) |
| PUT | `/api/auth/{provider}` | Save provider credentials |
| DELETE | `/api/auth/{provider}` | Clear provider credentials |
| POST | `/api/prepare-model` | Download/initialize ASR model |
| POST | `/api/run` | Start pipeline |
| POST | `/api/cancel` | Cancel running pipeline |
| GET | `/api/stream` | SSE progress events |
| GET | `/api/cache/stats` | Cache statistics |
| DELETE | `/api/cache` | Clear API cache |

## Build

```bash
cd desktop
npm run electron:build
```

## Credentials

Provider credentials are managed by the Python backend:

- auth file: `~/.transub/auth.toml`
- override path: `TRANSUB_AUTH`
- environment variables win over auth-file values
- never commit auth files or print API keys in logs

## Debug Logs

Each desktop run creates a local log file under `~/.cache/transub/logs/`.
The frontend can reveal that file in Finder after the run starts. Keep the main
UI oriented around task status; use the saved log only for troubleshooting.
