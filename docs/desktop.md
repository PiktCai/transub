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
During transcription, the backend emits progress whenever faster-whisper yields
a segment, using segment timestamps against the audio duration.

Run state is also persisted in the renderer's local storage. If a run fails,
leaving and returning to the Run tab should preserve the failed state, selected
video, debug log path, and the ability to run again after configuration changes.

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

For distributable builds, run the package script:

```bash
cd desktop
npm run package
```

`npm run package` runs `build:server` automatically before electron-builder.
The server binary is generated at `desktop/resources/transub-server` and is
intentionally ignored by Git because it is large and platform-specific. Use
`npm run build:server` directly only when validating the bundled backend in
isolation. The app icon sources live in `desktop/assets/`; electron-builder uses
`desktop/assets/icon.icns`, `desktop/assets/icon.ico`, and
`desktop/assets/icon.png`.

## Credentials

Provider credentials are managed by the Python backend:

- auth file: `~/.transub/auth.toml`
- override path: `TRANSUB_AUTH`
- environment variables win over auth-file values
- never commit auth files or print API keys in logs

The Providers page follows a secret-manager style flow:

- saved API keys are never echoed back into the password field;
- switching providers clears the key input so secrets do not appear under the wrong provider;
- saving a provider stores credentials only;
- selecting the provider for actual pipeline translation is a separate `Use for translation` action that writes the active LLM provider/model/base into `transub.conf`.

## Debug Logs

Each desktop run creates a local log file under `~/.cache/transub/logs/`.
The frontend can reveal that file in Finder after the run starts. Keep the main
UI oriented around task status; use the saved log only for troubleshooting.
