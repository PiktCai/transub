# Transub Desktop

The desktop app is an Electron + React + TypeScript shell for the Python
pipeline. It is the primary GUI in this repository.

## Run During Development

```bash
cd desktop
npm install
npm run electron:dev
```

Use `npm run electron:dev` when testing real functionality. It starts the
Electron main process, preload bridge, renderer dev server, and native file
dialogs. `npm run dev` is only a browser preview for layout work and cannot
open system file pickers.

## Build

```bash
cd desktop
npm run electron:build
```

## First-Run Local Model Setup

The GUI exposes a **Prepare local model** action on the Run page. It calls:

```bash
uv run transub prepare-model
```

This initializes the selected faster-whisper model before the user runs a full
job. The first run may download model files; later runs use the local cache.
The command is safe to run before any video is selected.

## Backend Bridge

Electron calls the Python CLI with argument arrays, not shell strings. This is
important: video paths may contain spaces, CJK characters, brackets, and other
characters that should remain a single argument.

The main process runs backend commands from the repository root so `uv run
transub ...` can resolve the local project.

## Credentials

Provider credentials are managed by the Python backend:

- auth file: `~/.transub/auth.toml`
- override path: `TRANSUB_AUTH`
- environment variables win over auth-file values
- never commit auth files or print API keys in logs
