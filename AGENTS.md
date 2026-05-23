# PROJECT KNOWLEDGE BASE

**Updated:** 2026-05-23
**Branch:** main

## OVERVIEW
Transub is archived as of 2026-05-23. This repository is the preserved Python/Electron implementation of a subtitle pipeline that extracts audio (ffmpeg), transcribes speech (faster-whisper), and translates subtitles (LLM) into SRT/VTT. Active experimentation has moved to the agent-native `transub` skill at https://github.com/PiktCai/skills/tree/main/transub.

## BRANCH STRUCTURE
- **`main`** — Archived/maintenance-only desktop-first implementation. Prefer documentation fixes or small preservation patches over new product features.
- **`legacy`** — Frozen at v0.2.2 (`633d7f8`, 2025-11-18). Original CLI-only tool with multiple ASR backends. Tagged as `v0.2.2-legacy`.

## STRUCTURE
```
transub/
├── transub/          # Core Python package (cli, config, transcribe, translate, subtitles)
├── desktop/          # Electron + React + TypeScript desktop frontend
├── docs/             # Project documentation (desktop and packaging guides)
├── .github/          # CI/CD for PyPI publishing
├── pyproject.toml    # Build config, dependencies
└── AGENTS.md         # This file
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CLI Commands & Wizard | `transub/cli.py` | Monolith (~1750 lines); contains all interactive UI |
| HTTP Backend | `transub/server.py` | FastAPI server for desktop app (localhost, dynamic fallback port) |
| Desktop Frontend | `desktop/` | Electron + React + TypeScript |
| Documentation Index | `docs/README.md` | Human-facing docs map |
| Packaging / Release | `docs/packaging.md` | PyPI and Homebrew release workflow |
| Configuration | `transub/config.py` | Pydantic v2 models for Whisper, LLM, Pipeline |
| Transcription | `transub/transcribe.py` | Single ASR path: faster-whisper with word timestamps |
| Translation | `transub/translate.py` | LLM translation with context-aware, glossary support, and agent loop pattern |
| Segmentation | `transub/segmentation.py` | NLP-based intelligent subtitle splitting using spaCy |
| LLM Optimization | `transub/optimize.py` | ASR error correction and translation polishing |
| Free Translation | `transub/free_translate.py` | Cost-free Bing/Google translation backends |
| Batch Processing | `transub/batch.py` | CSV/JSON-based batch task processing |
| Disk Caching | `transub/cache.py` | API response caching for cost savings |
| Auth/Credentials | `transub/auth.py` | Provider-scoped API keys in `~/.transub/auth.toml` |
| Subtitle Processing | `transub/subtitles.py` | SRT/VTT parsing, refinement, word-timing |
| Desktop GUI | `desktop/`, `docs/desktop.md` | Electron + React desktop frontend |
| Agent Skill Successor | https://github.com/PiktCai/skills/tree/main/transub | Preferred direction for future subtitle workflow work |
| Tests | `transub/test_*.py` | **Co-located** in package (not `tests/`); run via `unittest` |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `app` | `typer.Typer` | `cli.py` | CLI entry point |
| `TransubConfig` | `BaseModel` | `config.py` | Top-level config validation |
| `SubtitleDocument` | `class` | `subtitles.py` | Core subtitle model & processing |
| `LLMTranslator` | `class` | `translate.py` | Translation engine |
| `transcribe_audio` | `func` | `transcribe.py` | Whisper abstraction |
| `PipelineState` | `class` | `state.py` | Run state persistence |
| `AuthManager` | `class` | `auth.py` | Provider credentials file manager |

## CONVENTIONS
- **Python 3.10+**: Uses `from __future__ import annotations` everywhere.
- **Typing**: Strict type hints; Pydantic for config validation.
- **Logging**: Use `transub.logger.setup_logging` for file logging; `rich.console.Console` for terminal UI.
- **UI**: All terminal output via Rich (`Panel`, `Table`, `Progress`).
- **Tests**: `unittest.TestCase`; `test_<module>.py` naming.
- **Performance tests**: Demo/performance tests are opt-in with `TRANSUB_RUN_PERF_TESTS=1` so default discovery stays fast and quiet.
- **Git checkpoints**: For GUI work or broad multi-file changes, make a small verified checkpoint commit before handing off. Prefer commits that can be reverted cleanly over one giant uncommitted working tree.
- **Desktop run UI**: Keep detailed command/backend logs out of the main run screen. Show stage status and saved file paths; write debug logs to `~/.cache/transub/logs/`.
- **Provider secrets UI**: Treat provider keys like a secret manager. Never echo saved keys into inputs; clear key drafts when switching providers; use an explicit action to set the active translation provider.

## ANTI-PATTERNS (THIS PROJECT)
- **Do NOT resume feature development here by default**: this repository is archived; move new workflow ideas to the `transub` skill unless explicitly asked to revive the app.
- **Do NOT split `cli.py` yet**: It is intentionally monolithic for now.
- **Do NOT run tests via pytest**: Use `python -m unittest`.
- **Do NOT commit `transub.conf`**: User-specific config; use `transub-sample.conf` as reference.
- **Do NOT commit auth files**: `~/.transub/auth.toml` contains provider API keys and must stay outside the repo.
- **Do NOT commit generated desktop artifacts**: keep `desktop/node_modules/`, `desktop/dist/`, `desktop/out/`, `desktop/build/`, `desktop/release/`, and `desktop/resources/transub-server` ignored.
- **Do NOT add ASR backend selectors**: transcription is intentionally faster-whisper only.

## UNIQUE STYLES
- **Co-located tests**: Tests live inside `transub/`, not `tests/`.
- **Smart Retry**: Custom circuit-breaker logic in `smart_retry.py` for LLM robustness.
- **Intelligent Segmentation**: NLP-based subtitle splitting using spaCy dependency parsing.
- **Context-Aware Translation**: LLM uses previous/next subtitle context for coherent translations.
- **Glossary Support**: Import custom terminology (JSON/CSV) for consistent translations.
- **LLM Optimization**: ASR error correction and translation polishing using agent loop pattern.
- **Free Translators**: Cost-free Bing/Google translation backends for budget-conscious users.
- **Provider Credentials**: LLM keys are provider-scoped. Environment variables still win, then `~/.transub/auth.toml`.
- **GUI Direction**: The desktop frontend is Electron/React with a native, tool-oriented Flexoki UI.
- **Desktop Icon Direction**: Tracked icon assets live in `desktop/assets/`; generated build/package outputs stay ignored.
- **Run Feedback Direction**: The run page should feel like a native task monitor, not a terminal emulator. Keep detailed logs saved and revealable from Finder.
- **Resume Direction**: Failed desktop runs should preserve selected video, stage, error, and log path when leaving the Run tab. Retrying should reuse cached audio/transcription/translation state where the backend supports it.
- **ASR Direction**: Be opinionated. Do not re-add API, whisper.cpp, MLX, SenseVoice, Qwen3-ASR, or openai-whisper backend selection unless explicitly requested.

## COMMANDS
```bash
transub init                    # Setup wizard
transub run video.mp4           # Full pipeline
transub run video.mp4 -T        # Transcribe only
transub run video.mp4 --free bing  # Use free Bing translator
transub batch tasks.csv         # Batch processing
transub serve                   # Start HTTP backend for desktop app
cd desktop && npm run electron:dev  # Launch real Electron desktop app
cd desktop && npm run package       # Build bundled Python server and package desktop app
transub prepare-model           # Pre-download/initialize configured local ASR model
python -m unittest discover     # Run all tests
TRANSUB_RUN_PERF_TESTS=1 python -m unittest transub.test_concurrent_performance transub.test_retry_performance transub.test_concurrent_demo
```

## NOTES
- For the archive rationale and future direction, read `docs/archive.md`.
- `cli.py` is the primary complexity hotspot; read it carefully for pipeline flow.
- Before changing desktop GUI work, read `docs/desktop.md`.
- Before changing release packaging, read `docs/packaging.md`.
- `aiohttp` is used for concurrent translation but is an optional dependency (`uv sync --extra async`).
- `spacy` is used for intelligent segmentation but is an optional dependency (`uv sync --extra nlp`).
- `faster-whisper` is a core dependency, not an optional backend. Run `uv sync`, then `uv run transub prepare-model` for first-use setup.
- `fastapi` + `uvicorn` power the HTTP backend for the desktop app: `uv sync --extra server`.
- `pyproject.toml` defines optional dependencies: `[async]`, `[nlp]`, `[server]`, and `[dev]`.
- The Electron desktop app lives in `desktop/`; run `cd desktop && npm run electron:dev` to start the real app. `npm run dev` is browser-only and cannot test native file pickers.
