# Transub

[中文说明](https://github.com/PiktCai/transub/blob/main/README.zh-CN.md)

Turn any **video** into ready-to-share subtitles. Transub extracts audio with `ffmpeg`, runs Whisper to transcribe the speech track, and hands the text to an LLM so you get well-translated subtitles without leaving the terminal.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Install Transub](#2-install-transub)
  - [3. Install a Whisper Backend](#3-install-a-whisper-backend)
  - [4. Configure Transub](#4-configure-transub)
  - [5. Run the Pipeline](#5-run-the-pipeline)
- [Configuration Overview](#configuration-overview)
- [CLI Cheatsheet](#cli-cheatsheet)
- [Development](#development)
- [Project Layout](#project-layout)
- [License](#license)

## Overview

Transub orchestrates a reproducible pipeline:

1. Extract audio from a video with `ffmpeg`.
2. Transcribe speech via Whisper (local, mlx, whisper.cpp, or API).
3. Translate subtitle batches with JSON-constrained prompts.
4. Emit `.srt` or `.vtt` files with tuned line breaks and timing.

Intermediate state is cached so interrupted runs can resume without repeating earlier steps.

## Key Features

- **End-to-end pipeline** — `transub run <video.mp4>` handles extraction → transcription → translation → export.
- **Desktop GUI** — Electron + React app in `desktop/` for visual configuration and monitoring.
- **Multiple transcription backends** — choose local Whisper, `mlx-whisper`, `whisper.cpp`, OpenAI-compatible APIs, or new backends (faster-whisper, SenseVoice, Qwen3-ASR).
- **Intelligent segmentation** — NLP-based subtitle splitting using dependency parsing for natural sentence boundaries.
- **LLM subtitle optimization** — ASR error correction and translation polishing using LLM agent loop.
- **Free translation backends** — cost-free translation using Bing/Google APIs (`--free bing` or `--free google`).
- **Batch processing** — process multiple videos from CSV/JSON task files (`transub batch tasks.csv`).
- **Disk caching** — API responses are cached to save costs and improve performance.
- **Reliable translations** — JSON-constrained prompts, retry logic, and configurable batch sizes.
- **Context-aware translation** — LLM uses previous/next subtitle context for coherent translations.
- **Glossary support** — import custom terminology (JSON/CSV) for consistent translations.
- **Subtitle polishing** — punctuation-aware line splitting, timing offsets, and optional spacing tweaks when different scripts appear in the same line.
- **Stateful execution** — cached progress in the work directory (defaults to `~/.cache/transub`) avoids rework across runs.

## Installation

### 1. Prerequisites

- **Python 3.10+**
- **ffmpeg**: Must be installed and available in your system's `PATH`.
  - **Windows:** `winget install Gyan.FFmpeg` or `choco install ffmpeg`
  - **macOS:** `brew install ffmpeg`
  - **Linux:** `sudo apt update && sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo pacman -S ffmpeg` (Arch)

### 2. Install Transub

Using `uv` (Recommended)

`uv` is a fast Python package installer and resolver. It installs CLI tools in isolated environments.

```bash
uv tool install transub
```

To update later, run:

```bash
uv tool upgrade transub
```

### 3. Install a Whisper Backend (Optional)

`transub` supports multiple transcription backends. Choose one based on your needs:

- **Cloud API (Recommended for quick start):**
  - Uses OpenAI's Whisper API or compatible endpoints
  - No local installation required
  - Set `OPENAI_API_KEY` environment variable
  - Configure with `backend = "api"` during setup

- **faster-whisper (Recommended for local use):**
  - 4x faster than standard Whisper with same accuracy
  - Supports word-level timestamps
  - CPU support with INT8 quantization
  ```bash
  uv sync --extra faster-whisper
  ```

- **SenseVoice (Best for Chinese/Japanese/Korean):**
  - Extremely fast (15x faster than Whisper)
  - Optimized for Chinese, English, Cantonese, Japanese, Korean
  - No word-level timestamps (segment-level only)
  ```bash
  uv sync --extra funasr
  ```

- **Qwen3-ASR (Best multilingual coverage):**
  - 52 languages and dialects
  - Apache 2.0 license
  - Requires GPU for official package
  ```bash
  uv sync --extra qwen-asr
  ```

- **Local backends (for offline use or custom models):**
  - **For most users (local, CPU/GPU):**
    ```bash
    uv add openai-whisper
    ```
  
  - **For Apple Silicon (macOS):**
    ```bash
    uv add mlx-whisper
    ```
  
  - **For `whisper.cpp`:**
    Follow the [whisper.cpp installation instructions](https://github.com/ggerganov/whisper.cpp) to build the `main` executable and make it available on your `PATH`.

### 4. Configure Transub

Run the interactive setup wizard to create your configuration file.

```bash
transub init
```

The wizard will guide you through selecting the backend, model, and LLM provider for translation.

**Note on API Keys:** If you use OpenAI for both transcription (Whisper API) and translation (GPT models), they share the same `OPENAI_API_KEY` by default. If you need separate keys for different services, you can customize `api_key_env` in the config file for each service.

### 5. Run the Pipeline

```bash
transub run /path/to/your/video.mp4
```

Subtitles are written alongside the source video unless you set `pipeline.output_dir` in your config. Override the cache location with `--work-dir` when you need an alternate workspace.

For first-time local transcription, prepare the model before running a long job:

```bash
transub prepare-model
```

This downloads or initializes the configured local ASR model and reuses the cache on later runs.

## Configuration Overview

Runtime settings live in `transub.conf` (TOML). Key sections:

- `[whisper]` — backend selection, model name, device overrides, and extra arguments.
- `[llm]` — translation provider/model, temperature, batch size, retry policy, and context window size.
- `[pipeline]` — output format, line-length targets, timing trim/offset, punctuation and spacing options, and glossary path.

Example:

```toml
[llm]
context_window = 2  # Number of previous/next lines for context

[pipeline]
output_format = "srt"
translation_max_chars_per_line = 26
translation_min_chars_per_line = 16
normalize_cjk_spacing = true
timing_offset_seconds = 0.05
glossary_path = "glossary.json"  # Optional: path to glossary file
```

Run `transub configure` for an interactive editor, or update the file manually. Configuration files are user-specific and should not be committed.

## CLI Cheatsheet

```bash
transub run demo.mp4 --config ~/transub.conf --work-dir /tmp/transub  # override work dir (defaults to ~/.cache/transub)
transub run demo.mp4 --free bing              # use free Bing translator instead of LLM
transub run demo.mp4 --free google            # use free Google translator instead of LLM
transub batch tasks.csv                       # process multiple videos from CSV file
transub prepare-model                         # download/initialize configured local ASR model
transub show_config
transub init --config ./transub.conf   # rerun the setup wizard
transub configure                      # edit config (0 saves, Q discards)
transub run demo.mp4 --transcribe-only # export raw transcription only
transub run demo.mp4 -T              # short flag for transcribe-only
transub --version                    # print the installed version
```

The work directory (defaults to `~/.cache/transub`) stores audio, transcription segments, translation progress, and pipeline state. If a run is interrupted, re-running the same command resumes where it left off. Use `--work-dir` to point at a custom cache location when needed.

## Development

If you want to contribute to `transub`, you can set up a development environment.

### Desktop GUI

The desktop frontend is an **Electron + React + TypeScript** app in `desktop/`. It provides a visual interface for configuring the pipeline, managing provider credentials, running transcription/translation, and previewing subtitles.

To run the desktop app in development:

```bash
cd desktop
npm install
npm run electron:dev
```

Use `npm run electron:dev` to test real desktop functionality such as native file pickers and the Python bridge. `npm run dev` is only a browser renderer preview.

Credential handling in the Python backend:

- provider-scoped credentials are stored in `~/.transub/auth.toml`;
- `TRANSUB_AUTH` can override the auth file path;
- environment variables still take precedence over auth-file keys;
- never commit auth files or print API keys in logs.

### Installation from Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PiktCai/transub.git
    cd transub
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install in editable mode with development dependencies:**
    ```bash
    uv sync --extra dev
    ```
    To enable concurrent translation support, install the `async` extra:
    ```bash
    uv sync --extra async
    ```
4.  **Install a Whisper backend for testing:**
    ```bash
    uv add openai-whisper
    ```

### Running Tests

```bash
python -m unittest
```

### Code Structure

- Source lives in `transub/` (`cli.py`, `config.py`, `transcribe.py`, `translate.py`, `subtitles.py`, etc.).
- Add tests beside related modules (e.g., `transub/test_subtitles.py`).
- Use Rich console utilities and `transub.logger.setup_logging` for consistent output.

## Project Layout

```
transub/
├── audio.py
├── batch.py
├── cache.py
├── cli.py
├── config.py
├── free_translate.py
├── optimize.py
├── segmentation.py
├── subtitles.py
├── transcribe.py
├── translate.py
└── test_*.py         # Co-located tests (unittest)

desktop/              # Electron + React desktop frontend
├── electron/         # Main process & preload
├── src/              # React app (pages, components, lib)
├── package.json
└── vite.config.ts
```

## License

This project is distributed for personal use and study; there is no formal contribution process at this time.  
Transub is released under the [MIT License](LICENSE).
