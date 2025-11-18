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
- **Multiple transcription backends** — choose local Whisper, `mlx-whisper`, `whisper.cpp`, or OpenAI-compatible APIs.
- **Reliable translations** — JSON-constrained prompts, retry logic, and configurable batch sizes.
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

`transub` supports multiple Whisper backends. Choose one based on your needs:

- **Cloud API (Recommended for quick start):**
  - Uses OpenAI's Whisper API or compatible endpoints
  - No local installation required
  - Set `OPENAI_API_KEY` environment variable
  - Configure with `backend = "api"` during setup

- **Local backends (for offline use or custom models):**
  - **For most users (local, CPU/GPU):**
    ```bash
    pip install openai-whisper
    ```
  
  - **For Apple Silicon (macOS):**
    ```bash
    pip install mlx-whisper
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

## Configuration Overview

Runtime settings live in `transub.conf` (TOML). Key sections:

- `[whisper]` — backend selection, model name, device overrides, and extra arguments.
- `[llm]` — translation provider/model, temperature, batch size, retry policy.
- `[pipeline]` — output format, line-length targets, timing trim/offset, punctuation and spacing options.

Example:

```toml
[pipeline]
output_format = "srt"
translation_max_chars_per_line = 26
translation_min_chars_per_line = 16
normalize_cjk_spacing = true
timing_offset_seconds = 0.05
```

Run `transub configure` for an interactive editor, or update the file manually. Configuration files are user-specific and should not be committed.

## CLI Cheatsheet

```bash
transub run demo.mp4 --config ~/transub.conf --work-dir /tmp/transub  # override work dir (defaults to ~/.cache/transub)
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
    pip install -e ".[dev]"
    ```
4.  **Install a Whisper backend for testing:**
    ```bash
    pip install openai-whisper
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
├── cli.py
├── config.py
├── subtitles.py
├── transcribe.py
├── translate.py
└── test_subtitles.py
```

## License

This project is distributed for personal use and study; there is no formal contribution process at this time.  
Transub is released under the [MIT License](LICENSE).
