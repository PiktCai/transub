# Transub

[中文说明](README.zh-CN.md)

Automate the journey from **video** to **Chinese subtitles** with a single Typer command. Transub extracts audio with `ffmpeg`, runs Whisper for transcription, and applies an LLM-driven translation pipeline that keeps punctuation, timing, and CJK/Latin spacing polished.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
  - [Before You Begin](#before-you-begin)
  - [Windows (PowerShell, Local Whisper)](#windows-powershell-local-whisper)
  - [macOS (Apple Silicon, mlx-whisper)](#macos-apple-silicon-mlx-whisper)
  - [Linux (Bash, Local Whisper)](#linux-bash-local-whisper)
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
- **Subtitle polishing** — punctuation-aware line splitting, timing offsets, and automatic spacing between Chinese and Latin characters.
- **Stateful execution** — cached progress in `.transub/` avoids rework across runs.

## Quick Start

### Before You Begin

- **Python 3.10+** installed on your system.
- **ffmpeg** available on `PATH`.
- **LLM API access** (default config expects an OpenAI-compatible endpoint; set the `api_key_env` variable in `transub.conf` to match your environment).
- Enough disk space for temporary audio and transcription artifacts under `./.transub/`.

Follow the guide for your platform to install dependencies and select the recommended Whisper backend.

> Tip: Need only the English transcription? Append `--transcribe-only` to `transub run` to skip translation.

### Windows (PowerShell, Local Whisper)

1. **Install prerequisites**
   - [Python 3.10+](https://www.python.org/downloads/windows/)
   - `ffmpeg`: `winget install Gyan.FFmpeg` or `choco install ffmpeg`
2. **Clone and create a virtual environment**
   ```powershell
   git clone https://github.com/PiktCai/transub.git
   cd transub
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e .
   pip install openai-whisper
   ```
   Install a CUDA-enabled PyTorch wheel if GPU acceleration is desired.
3. **Configure Transub**
   ```powershell
   transub init
   ```
   Keep `backend = "local"` in `[whisper]` and select a model such as `small` or `medium`.
4. **Run the pipeline**
   ```powershell
   transub run .\video.mp4 --work-dir .\.transub
   ```
   Subtitles are written to `.\output\` (e.g., `video.zh_cn.srt`, `video.en.srt`).

### macOS (Apple Silicon, mlx-whisper)

1. **Install prerequisites**
   - Python 3.10+ (`brew install python@3.11` if needed)
   - `ffmpeg`: `brew install ffmpeg`
2. **Clone and set up the environment**
   ```bash
   git clone https://github.com/PiktCai/transub.git
   cd transub
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install mlx-whisper
   ```
   `mlx-whisper` runs Whisper efficiently on Apple Silicon GPUs/NPUs.
3. **Configure Transub**
   ```bash
   transub init
   ```
   Set `backend = "mlx"` in `[whisper]`; choose a model like `mlx-community/whisper-small.en-mlx` and optionally supply `mlx_model_dir` for local weights.
4. **Run the pipeline**
   ```bash
   transub run ./video.mp4 --work-dir ./.transub
   ```
   Subtitles appear in `./output/` as `.srt` or `.vtt` depending on your configuration.

### Linux (Bash, Local Whisper)

1. **Install prerequisites**
   ```bash
   sudo apt update && sudo apt install ffmpeg python3 python3-venv  # Debian/Ubuntu
   # Arch: sudo pacman -S ffmpeg python python-virtualenv
   ```
2. **Clone and create a virtual environment**
   ```bash
   git clone https://github.com/PiktCai/transub.git
   cd transub
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install openai-whisper
   ```
   Install an appropriate PyTorch wheel (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`) if your GPU should accelerate Whisper.
3. **Configure Transub**
   ```bash
   transub init
   ```
   Use the wizard with `backend = "local"` and pick a Whisper model that fits your hardware.
4. **Run the pipeline**
   ```bash
   transub run ./video.mp4 --work-dir ./.transub
   ```
   Generated subtitles are saved under `./output/`.

> Tip: Clear the `.transub/` work directory between experiments if you switch videos or Whisper configurations to avoid stale caches.

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
transub run demo.mp4 --config ~/transub.conf --work-dir /tmp/transub
transub show_config
transub init --config ./transub.conf   # rerun the setup wizard
transub configure                      # edit an existing config
transub run demo.mp4 --transcribe-only # export English transcription only
transub run demo.mp4 -T              # short flag for transcribe-only
```

Cache directory `.transub/` stores audio, transcription segments, translation progress, and pipeline state. If a run is interrupted, re-running the same command resumes where it left off.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest
```

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
