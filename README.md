# Transub

[中文说明](README.zh-CN.md)

Transub is a Typer-based CLI that automates the journey from **video** to **Chinese subtitles**. It extracts audio with `ffmpeg`, transcribes English speech via Whisper (local, mlx, whisper.cpp, or API backends), and batches subtitle translation through an LLM. Post-processing keeps line breaks natural, timing aligned, and CJK/Latin spacing consistent.

## Key Features

- **End-to-end pipeline**: `transub run <video.mp4>` handles audio extraction → transcription → translation → subtitle export.
- **Backend flexibility**: choose from local Whisper models, `mlx-whisper`, `whisper.cpp`, or OpenAI-compatible APIs.
- **Robust translation**: JSON-constrained prompts, retry logic, and configurable batch sizes keep LLM output reliable.
- **Subtitle polishing**: punctuation-aware line splitting, configurable min/max line length, timing offsets, and automatic spacing between Chinese and Latin characters.
- **Stateful execution**: cached artifacts enable resuming partially completed runs; generated outputs live in `./output`.

## Quick Start

### Prerequisites

- Python 3.10+
- `ffmpeg`
- A compatible translation API (default: OpenRouter / DeepSeek)
- Optional: install `mlx-whisper`, `openai-whisper`, or `whisper.cpp` depending on your transcription backend

### Installation

```bash
git clone https://github.com/PiktCai/transub.git
cd transub
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Generate Your First Subtitles

```bash
transub init           # guided configuration wizard (creates transub.conf)
transub run video.mp4  # produce zh/en subtitles under ./output
```

Outputs:

- `video.zh_cn.srt` – translated subtitles with normalized spacing.
- `video.en.srt` – English transcription (configurable).

## Configuration

All runtime options live in `transub.conf` (TOML). Important sections:

- `[whisper]` – backend selection, model name, device, advanced parameters.
- `[llm]` – translation provider/model, temperature, batch size, retry policy.
- `[pipeline]` – output format, line-length targets, timing trim/offset, punctuation removal, CJK spacing.

Example snippet:

```toml
[pipeline]
output_format = "srt"
timing_offset_seconds = 0.05
translation_max_chars_per_line = 26
translation_min_chars_per_line = 16
normalize_cjk_spacing = true
```

Use `transub configure` to adjust these interactively, or edit the file directly.

## CLI Essentials

```bash
transub run demo.mp4 --config ~/transub.conf --work-dir /tmp/transub
transub show_config
transub init --config ./transub.conf        # rerun the setup wizard
transub configure                           # tweak an existing config
```

Cache directory `.transub/` stores intermediate state (audio cache, transcription JSON, translation progress). If a run fails or you cancel midway, the next `transub run` resumes from that state.

## Development Workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest
```

- Core modules live under `transub/` (`cli.py`, `config.py`, `transcribe.py`, `translate.py`, `subtitles.py`).
- Add feature-specific tests under `transub/test_subtitles.py` or a new module in `tests/`.
- Keep CLI messages styled via Rich; reuse shared helpers for logging (`transub.logger`).

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

This project is shared for personal use and study; there is no formal contribution process at this time.  
Transub is released under the [MIT License](LICENSE).
