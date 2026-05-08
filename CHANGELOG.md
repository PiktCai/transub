# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- **Standardized on faster-whisper**: Removed all alternative ASR backends (local/whisper, mlx, whisper.cpp, cloud API, SenseVoice, Qwen3-ASR). Transcription now uses only `faster-whisper` with word-level timestamps.
- **Simplified WhisperConfig**: Removed `backend`, `execution_mode`, `cli_path`, `cpp_*`, `mlx_*`, `api_*`, `tune_segmentation`, `forced_aligner` fields. Config now has `model`, `device`, `language`, `word_timestamps`, and Whisper tuning parameters.
- **Concurrent translation**: Improved error handling; failed chunks now raise `LLMTranslationError` instead of being silently skipped. Removed synchronous fallback retry path.
- **CLI**: `WHISPER_MODEL_SUGGESTIONS` simplified from nested dict to flat list. Removed backend selection from wizard and configure commands.

### Added
- `transub prepare-model` command for pre-downloading/initializing the local ASR model.
- Progress messages (`print(flush=True)`) for long-running CLI steps so Electron GUI can display them.
- Desktop FastAPI status endpoint and per-run debug log files under `~/.cache/transub/logs/`.

### Removed
- All non-faster-whisper ASR backends and their configuration fields.
- `DEFAULT_OPENAI_TRANSCRIBE_URL` constant.
- `DownloadProgressBar`, `DOWNLOAD_CONSOLE`, `_suppress_tqdm` helpers (no longer needed).

### Fixed
- Desktop pipeline events are now broadcast from background worker threads through the server event loop, so the GUI receives progress and completion updates reliably.
- Electron now falls back to a nearby backend port when `18789` is already occupied and reports API errors as structured responses.
- The run screen no longer presents a black terminal-style log panel as the primary UI; it shows task stages, output paths, and a revealable debug log instead.

## [0.2.2] - 2025-10-27

### Added
- **Smart retry handler**: Improved API call resilience with intelligent retry logic for better reliability
- **CJK punctuation handling**: Enhanced punctuation processing for Chinese, Japanese, and Korean text

### Fixed
- **Translation retry logic**: Improved handling of partial responses from LLM APIs
- **Subtitle splitting**: Refined splitting algorithm for better rhythm and readability

### Improved
- Better error handling and retry mechanisms for API calls
- Enhanced subtitle processing stability

## [0.2.1] - 2025-10-27

### Improved

- **Subtitle splitting logic**: Adjusted line splitting algorithm to favor natural semantic boundaries over maximizing line length. This results in shorter, more readable lines, aligning the style closer to professional subtitle standards.
- **Subtitle rhythm and stability**: Made the merging logic for short-duration subtitles more aggressive. This significantly reduces the "flickering" of very short lines, providing a smoother and more stable viewing experience.

## [0.2.0] - 2025-10-27

### 🚨 BREAKING CHANGES

- **Removed legacy config parameters**: Character-based limits (`max_chars_per_line`, `translation_max_chars_per_line`, etc.) replaced by display width parameters (`max_display_width`, `translation_max_display_width`, etc.). `timing_trim_seconds` removed in favor of automatic pause detection.
- **API changes**: `adjust_timing()` method replaced by `apply_offset()`. `refine()` now requires `min_duration` parameter (default: 1.2s).
- **Word-level timestamps enabled by default**: All Whisper backends now extract word-level timestamps for precise timing.
- **`min_line_duration` default changed**: Increased from 0.6s to 1.2s to meet professional subtitle standards and eliminate "flash" subtitles.

### Added

- **Word-level timing & intelligent splitting**: Subtitles split at natural pauses, silence segments removed automatically, and lines respect semantic boundaries for professional quality
  - `pause_threshold_seconds` (0.3s), `silence_threshold_seconds` (2.0s), `remove_silence_segments` (true), `prefer_sentence_boundaries` (true)
- **Display width control**: Industry-standard single-line display guarantee with CJK character awareness (2× width)
  - Source: `max_display_width` (42.0), `min_display_width` (20.0)
  - Translation: `translation_max_display_width` (30.0), `translation_min_display_width` (15.0)
- **CJK punctuation simplification**: Optional replacement of commas/periods with spaces (`simplify_cjk_punctuation`, default: false)
- **Orphaned word merging**: Short lines automatically merged to previous line (up to 25% overage) for better readability

### Changed

- Simplified time offset logic, enabled word timestamps for all backends, enhanced configuration UI

### Improved

- **Duration-aware merging**: Short-duration segments now automatically merge to avoid "flash" subtitles (respecting professional standard: <1s segments ≤5%)
  - Merging respects long silence gaps and won't merge across intentional splits
  - Significantly reduces short subtitle flashes while maintaining readability
- **CPS (Characters Per Second) control**: New `max_cps` parameter prevents information overload
  - Default: 20.0 for mixed text (suitable for English with numbers/symbols)
  - Recommended: 12-15 for CJK-only content
  - All merge operations now respect CPS limits to ensure comfortable reading speed
- Faster splitting algorithm, accurate timing without manual tuning, better handling of varied speech patterns

## [0.1.2] - 2025-10-26

### Fixed
- Support both `mlx-whisper` and `mlx_whisper` CLI names for MLX backend detection
- Updated license format to SPDX standard, eliminating setuptools warnings
- Properly excluded test files from distribution packages

### Documentation
- Clarified that Whisper backend installation is optional when using cloud API
- Highlighted API backend option as recommended for quick start
- Removed outdated pipx inject notes (smart fallback logic makes them unnecessary)
- Added explanation of shared API key usage for OpenAI services
- Improved sample configuration with API backend examples

## [0.1.1] - 2025-10-26

### Changed
- Default the working directory to `~/.cache/transub`, keeping intermediate audio, transcription segments, and translation progress out of the project tree.
- Export subtitles to the same directory as the source video unless `pipeline.output_dir` is set, reducing stray output folders.
- Refreshed English and Chinese READMEs to highlight the new defaults, add pip/pipx installation flows, and ensure links work on PyPI.

### Packaging
- Bumped the published version to `0.1.1`.
- Shipped `transub-sample.conf` from within the package so editable installs and wheels reference the same path.

## [0.1.0] - 2025-10-26

- Initial release.
