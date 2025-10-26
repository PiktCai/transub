# Changelog

All notable changes to this project will be documented in this file.

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
