# Transub Documentation

This directory contains project documentation beyond the root README files.

## Documents

| File | Purpose |
|------|---------|
| [`desktop.md`](desktop.md) | Electron desktop architecture, local FastAPI backend, desktop API endpoints, and debug logs. |
| [`packaging.md`](packaging.md) | PyPI and Homebrew release workflow. |

## Root Files

The repository root intentionally keeps only high-signal entry files:

- `README.md` and `README.zh-CN.md` for user-facing project entry points.
- `CHANGELOG.md` for release notes.
- `LICENSE`, `MANIFEST.in`, `pyproject.toml`, and `uv.lock` for packaging and project metadata.

Longer maintenance notes should live here under `docs/`.
