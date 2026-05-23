# Archive Status

Transub is archived as of 2026-05-23. The repository remains available as the preserved Python/Electron implementation of the subtitle pipeline, but it is no longer the preferred place for new workflow experiments.

## Successor Direction

Future subtitle workflow work should start from the agent-native `transub` skill:

```text
https://github.com/PiktCai/skills/tree/main/transub
```

That skill keeps the durable parts of this project:

- local `ffmpeg` and `faster-whisper` transcription;
- SRT/VTT/segments JSON parsing, validation, and export;
- line-index and timestamp preservation rules;
- subtitle translation, ASR correction, polishing, glossary, and QA guidance.

The skill intentionally leaves behind the heavier product shell:

- Electron desktop UI;
- FastAPI/SSE progress plumbing;
- provider secret management screens;
- PyPI/Homebrew/desktop packaging work;
- complex concurrent API orchestration unless explicitly needed again.

## Maintenance Policy

Use this repository for:

- reference when improving the `transub` skill;
- small documentation corrections;
- security or secret-handling fixes;
- reproducibility checks of the old CLI/backend behavior;
- deliberate app revival work if explicitly requested.

Avoid adding new product features here by default. If a change is a workflow rule, prompt, QA checklist, or deterministic helper for agents, put it in the `transub` skill first.

## Verification

If touching this archive, keep the old project constraints:

```bash
python -m unittest discover
```

Use `uv` for dependency installation. Desktop work still requires the Electron app path documented in `docs/desktop.md`, but should be treated as maintenance-only unless the archive status changes.
