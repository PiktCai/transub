from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional


class PipelineState:
    """Simple JSON-backed state to support resuming a pipeline run."""

    def __init__(self, path: Path, data: dict) -> None:
        self.path = path
        self._data = data

    @classmethod
    def load(cls, path: Path, video_path: Path) -> "PipelineState":
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = {
                "video": str(video_path.resolve()),
                "audio": {},
                "transcription": {},
                "translation": {},
            }
        return cls(path, data)

    # ---- Audio -----------------------------------------------------

    def get_audio_path(self) -> Optional[Path]:
        audio = self._data.get("audio", {})
        path = audio.get("path")
        return Path(path) if path else None

    def set_audio_path(self, path: Path) -> None:
        self._data.setdefault("audio", {})["path"] = str(path)
        self.save()

    # ---- Transcription ---------------------------------------------

    def get_segments_path(self) -> Optional[Path]:
        transcription = self._data.get("transcription", {})
        path = transcription.get("segments_path")
        return Path(path) if path else None

    def mark_transcription(
        self, segments_path: Path, total_lines: int
    ) -> None:
        self._data.setdefault("transcription", {}).update(
            {
                "segments_path": str(segments_path),
                "total_lines": total_lines,
            }
        )
        self.save()

    def transcription_total_lines(self) -> Optional[int]:
        transcription = self._data.get("transcription", {})
        return transcription.get("total_lines")

    # ---- Translation -----------------------------------------------

    def translation_progress_path(self, default: Path) -> Path:
        translation = self._data.setdefault("translation", {})
        path = translation.get("progress_path")
        if not path:
            translation["progress_path"] = str(default)
            self.save()
            return default
        return Path(path)

    def completed_lines(self) -> set[str]:
        translation = self._data.get("translation", {})
        return set(translation.get("completed_lines", []))

    def mark_lines_completed(self, line_ids: Iterable[str]) -> None:
        translation = self._data.setdefault("translation", {})
        completed = set(translation.get("completed_lines", []))
        completed.update(line_ids)
        translation["completed_lines"] = sorted(completed, key=int)
        self.save()

    def reset_translation(self) -> None:
        if "translation" in self._data:
            del self._data["translation"]
        self.save()

    # ---- Cleanup ---------------------------------------------------

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    # ---- Persistence -----------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)


def load_translation_progress(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("line_translations", {})


def persist_translation_progress(path: Path, translations: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"line_translations": translations}, fh, ensure_ascii=False, indent=2)


__all__ = ["PipelineState", "load_translation_progress", "persist_translation_progress"]

