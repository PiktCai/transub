from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import PipelineConfig


class AudioExtractionError(RuntimeError):
    """Raised when ffmpeg fails to extract audio."""


FFMPEG_NOT_FOUND = (
    "ffmpeg is required but was not found in PATH. "
    "Install ffmpeg and try again."
)


def ensure_ffmpeg() -> str:
    """Return ffmpeg executable path or raise if missing."""

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise FileNotFoundError(FFMPEG_NOT_FOUND)
    return ffmpeg_path


def extract_audio(
    video_path: Path,
    pipeline: PipelineConfig,
    work_dir: Path,
) -> Path:
    """Extract audio from the given video file using ffmpeg."""

    ffmpeg_path = ensure_ffmpeg()
    audio_format = pipeline.audio_format
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / f"{video_path.stem}.{audio_format}"

    cmd = _build_ffmpeg_cmd(
        ffmpeg_path=ffmpeg_path,
        video_path=video_path,
        audio_path=audio_path,
        audio_format=audio_format,
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed with exit code {result.returncode}: {result.stderr.strip()}"
        )
    return audio_path


def _build_ffmpeg_cmd(
    ffmpeg_path: str, video_path: Path, audio_path: Path, audio_format: str
) -> list[str]:
    """Construct ffmpeg command depending on desired output format."""

    base_cmd = [ffmpeg_path, "-y", "-i", str(video_path), "-vn"]

    if audio_format == "wav":
        base_cmd += ["-ac", "1", "-ar", "16000", "-f", "wav", str(audio_path)]
    elif audio_format == "mp3":
        base_cmd += ["-acodec", "libmp3lame", str(audio_path)]
    elif audio_format == "flac":
        base_cmd += ["-acodec", "flac", str(audio_path)]
    elif audio_format == "m4a":
        base_cmd += ["-acodec", "aac", str(audio_path)]
    elif audio_format == "ogg":
        base_cmd += ["-acodec", "libvorbis", str(audio_path)]
    else:
        raise ValueError(f"Unsupported audio format: {audio_format}")

    return base_cmd


__all__ = [
    "AudioExtractionError",
    "extract_audio",
    "ensure_ffmpeg",
]

