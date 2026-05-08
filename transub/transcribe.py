from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import WhisperConfig
from .subtitles import SubtitleDocument


class TranscriptionError(RuntimeError):
    """Raised when audio transcription fails."""


def check_dependencies(config: WhisperConfig) -> None:
    """Ensure the single supported transcription engine is installed."""

    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise TranscriptionError(
            "Transub uses faster-whisper for local transcription. "
            "From the project folder, run: uv sync"
        ) from exc


def prepare_transcription_model(config: WhisperConfig) -> str:
    """Download or initialize the configured faster-whisper model."""

    check_dependencies(config)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - check_dependencies covers this
        raise TranscriptionError(
            "Transub uses faster-whisper for local transcription. "
            "From the project folder, run: uv sync"
        ) from exc

    model_size = config.model or "base"
    device = config.device or "cpu"
    compute_type = _compute_type_for_device(device)

    WhisperModel(model_size, device=device, compute_type=compute_type)
    return (
        f"faster-whisper model is ready: {model_size} "
        f"({device}, {compute_type})"
    )


def transcribe_audio(
    audio_path: Path,
    config: WhisperConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SubtitleDocument:
    """Transcribe audio with faster-whisper and word-level timestamps."""

    check_dependencies(config)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - check_dependencies covers this
        raise TranscriptionError(
            "Transub uses faster-whisper for local transcription. "
            "From the project folder, run: uv sync"
        ) from exc

    model_size = config.model or "base"
    device = config.device or "cpu"
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=_compute_type_for_device(device),
    )

    transcribe_kwargs: dict[str, Any] = dict(config.extra_args)
    if config.language and config.language != "auto":
        transcribe_kwargs.setdefault("language", config.language)
    transcribe_kwargs.setdefault("word_timestamps", True)
    if config.temperature is not None:
        transcribe_kwargs.setdefault("temperature", config.temperature)
    if config.compression_ratio_threshold is not None:
        transcribe_kwargs.setdefault(
            "compression_ratio_threshold",
            config.compression_ratio_threshold,
        )
    if config.logprob_threshold is not None:
        transcribe_kwargs.setdefault("log_prob_threshold", config.logprob_threshold)
    if config.no_speech_threshold is not None:
        transcribe_kwargs.setdefault("no_speech_threshold", config.no_speech_threshold)
    if config.condition_on_previous_text is not None:
        transcribe_kwargs.setdefault(
            "condition_on_previous_text",
            config.condition_on_previous_text,
        )
    if config.initial_prompt:
        transcribe_kwargs.setdefault("initial_prompt", config.initial_prompt)

    try:
        segments_gen, info = model.transcribe(str(audio_path), **transcribe_kwargs)
    except TypeError as exc:
        raise TranscriptionError(f"faster-whisper rejected transcription options: {exc}") from exc

    duration = max(float(getattr(info, "duration_after_vad", 0.0) or getattr(info, "duration", 0.0) or 0.0), 0.0)
    segments: list[dict[str, Any]] = []
    for segment in segments_gen:
        text = segment.text.strip()
        if not text:
            continue
        payload: dict[str, Any] = {
            "start": segment.start,
            "end": segment.end,
            "text": text,
        }
        words = getattr(segment, "words", None)
        if words:
            payload["words"] = [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                }
                for word in words
            ]
        segments.append(payload)
        if progress_callback:
            progress_callback(
                {
                    "segment_count": len(segments),
                    "position": float(segment.end or 0.0),
                    "duration": duration,
                    "text": text,
                }
            )

    if not segments:
        raise TranscriptionError("faster-whisper returned no subtitle segments.")

    return SubtitleDocument.from_whisper_segments(segments)


def _compute_type_for_device(device: str) -> str:
    if device == "cpu":
        return "int8"
    return "float16"


__all__ = ["transcribe_audio", "prepare_transcription_model", "check_dependencies", "TranscriptionError"]
