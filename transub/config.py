from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from textwrap import dedent

try:  # Python 3.11+
    import tomllib as tomli  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli  # type: ignore[no-redef]

import tomli_w
from pydantic import BaseModel, Field, ValidationError, model_validator

CONFIG_FILENAME = "transub.conf"

DEFAULT_TRANSLATION_PROMPT = dedent(
    """\
    # Role: Senior Subtitle Translator
    You are an experienced subtitle translator who delivers clear, natural ${targetLanguage} subtitles.

    # Guidelines
    1. Keep each subtitle line independent; never merge or split entries.
    2. Prefer conversational ${targetLanguage} that reads well in subtitles.
    3. Use punctuation (comma, period, ellipsis) to preserve rhythm and tone.
    4. Translate terminology accurately and keep it consistent across lines.
    5. Use straight quotes instead of curly quotes.
    6. When the translation mixes different scripts (for example, Latin letters within Chinese text), add spacing where appropriate so the subtitle stays legible.

    # Output Format
    1. Return a JSON object that mirrors the input keys (IDs) and only translates the values.
    2. Do not add commentary or text outside the JSON object.
    3. Ensure the JSON is syntactically valid and contains the same number of entries as the input.

    Review your work to confirm the translation is fluent, faithful to the source, and grammatically correct. Adapt passive/active voice as needed for natural ${targetLanguage} subtitles.

    # Example

    Input:
    {"0": "Welcome to China", "1": "China is a beautiful country"}

    Output:
    {"0": "欢迎来到中国", "1": "中国是一个美丽的国家"}
    """
).strip()


class WhisperConfig(BaseModel):
    """Configuration for Whisper transcription."""

    backend: str = Field(
        default="local", description="Which backend to use: local, api, cpp, or mlx"
    )
    execution_mode: str = Field(
        default="internal",
        description="Execution strategy: 'internal' (python import) or 'external' (cli subprocess)",
    )
    cli_path: str | None = Field(
        default=None,
        description="Path to the CLI executable for 'external' mode",
    )
    model: str = Field(default="base", description="Whisper model size")
    device: str | None = Field(
        default=None,
        description="Override compute device, e.g. cuda, cpu, mps",
    )
    api_url: str | None = Field(
        default=None,
        description="Custom transcription API endpoint when backend=api",
    )
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable storing the speech-to-text API key",
    )
    cpp_binary: str = Field(
        default="whisper-cpp",
        description="Executable name or path for the whisper.cpp CLI",
    )
    cpp_model_path: str | None = Field(
        default=None,
        description="Path to the ggml/gguf model file when using whisper.cpp backend",
    )
    cpp_threads: int | None = Field(
        default=None,
        description="Optional number of threads for whisper.cpp",
        ge=1,
    )
    cpp_extra_args: list[str] = Field(
        default_factory=list,
        description="Additional CLI arguments for whisper.cpp backend",
    )
    mlx_model_dir: str | None = Field(
        default=None,
        description="Directory containing mlx-whisper converted model weights",
    )
    mlx_dtype: str | None = Field(
        default=None,
        description="Computation dtype for mlx-whisper (auto, float16, float32, etc.)",
    )
    mlx_device: str | None = Field(
        default=None,
        description="Target device for mlx-whisper (cpu, mps). Defaults to mlx auto detect.",
    )
    mlx_extra_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments forwarded to mlx-whisper",
    )
    language: str | None = Field(
        default="en",
        description="Language hint passed to Whisper when transcribing",
    )
    extra_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional backend-specific arguments",
    )
    tune_segmentation: bool = Field(
        default=True,
        description="Apply recommended Whisper segmentation parameters to reduce fragmenting.",
    )
    temperature: float | None = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for local Whisper decoding (None to keep library default).",
    )
    compression_ratio_threshold: float | None = Field(
        default=2.6,
        ge=0.0,
        description="Compression ratio threshold before a segment is discarded (None to keep default).",
    )
    logprob_threshold: float | None = Field(
        default=-1.0,
        description="Minimum average log probability for valid decoding (None to keep default).",
    )
    no_speech_threshold: float | None = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability threshold for classifying a segment as silence (None to keep default).",
    )
    condition_on_previous_text: bool | None = Field(
        default=True,
        description="Condition decoding on previous text to maintain context.",
    )
    initial_prompt: str | None = Field(
        default=None,
        description="Optional initial prompt to guide Whisper's segmentation and terminology.",
    )
    word_timestamps: bool = Field(
        default=True,
        description="Extract word-level timestamps for more accurate subtitle timing (strongly recommended).",
    )

    @model_validator(mode="after")
    def validate_backend(self) -> "WhisperConfig":
        backend = self.backend.lower()
        if backend not in {"local", "api", "cpp", "mlx"}:
            raise ValueError("backend must be 'local', 'api', 'cpp', or 'mlx'")
        object.__setattr__(self, "backend", backend)
        if backend == "cpp" and not self.cpp_model_path:
            raise ValueError("cpp_model_path must be set when backend is 'cpp'")
        return self


class LLMConfig(BaseModel):
    """Configuration for the translation LLM."""

    provider: str = Field(default="openai", description="LLM provider identifier")
    model: str = Field(default="gpt-4o-mini", description="Model name")
    api_base: str | None = Field(
        default=None, description="Custom endpoint base URL if needed"
    )
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable storing the API key",
    )
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_retries: int = Field(default=3, ge=0, le=10)
    request_timeout: float = Field(default=60.0, gt=0)
    batch_size: int = Field(
        default=5,
        description="Number of subtitle lines per translation request",
        ge=1,
        le=50,
    )
    target_language: str = Field(default="zh", description="Target translation language")
    style: str | None = Field(
        default="Simplified Chinese",
        description="Optional description of translation tone/style",
    )


class PipelineConfig(BaseModel):
    """Configuration for pipeline level options."""

    output_format: str = Field(
        default="srt",
        description="Subtitle output format (srt or vtt)",
    )
    audio_format: str = Field(
        default="wav",
        description="Intermediate audio format for whisper input",
    )
    keep_temp_audio: bool = Field(
        default=False, description="Keep intermediate extracted audio file"
    )
    output_dir: str | None = Field(
        default=None,
        description="Directory for generated subtitle files",
    )
    save_source_subtitles: bool = Field(
        default=True,
        description="Whether to save the intermediate source-language subtitles",
    )
    max_display_width: float = Field(
        default=42.0,
        ge=20.0,
        le=100.0,
        description="Maximum display width for subtitles (industry standard: 42 for English, CJK chars count as 2)",
    )
    min_display_width: float = Field(
        default=20.0,
        ge=5.0,
        le=80.0,
        description="Minimum display width to avoid very short subtitle lines",
    )
    prefer_sentence_boundaries: bool = Field(
        default=True,
        description="Prefer splitting at sentence/phrase boundaries for better readability",
    )
    timing_offset_seconds: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Shift applied to subtitle start/end times (positive delays subtitles, negative moves earlier)",
    )
    min_line_duration: float = Field(
        default=1.2,
        ge=0.5,
        le=5.0,
        description="Minimum duration (seconds) each subtitle should stay on screen (professional standard: 1.0-1.5s)",
    )
    max_cps: float = Field(
        default=20.0,
        ge=10.0,
        le=30.0,
        description="Maximum characters per second (CPS) to prevent information overload (20 for mixed text, 12-15 for CJK-only)",
    )
    remove_trailing_punctuation: bool = Field(
        default=True,
        description="Remove trailing punctuation from translated subtitles",
    )
    normalize_cjk_spacing: bool = Field(
        default=True,
        description="Insert spaces between CJK characters and Latin/digit sequences for readability",
    )
    prompt_preamble: str = Field(
        default=DEFAULT_TRANSLATION_PROMPT,
        description="System prompt prepended to LLM translation requests",
    )
    translation_max_display_width: float | None = Field(
        default=30.0,
        ge=15.0,
        le=80.0,
        description="Maximum display width for translated subtitles (30.0 suitable for CJK languages)",
    )
    translation_min_display_width: float | None = Field(
        default=15.0,
        ge=5.0,
        le=60.0,
        description="Minimum display width for translated subtitles",
    )
    refine_source_subtitles: bool = Field(
        default=False,
        description="Whether to re-refine and reflow source subtitles on export",
    )
    simplify_cjk_punctuation: bool = Field(
        default=False,
        description="Replace commas and periods with spaces in CJK subtitles (common style for Chinese subtitles)",
    )
    pause_threshold_seconds: float = Field(
        default=0.3,
        ge=0.1,
        le=2.0,
        description="Minimum gap between words to consider as a natural pause (used for smart splitting with word_timestamps)",
    )
    silence_threshold_seconds: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Minimum gap between words to consider as silence (such segments can be removed)",
    )
    remove_silence_segments: bool = Field(
        default=True,
        description="Remove subtitle segments that contain only long silences (requires word_timestamps)",
    )

    @model_validator(mode="after")
    def validate_format(self) -> "PipelineConfig":
        fmt = self.output_format.lower()
        if fmt not in {"srt", "vtt"}:
            raise ValueError("output_format must be 'srt' or 'vtt'")
        object.__setattr__(self, "output_format", fmt)
        audio_fmt = self.audio_format.lower()
        if audio_fmt not in {"wav", "mp3", "flac", "m4a", "ogg"}:
            raise ValueError(
                "audio_format must be one of wav, mp3, flac, m4a, ogg"
            )
        object.__setattr__(self, "audio_format", audio_fmt)
        if self.min_display_width > self.max_display_width:
            raise ValueError("min_display_width cannot exceed max_display_width")
        if (
            self.translation_max_display_width is not None
            and self.translation_min_display_width is not None
            and self.translation_min_display_width > self.translation_max_display_width
        ):
            raise ValueError(
                "translation_min_display_width cannot exceed translation_max_display_width"
            )
        return self


class TransubConfig(BaseModel):
    """Top-level configuration model for Transub."""

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


@dataclass
class ConfigManager:
    """Handles loading and saving the CLI configuration."""

    path: Path

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> TransubConfig:
        if not self.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.path}. Run 'transub init' first."
            )
        with self.path.open("rb") as fh:
            raw = tomli.load(fh)
        try:
            return TransubConfig.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(
                "Configuration file is invalid. Please fix the errors or re-run 'transub init'."
            ) from exc

    def save(self, config: TransubConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            tomli_w.dump(
                config.model_dump(mode="json", exclude_none=True), fh
            )

    @classmethod
    def default_path(cls) -> Path:
        # Environment variable overrides everything so packaged installs can
        # point at a shared config location.
        env_path = os.getenv("TRANSUB_CONFIG")
        if env_path:
            return Path(env_path).expanduser()

        return Path.home() / ".transub" / CONFIG_FILENAME


def load_or_create_default(path: Optional[Path] = None) -> TransubConfig:
    """Load config if present, otherwise write and return defaults."""

    cfg_path = path or ConfigManager.default_path()
    manager = ConfigManager(cfg_path)
    if manager.exists():
        return manager.load()
    default_config = TransubConfig()
    manager.save(default_config)
    return default_config


__all__ = [
    "CONFIG_FILENAME",
    "DEFAULT_TRANSLATION_PROMPT",
    "ConfigManager",
    "TransubConfig",
    "WhisperConfig",
    "LLMConfig",
    "PipelineConfig",
    "load_or_create_default",
]
