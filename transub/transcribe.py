from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import requests
from rich.console import Console
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    HfHubHTTPError,
)

from .config import WhisperConfig
from .subtitles import SubtitleDocument

DEFAULT_OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"


class TranscriptionError(RuntimeError):
    """Raised when audio transcription fails."""


class DownloadProgressBar(tqdm):
    """Custom tqdm progress bar with richer formatting for downloads."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault(
            "bar_format",
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        kwargs.setdefault("dynamic_ncols", True)
        super().__init__(*args, **kwargs)


DOWNLOAD_CONSOLE = Console()


@contextmanager
def _suppress_tqdm() -> Any:
    previous = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = previous


def check_dependencies(config: WhisperConfig) -> None:
    """Check for required dependencies and determine execution mode."""
    backend = config.backend
    if backend == "api":
        return  # No local dependencies to check

    # Try to import the package first
    try:
        if backend == "local":
            import whisper  # type: ignore
        elif backend == "mlx":
            import mlx_whisper  # type: ignore
        config.execution_mode = "internal"
        return
    except ImportError:
        pass  # Package not found, try to find a CLI tool

    # If package import fails, look for a CLI tool
    cli_map: dict[str, list[str]] = {
        "local": ["whisper"],
        "mlx": ["mlx-whisper", "mlx_whisper"],
        "cpp": [config.cpp_binary] if config.cpp_binary else ["whisper-cpp"],
    }
    cli_candidates = cli_map.get(backend)
    if not cli_candidates:
        return # Should not happen with validated config

    for candidate in cli_candidates:
        exec_path = shutil.which(candidate)
        if exec_path:
            config.execution_mode = "external"
            config.cli_path = exec_path
            if backend == "cpp":
                config.cpp_binary = exec_path
            return

    # If neither package nor CLI is found, raise an error
    if backend == "local":
        raise TranscriptionError(
            "For the 'local' backend, you must either install 'openai-whisper' in the same "
            "environment as transub, or have the 'whisper' command-line tool in your PATH."
        )
    if backend == "mlx":
        raise TranscriptionError(
            "For the 'mlx' backend, you must either install 'mlx-whisper' in the same "
            "environment as transub, or have the 'mlx-whisper' (or 'mlx_whisper') command-line tool in your PATH."
        )
    if backend == "cpp":
        binary = config.cpp_binary or "whisper-cpp"
        raise TranscriptionError(
            f"whisper.cpp executable '{binary}' not found in PATH. "
            "Please install whisper.cpp and ensure it is in your PATH."
        )


def transcribe_audio(audio_path: Path, config: WhisperConfig) -> SubtitleDocument:
    """Transcribe audio file using the configured backend."""

    if config.backend == "local":
        return _transcribe_local(audio_path, config)
    if config.backend == "api":
        return _transcribe_api(audio_path, config)
    if config.backend == "cpp":
        return _transcribe_cpp(audio_path, config)
    if config.backend == "mlx":
        return _transcribe_mlx(audio_path, config)
    raise TranscriptionError(f"Unsupported whisper backend: {config.backend}")


def _ensure_local_mlx_repo(repo_id: str) -> str:
    """Ensure MLX whisper model repo is present locally and return its path."""

    # If repo_id already points to an existing directory, use it directly.
    repo_path = Path(repo_id).expanduser()
    if repo_path.exists():
        return str(repo_path)

    token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )

    try:
        return snapshot_download(
            repo_id=repo_id,
            local_files_only=True,
            token=token,
        )
    except LocalEntryNotFoundError:
        DOWNLOAD_CONSOLE.print(
            f"Downloading MLX Whisper model [bold]{repo_id}[/]…"
        )
        try:
            return snapshot_download(
                repo_id=repo_id,
                resume_download=True,
                tqdm_class=DownloadProgressBar,
                token=token,
            )
        except RepositoryNotFoundError as err:
            raise TranscriptionError(
                "Unable to download MLX Whisper model. The repository may be private "
                "or requires authentication. Provide a Hugging Face token via the "
                "HUGGINGFACEHUB_API_TOKEN / HF_TOKEN environment variable, or download "
                "the model manually and set 'mlx_model_dir' to the local path."
            ) from err
        except HfHubHTTPError as err:
            raise TranscriptionError(
                "Failed to contact Hugging Face Hub. Network access may be disabled on "
                "this environment. Download the MLX model manually and configure "
                "'mlx_model_dir' to point to it."
            ) from err
        except Exception as exc:  # pragma: no cover - defensive
            raise TranscriptionError(
                f"Unexpected error while downloading MLX model: {exc}"
            ) from exc


def _transcribe_local(audio_path: Path, config: WhisperConfig) -> SubtitleDocument:
    if config.execution_mode == "internal":
        return _transcribe_local_internal(audio_path, config)
    elif config.execution_mode == "external":
        return _transcribe_local_external(audio_path, config)
    else:
        raise TranscriptionError(f"Unsupported execution mode for local backend: {config.execution_mode}")


def _transcribe_local_internal(audio_path: Path, config: WhisperConfig) -> SubtitleDocument:
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise TranscriptionError(
            "whisper Python package is required for internal transcription. "
            "Install it via 'pip install openai-whisper' or ensure the 'whisper' CLI is in your PATH."
        ) from exc

    device = config.device or None
    with _suppress_tqdm():
        model = whisper.load_model(config.model, device=device)
    transcribe_kwargs: Dict[str, Any] = dict(config.extra_args)
    if config.language:
        transcribe_kwargs.setdefault("language", config.language)
    if config.word_timestamps:
        transcribe_kwargs.setdefault("word_timestamps", True)
    if config.tune_segmentation:
        if config.temperature is not None:
            transcribe_kwargs.setdefault("temperature", config.temperature)
        if config.compression_ratio_threshold is not None:
            transcribe_kwargs.setdefault(
                "compression_ratio_threshold", config.compression_ratio_threshold
            )
        if config.logprob_threshold is not None:
            transcribe_kwargs.setdefault("logprob_threshold", config.logprob_threshold)
        if config.no_speech_threshold is not None:
            transcribe_kwargs.setdefault("no_speech_threshold", config.no_speech_threshold)
        if config.condition_on_previous_text is not None:
            transcribe_kwargs.setdefault(
                "condition_on_previous_text", config.condition_on_previous_text
            )
        if config.initial_prompt:
            transcribe_kwargs.setdefault("initial_prompt", config.initial_prompt)

    result = model.transcribe(str(audio_path), **transcribe_kwargs)
    segments = result.get("segments") or []
    if not segments:
        raise TranscriptionError("Whisper returned no segments")
    return SubtitleDocument.from_whisper_segments(segments)


def _transcribe_local_external(audio_path: Path, config: WhisperConfig) -> SubtitleDocument:
    if not config.cli_path:
        raise TranscriptionError("cli_path not set for external local execution.")

    output_dir = audio_path.parent
    output_json = output_dir / f"{audio_path.stem}.json"

    cmd = [
        config.cli_path,
        str(audio_path),
        "--model",
        config.model,
        "--output_dir",
        str(output_dir),
        "--output_format",
        "json",
    ]
    if config.language:
        cmd.extend(["--language", config.language])
    if config.device:
        cmd.extend(["--device", config.device])
    if config.word_timestamps:
        cmd.extend(["--word_timestamps", "True"])

    # Add extra arguments from extra_args
    for key, value in (config.extra_args or {}).items():
        cmd.append(f"--{key}")
        if not isinstance(value, bool) or value:
            cmd.append(str(value))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise TranscriptionError(
            f"Whisper CLI failed with exit code {result.returncode}: {result.stderr.strip()}"
        )

    if not output_json.exists():
        raise TranscriptionError(
            f"Whisper CLI did not produce JSON output at {output_json}"
        )

    with output_json.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    
    output_json.unlink()

    segments = payload.get("segments") or []
    if not segments:
        raise TranscriptionError("Whisper CLI output did not contain segments")
    return SubtitleDocument.from_whisper_segments(segments)


def _transcribe_api(audio_path: Path, config: WhisperConfig) -> SubtitleDocument:
    api_url = config.api_url or DEFAULT_OPENAI_TRANSCRIBE_URL
    api_key_env = config.api_key_env or "OPENAI_API_KEY"
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise TranscriptionError(
            f"Environment variable {api_key_env} is not set for API transcription"
        )
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": config.model,
        **{k: v for k, v in (config.extra_args or {}).items()},
    }
    if config.language:
        data.setdefault("language", config.language)
    if config.word_timestamps:
        # OpenAI API uses timestamp_granularities parameter
        data.setdefault("timestamp_granularities", ["word", "segment"])
    if config.tune_segmentation:
        if config.temperature is not None:
            data.setdefault("temperature", config.temperature)
        if config.initial_prompt:
            data.setdefault("prompt", config.initial_prompt)

    with audio_path.open("rb") as file_handle:
        files = {"file": (audio_path.name, file_handle)}
        response = requests.post(
            api_url, headers=headers, files=files, data=data, timeout=300
        )
    if response.status_code >= 400:
        raise TranscriptionError(
            f"Transcription API error {response.status_code}: {response.text.strip()}"
        )

    payload = response.json()
    segments = _extract_segments_from_api(payload)
    if not segments:
        raise TranscriptionError(
            "Transcription API returned unexpected payload without segments"
        )
    return SubtitleDocument.from_whisper_segments(segments)


def _transcribe_cpp(audio_path: Path, config: WhisperConfig) -> SubtitleDocument:
    binary = config.cpp_binary or "whisper-cpp"
    exec_path = shutil.which(binary)
    if not exec_path:
        raise TranscriptionError(
            f"whisper.cpp executable '{binary}' not found in PATH."
        )
    if not config.cpp_model_path:
        raise TranscriptionError("cpp_model_path must be set for whisper.cpp backend.")

    output_prefix = audio_path.stem + "_whispercpp"
    output_json = audio_path.parent / f"{output_prefix}.json"
    try:
        output_json.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass

    cmd: List[str] = [
        exec_path,
        "-f",
        str(audio_path),
        "-m",
        config.cpp_model_path,
        "-oj",
        "-of",
        str(audio_path.parent / output_prefix),
    ]
    if config.language:
        cmd.extend(["-l", config.language])
    if config.cpp_threads:
        cmd.extend(["-t", str(config.cpp_threads)])
    if config.cpp_extra_args:
        cmd.extend(config.cpp_extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise TranscriptionError(
            "whisper.cpp failed with exit code "
            f"{result.returncode}: {result.stderr.strip()}"
        )
    if not output_json.exists():
        raise TranscriptionError(
            f"whisper.cpp did not produce JSON output at {output_json}"
        )

    with output_json.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    segments = _extract_segments_from_cpp(payload)
    try:
        output_json.unlink()
    except OSError:
        pass
    if not segments:
        raise TranscriptionError("whisper.cpp JSON output did not contain segments")
    return SubtitleDocument.from_whisper_segments(segments)


def _transcribe_mlx(audio_path: Path, config: WhisperConfig) -> SubtitleDocument:
    if config.execution_mode == "internal":
        return _transcribe_mlx_internal(audio_path, config)
    elif config.execution_mode == "external":
        return _transcribe_mlx_external(audio_path, config)
    else:
        raise TranscriptionError(f"Unsupported execution mode for mlx backend: {config.execution_mode}")


def _transcribe_mlx_internal(audio_path: Path, config: WhisperConfig) -> SubtitleDocument:
    try:
        import mlx_whisper  # type: ignore
    except ImportError as exc:
        raise TranscriptionError(
            "mlx-whisper package is required for internal execution. "
            "Install it or ensure the 'mlx_whisper' CLI tool is in your PATH for external execution."
        ) from exc

    kwargs: Dict[str, Any] = dict(config.mlx_extra_args or {})
    if config.language:
        kwargs.setdefault("language", config.language)
    if config.word_timestamps:
        kwargs.setdefault("word_timestamps", True)
    if config.tune_segmentation:
        if config.no_speech_threshold is not None:
            kwargs.setdefault("no_speech_threshold", config.no_speech_threshold)
        if config.temperature is not None:
            kwargs.setdefault("temperature", config.temperature)
        if config.initial_prompt:
            kwargs.setdefault("initial_prompt", config.initial_prompt)

    model_identifier: str | None = None
    if config.mlx_model_dir:
        candidate_dir = Path(config.mlx_model_dir).expanduser()
        if not candidate_dir.exists():
            raise TranscriptionError(
                f"mlx_model_dir does not exist: {candidate_dir}"
            )
        model_identifier = str(candidate_dir)
    elif config.model:
        model_identifier = config.model.strip()

    if not model_identifier:
        repo_id = "mlx-community/whisper-small.en"
    else:
        candidate_path = Path(model_identifier)
        if candidate_path.exists():
            repo_id = str(candidate_path)
        elif model_identifier.startswith("mlx-community/") or "/" in model_identifier:
            repo_id = model_identifier
        else:
            repo_id = f"mlx-community/whisper-{model_identifier}"

    local_repo_path = _ensure_local_mlx_repo(repo_id)

    transcribe_fn = getattr(mlx_whisper, "transcribe", None)
    if not callable(transcribe_fn):
        raise TranscriptionError(
            "Installed mlx-whisper package does not expose a 'transcribe' function."
        )

    transcribe_kwargs = dict(kwargs)
    if config.mlx_dtype:
        dtype_lower = config.mlx_dtype.lower()
        if dtype_lower in {"float32", "fp32"}:
            transcribe_kwargs.setdefault("fp16", False)
        elif dtype_lower in {"float16", "fp16"}:
            transcribe_kwargs.setdefault("fp16", True)
    if config.mlx_device:
        transcribe_kwargs.setdefault("device", config.mlx_device)

    try:
        result = transcribe_fn(
            str(audio_path),
            path_or_hf_repo=local_repo_path,
            **transcribe_kwargs,
        )
    except TypeError:
        result = transcribe_fn(
            audio=str(audio_path),
            path_or_hf_repo=local_repo_path,
            **transcribe_kwargs,
        )

    if not isinstance(result, dict):
        raise TranscriptionError("mlx-whisper returned unexpected result format.")
    segments = result.get("segments") or []
    if not segments:
        raise TranscriptionError("mlx-whisper returned no segments.")
    return SubtitleDocument.from_whisper_segments(segments)


def _transcribe_mlx_external(audio_path: Path, config: WhisperConfig) -> SubtitleDocument:
    if not config.cli_path:
        raise TranscriptionError("cli_path not set for external mlx execution.")

    output_dir = Path(config.mlx_model_dir or '.')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = audio_path.stem
    output_json = output_dir / f"{output_name}.json"

    cmd = [
        config.cli_path,
        str(audio_path),
        "--model",
        config.model,
        "--output-dir",
        str(output_dir),
        "--output-name",
        output_name,
        "--output-format",
        "json",
    ]
    if config.language:
        cmd.extend(["--language", config.language])
    if config.word_timestamps:
        cmd.extend(["--word-timestamps", "True"])

    # Add extra arguments from mlx_extra_args
    for key, value in (config.mlx_extra_args or {}).items():
        cmd.append(f"--{key}")
        if not isinstance(value, bool) or value:
            cmd.append(str(value))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise TranscriptionError(
            f"mlx_whisper CLI failed with exit code {result.returncode}: {result.stderr.strip()}"
        )

    if not output_json.exists():
        raise TranscriptionError(
            f"mlx_whisper CLI did not produce JSON output at {output_json}"
        )

    with output_json.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    
    output_json.unlink()

    segments = payload.get("segments") or []
    if not segments:
        raise TranscriptionError("mlx_whisper CLI output did not contain segments")
    return SubtitleDocument.from_whisper_segments(segments)


def _extract_segments_from_api(payload: Dict[str, Any]) -> list[dict[str, Any]]:
    if "segments" in payload:
        return payload["segments"]
    if "text" in payload and "timestamps" in payload:
        return payload["timestamps"]
    if "text" in payload:
        return [
            {
                "start": 0,
                "end": 0,
                "text": payload["text"],
            }
        ]
    return []


def _extract_segments_from_cpp(payload: Dict[str, Any]) -> list[dict[str, Any]]:
    segments = payload.get("segments")
    if not segments and "transcription" in payload:
        segments = payload["transcription"].get("segments")
    results: list[dict[str, Any]] = []
    for segment in segments or []:
        text = segment.get("text")
        if not text:
            continue
        if "start" in segment and "end" in segment:
            start = float(segment["start"])
            end = float(segment["end"])
        else:
            t0 = segment.get("t0")
            t1 = segment.get("t1")
            if t0 is None or t1 is None:
                continue
            # whisper.cpp reports t0/t1 in 10ms units
            start = float(t0) * 0.01
            end = float(t1) * 0.01
        results.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )
    return results


__all__ = ["transcribe_audio", "TranscriptionError"]
