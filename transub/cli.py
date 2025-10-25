from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.progress import SpinnerColumn, Progress, TextColumn, TaskProgressColumn
from .audio import AudioExtractionError, extract_audio
from .config import ConfigManager, TransubConfig, DEFAULT_TRANSLATION_PROMPT
from .logger import setup_logging
from .state import (
    PipelineState,
    load_translation_progress,
    persist_translation_progress,
)
from .subtitles import SubtitleDocument
from .transcribe import TranscriptionError, transcribe_audio
from .translate import LLMTranslationError, translate_subtitles

app = typer.Typer(add_completion=False, help="Generate Chinese subtitles from English videos.")
console = Console()
WHISPER_MODEL_SUGGESTIONS: dict[str, list[str]] = {
    "local": [
        "small",
        "medium",
        "large-v3",
        "large-v2",
        "base",
        "tiny",
    ],
    "mlx": [
        "mlx-community/whisper-small.en-mlx",
        "mlx-community/whisper-medium.en-mlx",
        "mlx-community/whisper-large-v3",
        "mlx-community/whisper-large-v2",
    ],
    "api": [
        "gpt-4o-mini-transcribe",
        "gpt-4o-transcribe",
        "whisper-1",
    ],
    "cpp": [
        "ggml-small.en.bin",
        "ggml-medium.en.bin",
        "ggml-large-v3.bin",
        "gguf-large-v3-q5_1.bin",
    ],
}


@app.command()
def init(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Custom path for the configuration file",
    )
) -> None:
    """Guided configuration setup."""

    manager = ConfigManager(config_path or ConfigManager.default_path())
    _run_wizard(manager, allow_overwrite=True)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Path to the video file"),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Custom configuration file location"
    ),
    work_dir: Optional[Path] = typer.Option(
        Path("./.transub"),
        "--work-dir",
        help="Temporary working directory for intermediate files",
    ),
    transcribe_only: bool = typer.Option(
        False,
        "--transcribe-only",
        "-T",
        help="Skip translation and export the transcription only.",
    ),
) -> None:
    """Run the end-to-end subtitle creation pipeline."""

    config = _load_config(config_path)

    console.print(Panel.fit("Transub pipeline", style="bold cyan"))
    console.print(f"Video: [bold]{video}[/]")

    work_dir = (work_dir or Path("./.transub")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(work_dir / "transub.log")
    logger.info("Starting pipeline for %s", video)

    state_path = work_dir / f"{video.stem}_state.json"
    state = PipelineState.load(state_path, video)

    audio_path: Optional[Path] = state.get_audio_path()
    segments_path: Optional[Path] = state.get_segments_path()
    translations_path: Optional[Path] = None
    interrupted = False

    success = False
    try:
        if audio_path and audio_path.exists():
            audio_path = audio_path.resolve()
            console.print(f"Using cached audio at [italic]{audio_path}[/]")
            logger.info("Using cached audio %s", audio_path)
        else:
            with console.status("Extracting audio…", spinner="dots"):
                audio_path = extract_audio(video, config.pipeline, work_dir)
            audio_path = audio_path.resolve()
            state.set_audio_path(audio_path)
            console.print(f"Audio extracted to [italic]{audio_path}[/]")
            logger.info("Audio extracted to %s", audio_path)

        if segments_path and segments_path.exists():
            with segments_path.open("r", encoding="utf-8") as fh:
                segment_payload = json.load(fh)
            source_doc = SubtitleDocument.from_serialized(segment_payload)
            console.print("Loaded cached transcription.")
            logger.info(
                "Loaded cached transcription from %s (%d lines)",
                segments_path,
                len(source_doc.lines),
            )
            if state.transcription_total_lines() is None:
                state.mark_transcription(segments_path, len(source_doc.lines))
        else:
            with console.status(
                "Transcribing audio with Whisper…", spinner="dots"
            ):
                raw_doc = transcribe_audio(audio_path, config.whisper)
            refined_doc = raw_doc.refine(
                max_chars=config.pipeline.max_chars_per_line,
                min_chars=config.pipeline.min_chars_per_line,
            )
            if (
                config.pipeline.timing_trim_seconds > 0
                or config.pipeline.timing_offset_seconds != 0
            ):
                refined_doc = refined_doc.adjust_timing(
                    trim=config.pipeline.timing_trim_seconds,
                    min_duration=config.pipeline.min_line_duration,
                    offset=config.pipeline.timing_offset_seconds,
                )
            segments_path = work_dir / f"{video.stem}_segments.json"
            segments_path.write_text(
                json.dumps(refined_doc.to_serializable(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            state.mark_transcription(segments_path, len(refined_doc.lines))
            source_doc = refined_doc
            console.print("Transcription complete.")
            logger.info(
                "Transcription finished with %d refined lines", len(refined_doc.lines)
            )

        total_lines = state.transcription_total_lines() or len(source_doc.lines)
        if total_lines <= 0:
            total_lines = len(source_doc.lines)
            if segments_path:
                state.mark_transcription(segments_path, total_lines)

        if transcribe_only:
            console.print("[cyan]Transcribe-only mode: skipping translation.[/]")
            logger.info("Transcribe-only mode enabled; skipping translation stage.")

            english_doc = source_doc
            max_trans_chars = config.pipeline.translation_max_chars_per_line
            min_trans_chars: int | None = None
            if (
                config.pipeline.refine_source_subtitles
                and max_trans_chars
            ):
                min_trans_chars = (
                    config.pipeline.translation_min_chars_per_line
                    or min(config.pipeline.min_chars_per_line, max_trans_chars)
                )
                english_doc = source_doc.refine(
                    max_chars=max_trans_chars,
                    min_chars=min_trans_chars,
                )

            english_path = _write_document(
                document=english_doc,
                target_dir=Path(config.pipeline.output_dir),
                stem=video.stem,
                suffix=".en",
                output_format=config.pipeline.output_format,
            )
            console.print(Panel.fit(f"✅ Transcription saved to {english_path}", style="green"))
            logger.info("Transcription exported to %s", english_path)

            success = True
            return

        translations_path = state.translation_progress_path(
            work_dir / f"{video.stem}_translations.json"
        )
        existing_translations: Dict[str, str] = load_translation_progress(translations_path)
        if existing_translations:
            state.mark_lines_completed(existing_translations.keys())
            logger.info(
                "Loaded cached translations for %d lines",
                len(existing_translations),
            )
            console.print(
                f"Resuming translation: {len(existing_translations)}/{total_lines} lines already completed."
            )

        translations_cache: Dict[str, str] = dict(existing_translations)

        initial_completed = len(translations_cache)

        def _progress_description(done: int) -> str:
            return f"[bold cyan]Translating[/] {done}/{total_lines}"

        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("{task.description}"),
            TaskProgressColumn(),
            console=console,
            transient=True,
        )

        with progress:
            task_id = progress.add_task(
                description=_progress_description(initial_completed),
                total=total_lines,
                completed=initial_completed,
            )

            def handle_progress(new_items: Dict[str, str]) -> None:
                translations_cache.update(new_items)
                persist_translation_progress(translations_path, translations_cache)
                state.mark_lines_completed(new_items.keys())
                done = len(translations_cache)
                progress.update(
                    task_id,
                    completed=done,
                    description=_progress_description(done),
                )
                logger.info(
                    "Translated lines %s",
                    ", ".join(sorted(new_items.keys(), key=int)),
                )

            translated_doc, usage_stats = translate_subtitles(
                source_doc,
                config.llm,
                config.pipeline,
                existing_translations=translations_cache,
                progress_callback=handle_progress,
            )
        console.print(
            "Translation complete. Tokens used: "
            f"prompt {usage_stats['prompt']}, "
            f"completion {usage_stats['completion']}, "
            f"total {usage_stats['total']}"
            f" (translated {len(translations_cache)}/{total_lines} lines)"
        )
        persist_translation_progress(translations_path, translations_cache)

        output_doc = translated_doc
        max_trans_chars = config.pipeline.translation_max_chars_per_line
        min_trans_chars = None
        if max_trans_chars:
            min_trans_chars = (
                config.pipeline.translation_min_chars_per_line
                or min(config.pipeline.min_chars_per_line, max_trans_chars)
            )
            output_doc = translated_doc.refine(
                max_chars=max_trans_chars,
                min_chars=min_trans_chars,
            )

        if config.pipeline.remove_trailing_punctuation:
            output_doc = output_doc.remove_trailing_punctuation()
        if config.pipeline.normalize_cjk_spacing:
            output_doc = output_doc.normalize_cjk_spacing()

        output_suffix = _language_suffix(config.llm.target_language)
        output_path = _write_document(
            document=output_doc,
            target_dir=Path(config.pipeline.output_dir),
            stem=video.stem,
            suffix=output_suffix,
            output_format=config.pipeline.output_format,
        )
        console.print(Panel.fit(f"✅ Finished! Output saved to {output_path}", style="green"))
        logger.info(
            "Translation finished. Output saved to %s. Tokens prompt=%d completion=%d total=%d",
            output_path,
            usage_stats["prompt"],
            usage_stats["completion"],
            usage_stats["total"],
        )

        if config.pipeline.save_source_subtitles:
            source_output_doc = source_doc
            if (
                config.pipeline.refine_source_subtitles
                and max_trans_chars
                and min_trans_chars is not None
            ):
                source_output_doc = source_doc.refine(
                    max_chars=max_trans_chars,
                    min_chars=min_trans_chars,
                )
            english_path = _write_document(
                document=source_output_doc,
                target_dir=Path(config.pipeline.output_dir),
                stem=video.stem,
                suffix=".en",
                output_format=config.pipeline.output_format,
            )
            console.print(f"English subtitles saved to [italic]{english_path}[/]")
            logger.info("English subtitles saved to %s", english_path)

        success = True

    except KeyboardInterrupt as exc:  # pragma: no cover - manual interrupt
        interrupted = True
        logger.info("Pipeline interrupted by user.")
        console.print("[yellow]Pipeline interrupted by user. Partial results kept.[/]")
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        logger.exception("Missing dependency: %s", exc)
        console.print(f"[red]Missing dependency:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except AudioExtractionError as exc:
        logger.exception("Audio extraction failed: %s", exc)
        console.print(f"[red]Audio extraction failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except TranscriptionError as exc:
        logger.exception("Transcription failed: %s", exc)
        console.print(f"[red]Transcription failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except LLMTranslationError as exc:
        logger.exception("Translation failed: %s", exc)
        console.print(f"[red]Translation failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if success:
            if (
                config
                and work_dir.exists()
                and not config.pipeline.keep_temp_audio
                and audio_path
            ):
                _cleanup_audio_file(work_dir, audio_path)
            try:
                if translations_path and translations_path.exists():
                    translations_path.unlink()
            except OSError:
                pass
            try:
                current_segments = state.get_segments_path()
                if current_segments and current_segments.exists():
                    current_segments.unlink()
            except OSError:
                pass
            state.clear()
        else:
            _offer_failure_cleanup(
                work_dir=work_dir,
                state=state,
                audio_path=audio_path,
                translations_path=translations_path,
                interrupted=interrupted,
            )


def _load_config(config_path: Optional[Path]) -> TransubConfig:
    manager = ConfigManager(config_path or ConfigManager.default_path())
    if manager.exists():
        return manager.load()

    target_path = manager.path
    console.print(
        Panel.fit(
            f"No configuration found at {target_path}.",
            style="yellow",
        )
    )
    if Confirm.ask("Run the setup wizard now?", default=True):
        _run_wizard(manager, allow_overwrite=True)
        return manager.load()

    console.print(
        "[red]Configuration missing.[/] Run `transub init` or specify --config pointing to an existing file."
    )
    raise typer.Exit(code=1)


@app.command()
def show_config(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Custom configuration file location"
    )
) -> None:
    """Display the current configuration."""

    manager = ConfigManager(config_path or ConfigManager.default_path())
    if not manager.exists():
        console.print("Configuration file not found. Run `transub init` first.")
        raise typer.Exit(code=1)
    config = manager.load()
    console.print(json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False))


@app.command()
def configure(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Custom configuration file location"
    )
) -> None:
    """Interactively adjust existing configuration."""

    manager = ConfigManager(config_path or ConfigManager.default_path())
    if not manager.exists():
        console.print("Configuration file not found. Run `transub init` first.")
        raise typer.Exit(code=1)

    config = manager.load()
    option_handlers = [
        ("Whisper backend & model", _configure_whisper_backend),
        ("Whisper advanced parameters", _configure_whisper_advanced),
        ("Translation LLM", _configure_llm),
        ("Pipeline & output", _configure_pipeline),
        ("View raw JSON", None),
    ]

    while True:
        summary_lines = _config_summary_lines(config)
        console.clear()
        console.print(
            Panel(
                "\n".join(summary_lines),
                title="Transub configuration editor",
                border_style="cyan",
            )
        )
        console.print("[bold]0.[/] Save & exit")
        for idx, (label, _) in enumerate(option_handlers, start=1):
            console.print(f"[bold]{idx}.[/] {label}")
        choice = Prompt.ask("Select an option", default="0")
        if choice == "0":
            manager.save(config)
            console.clear()
            console.print(
                Panel.fit(f"Configuration saved to {manager.path}", style="green")
            )
            return
        try:
            idx = int(choice)
        except ValueError:
            _wait_for_enter("Please enter a valid number. Press Enter to continue...")
            continue
        if idx < 1 or idx > len(option_handlers):
            _wait_for_enter("Selection out of range. Press Enter to continue...")
            continue
        label, handler = option_handlers[idx - 1]
        if handler is None:
            console.clear()
            console.print(Panel(json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False), title="Raw configuration"))
            _wait_for_enter()
        else:
            handler(config)


def _prompt_for_config() -> TransubConfig:
    console.print("We'll gather a few preferences. Press enter to accept defaults.")
    whisper_backend = Prompt.ask(
        "Whisper backend", choices=["local", "api", "cpp", "mlx"], default="local"
    )
    model_suggestions = WHISPER_MODEL_SUGGESTIONS.get(whisper_backend, [])
    default_model = "base"
    if model_suggestions:
        default_model = model_suggestions[0]
        console.print(
            f"Suggested models for backend '{whisper_backend}': "
            + ", ".join(model_suggestions)
        )
    whisper_model = Prompt.ask(
        "Whisper model id or path", default=default_model, show_choices=False
    )
    whisper_device = Prompt.ask(
        "Preferred device (cuda/cpu/mps)", default="", show_choices=False
    )
    whisper_language = Prompt.ask("Source language hint", default="en")
    whisper_tune_segmentation = Confirm.ask(
        "Enable Whisper segmentation tuning (reduce fragmenting)?", default=True
    )
    whisper_initial_prompt = Prompt.ask(
        "Whisper initial prompt (optional, useful for terminology)", default="", show_choices=False
    )

    llm_provider = Prompt.ask("LLM provider name", default="openai")
    llm_model = Prompt.ask("LLM model id", default="gpt-4o-mini")
    llm_api_base = Prompt.ask("LLM API base URL (blank for default)", default="")
    llm_api_key_env = Prompt.ask("LLM API key env var", default="OPENAI_API_KEY")
    llm_batch_size = _prompt_int("Lines per translation batch", default=5, minimum=1, maximum=50)
    target_language = Prompt.ask("Target language", default="zh")
    translation_style = Prompt.ask(
        "Desired style (e.g. Simplified Chinese/Traditional Chinese/Colloquial)", default="Simplified Chinese"
    )

    output_format = Prompt.ask("Output subtitle format", choices=["srt", "vtt"], default="srt")
    audio_format = Prompt.ask(
        "Intermediate audio format", choices=["wav", "mp3", "flac", "m4a", "ogg"], default="wav"
    )
    output_dir = Prompt.ask("Output directory", default="./output")
    keep_temp = Confirm.ask("Keep extracted audio file?", default=False)
    save_source = Confirm.ask("Save English subtitles as well?", default=True)
    max_chars = _prompt_int("Max characters per subtitle line", default=54, minimum=20, maximum=160)
    min_chars = _prompt_int(
        "Min characters per subtitle line", default=22, minimum=10, maximum=max_chars
    )
    translation_max_chars = _prompt_int(
        "Max characters per translated line", default=26, minimum=10, maximum=160
    )
    translation_min_chars = _prompt_int(
        "Min characters per translated line", default=16, minimum=1, maximum=translation_max_chars
    )

    default_prompt = DEFAULT_TRANSLATION_PROMPT
    if Confirm.ask("Open editor for system prompt? (multi-line supported)", default=False):
        edited = typer.edit(default_prompt + "\n")
        pipeline_prompt = (edited or default_prompt).strip()
    else:
        pipeline_prompt = Prompt.ask(
            "Custom system prompt for translation",
            default=default_prompt,
        )

    whisper_config: dict[str, object] = {
        "backend": whisper_backend,
        "model": whisper_model,
        "device": whisper_device or None,
        "language": whisper_language or None,
        "api_key_env": "OPENAI_API_KEY",
        "tune_segmentation": whisper_tune_segmentation,
        "initial_prompt": whisper_initial_prompt or None,
    }
    if whisper_backend == "api":
        whisper_api_url = Prompt.ask(
            "Whisper API URL", default="https://api.openai.com/v1/audio/transcriptions"
        )
        whisper_api_key_env = Prompt.ask(
            "Whisper API key env var", default="OPENAI_API_KEY"
        )
        whisper_config["api_url"] = whisper_api_url
        whisper_config["api_key_env"] = whisper_api_key_env
    elif whisper_backend == "cpp":
        cpp_binary = Prompt.ask(
            "whisper.cpp executable name or path", default="whisper-cpp"
        )
        cpp_model_path = _prompt_non_empty(
            "whisper.cpp model file path (.bin/.gguf)", default=""
        )
        cpp_threads = _prompt_optional_int(
            "whisper.cpp thread count (leave blank for auto)", minimum=1
        )
        cpp_extra_args_raw = Prompt.ask(
            "Extra whisper.cpp CLI arguments (space separated, optional)", default=""
        )
        cpp_extra_args = shlex.split(cpp_extra_args_raw) if cpp_extra_args_raw else []
        whisper_config.update(
            {
                "cpp_binary": cpp_binary,
                "cpp_model_path": cpp_model_path,
                "cpp_threads": cpp_threads,
                "cpp_extra_args": cpp_extra_args,
            }
        )
    elif whisper_backend == "mlx":
        mlx_model_dir = Prompt.ask("mlx-whisper model directory (optional)", default="")
        mlx_dtype = Prompt.ask("mlx-whisper dtype (auto/float16/float32, optional)", default="")
        mlx_device = Prompt.ask("mlx-whisper device (auto/mps/cpu, optional)", default="")
        mlx_extra_args = _prompt_json_dict(
            "Extra mlx-whisper arguments (JSON object, optional)", default="{}"
        )
        whisper_config.update(
            {
                "mlx_model_dir": mlx_model_dir or None,
                "mlx_dtype": mlx_dtype or None,
                "mlx_device": mlx_device or None,
                "mlx_extra_args": mlx_extra_args,
            }
        )

    llm_config = {
        "provider": llm_provider,
        "model": llm_model,
        "api_base": llm_api_base or None,
        "api_key_env": llm_api_key_env,
        "batch_size": llm_batch_size,
        "target_language": target_language,
        "style": translation_style or None,
    }

    pipeline_config = {
        "output_format": output_format,
        "audio_format": audio_format,
        "output_dir": output_dir,
        "keep_temp_audio": keep_temp,
        "save_source_subtitles": save_source,
        "max_chars_per_line": max_chars,
        "min_chars_per_line": min_chars,
        "translation_max_chars_per_line": translation_max_chars,
        "translation_min_chars_per_line": translation_min_chars,
        "prompt_preamble": pipeline_prompt,
    }

    data = {
        "whisper": whisper_config,
        "llm": llm_config,
        "pipeline": pipeline_config,
    }
    return TransubConfig.model_validate(data)


def _run_wizard(manager: ConfigManager, allow_overwrite: bool) -> None:
    console.print(Panel.fit("Transub setup assistant", style="bold cyan"))
    if manager.exists():
        if not allow_overwrite:
            console.print("Configuration untouched.")
            raise typer.Exit(code=0)
        overwrite = Confirm.ask(
            f"A config already exists at [bold]{manager.path}[/]. Overwrite?", default=False
        )
        if not overwrite:
            console.print("Configuration untouched.")
            raise typer.Exit(code=0)

    config = _prompt_for_config()
    manager.save(config)
    console.print(
        Panel.fit(
            f"Configuration saved to {manager.path}",
            style="green",
        )
    )


def _config_summary_lines(config: TransubConfig) -> list[str]:
    whisper = config.whisper
    llm = config.llm
    pipeline = config.pipeline
    return [
        f"[bold]Whisper[/]: backend={whisper.backend} | model={whisper.model} | device={whisper.device or 'auto'} | language={whisper.language or 'auto'}",
        f"[bold]LLM[/]: provider={llm.provider} | model={llm.model} | target={llm.target_language} | batch_size={llm.batch_size}",
        f"[bold]Pipeline[/]: format={pipeline.output_format} | audio={pipeline.audio_format} | dir={pipeline.output_dir} | keep_audio={_fmt_bool(pipeline.keep_temp_audio)} | save_en={_fmt_bool(pipeline.save_source_subtitles)} | max_line={pipeline.max_chars_per_line} | translated_max={pipeline.translation_max_chars_per_line}",
    ]


def _wait_for_enter(message: str = "Press Enter to continue...") -> None:
    console.input(f"[dim]{message}[/]")


def _configure_whisper_backend(config: TransubConfig) -> None:
    whisper = config.whisper
    while True:
        console.clear()
        summary_lines = [
            f"Backend: {whisper.backend}",
            f"Model: {whisper.model}",
            f"Device: {whisper.device or 'auto'} | Language: {whisper.language or 'auto'}",
        ]
        if whisper.backend == "api":
            summary_lines.append(
                f"API url: {whisper.api_url or '(default)'} | key env: {whisper.api_key_env}"
            )
        elif whisper.backend == "cpp":
            summary_lines.append(
                f"cpp_binary: {whisper.cpp_binary} | model_path: {whisper.cpp_model_path or '(none)'} | threads: {whisper.cpp_threads or 'auto'}"
            )
        elif whisper.backend == "mlx":
            summary_lines.append(
                f"mlx_model_dir: {whisper.mlx_model_dir or '(auto)'} | dtype: {whisper.mlx_dtype or 'auto'} | device: {whisper.mlx_device or 'auto'}"
            )
        console.print(
            Panel("\\n".join(summary_lines), title="Whisper backend & model", border_style="cyan")
        )

        options = [
            ("Change backend", "backend"),
            ("Change model", "model"),
            ("Change device", "device"),
            ("Change language", "language"),
        ]
        if whisper.backend == "api":
            options.append(("Set API settings", "api"))
        if whisper.backend == "cpp":
            options.append(("Set whisper.cpp options", "cpp"))
        if whisper.backend == "mlx":
            options.append(("Set mlx options", "mlx"))

        console.print("0. Back")
        for idx, (label, _) in enumerate(options, start=1):
            console.print(f"{idx}. {label}")
        choice = Prompt.ask("Select an option", default="0")
        if choice == "0":
            return
        try:
            _, key = options[int(choice) - 1]
        except (ValueError, IndexError):
            _wait_for_enter("Selection out of range. Press Enter to continue...")
            continue

        if key == "backend":
            new_backend = Prompt.ask(
                "Backend",
                choices=["local", "api", "cpp", "mlx"],
                default=whisper.backend,
            )
            if new_backend != whisper.backend:
                whisper.backend = new_backend
                suggestions = WHISPER_MODEL_SUGGESTIONS.get(new_backend, [])
                if suggestions and Confirm.ask(
                    f"Use suggested model '{suggestions[0]}'?",
                    default=True,
                ):
                    whisper.model = suggestions[0]
            continue

        if key == "model":
            suggestions = WHISPER_MODEL_SUGGESTIONS.get(whisper.backend, [])
            if suggestions:
                console.print("Suggested models: " + ", ".join(suggestions))
            whisper.model = Prompt.ask(
                "Model id or path",
                default=whisper.model,
                show_choices=False,
            )
        elif key == "device":
            whisper.device = (
                Prompt.ask(
                    "Preferred device (blank for auto)",
                    default=whisper.device or "",
                    show_choices=False,
                )
                or None
            )
        elif key == "language":
            whisper.language = (
                Prompt.ask(
                    "Source language hint (blank for auto)",
                    default=whisper.language or "",
                    show_choices=False,
                )
                or None
            )
        elif key == "api":
            whisper.api_url = (
                Prompt.ask(
                    "API URL (blank for default)",
                    default=whisper.api_url or "https://api.openai.com/v1/audio/transcriptions",
                    show_choices=False,
                )
                or None
            )
            whisper.api_key_env = Prompt.ask(
                "API key environment variable",
                default=whisper.api_key_env,
            )
        elif key == "cpp":
            whisper.cpp_binary = Prompt.ask(
                "whisper.cpp executable",
                default=whisper.cpp_binary,
                show_choices=False,
            )
            whisper.cpp_model_path = (
                Prompt.ask(
                    "whisper.cpp model path (.bin/.gguf) (blank to clear)",
                    default=whisper.cpp_model_path or "",
                    show_choices=False,
                )
                or None
            )
            threads_raw = Prompt.ask(
                "whisper.cpp threads (blank for auto)",
                default=str(whisper.cpp_threads or ""),
                show_choices=False,
            ).strip()
            whisper.cpp_threads = int(threads_raw) if threads_raw else None
            extra_args_raw = Prompt.ask(
                "Extra whisper.cpp arguments (space separated)",
                default=" ".join(whisper.cpp_extra_args),
                show_choices=False,
            ).strip()
            whisper.cpp_extra_args = shlex.split(extra_args_raw) if extra_args_raw else []
        elif key == "mlx":
            whisper.mlx_model_dir = (
                Prompt.ask(
                    "mlx-whisper model directory (blank for auto)",
                    default=whisper.mlx_model_dir or "",
                    show_choices=False,
                )
                or None
            )
            whisper.mlx_dtype = (
                Prompt.ask(
                    "mlx dtype (auto/float16/float32)",
                    default=whisper.mlx_dtype or "",
                    show_choices=False,
                )
                or None
            )
            whisper.mlx_device = (
                Prompt.ask(
                    "mlx device (auto/mps/cpu)",
                    default=whisper.mlx_device or "",
                    show_choices=False,
                )
                or None
            )
            extra_json = Prompt.ask(
                "Extra mlx-whisper arguments (JSON)",
                default=json.dumps(whisper.mlx_extra_args or {}),
                show_choices=False,
            )
            try:
                whisper.mlx_extra_args = json.loads(extra_json) if extra_json.strip() else {}
            except json.JSONDecodeError:
                _wait_for_enter("Invalid JSON. Press Enter to continue...")


def _configure_whisper_advanced(config: TransubConfig) -> None:
    whisper = config.whisper
    while True:
        console.clear()
        summary_lines = [
            f"Segmentation tuning: {_fmt_bool(whisper.tune_segmentation)} | Temperature: {whisper.temperature if whisper.temperature is not None else 'auto'} | Compression ratio: {whisper.compression_ratio_threshold}",
            f"Logprob threshold: {whisper.logprob_threshold} | No-speech threshold: {whisper.no_speech_threshold}",
            f"Condition on previous text: {_fmt_bool(bool(whisper.condition_on_previous_text))} | Initial prompt: {whisper.initial_prompt or '(none)'}",
        ]
        console.print(
            Panel("\\n".join(summary_lines), title="Whisper advanced parameters", border_style="cyan")
        )

        options = [
            ("Toggle segmentation tuning", "seg"),
            ("Set temperature", "temp"),
            ("Set compression ratio threshold", "comp"),
            ("Set logprob threshold", "logprob"),
            ("Set no-speech threshold", "nospeech"),
            ("Set condition on previous text", "condition"),
            ("Edit initial prompt", "prompt"),
        ]
        for idx, (label, _) in enumerate(options, start=1):
            console.print(f"{idx}. {label}")
        console.print("0. Back")
        choice = Prompt.ask("Select an option", default="0")
        if choice == "0":
            return
        try:
            _, key = options[int(choice) - 1]
        except (ValueError, IndexError):
            _wait_for_enter("Selection out of range. Press Enter to continue...")
            continue

        if key == "seg":
            whisper.tune_segmentation = not whisper.tune_segmentation
        elif key == "temp":
            whisper.temperature = _prompt_float_optional(
                "Temperature (blank to keep, 'none' to clear)", whisper.temperature
            )
        elif key == "comp":
            whisper.compression_ratio_threshold = _prompt_float_optional(
                "Compression ratio threshold (blank to keep)",
                whisper.compression_ratio_threshold,
            )
        elif key == "logprob":
            whisper.logprob_threshold = _prompt_float_optional(
                "Logprob threshold (blank to keep)", whisper.logprob_threshold
            )
        elif key == "nospeech":
            whisper.no_speech_threshold = _prompt_float_optional(
                "No-speech threshold (blank to keep)", whisper.no_speech_threshold
            )
        elif key == "condition":
            whisper.condition_on_previous_text = Confirm.ask(
                "Condition on previous text?",
                default=bool(whisper.condition_on_previous_text)
                if whisper.condition_on_previous_text is not None
                else True,
            )
        elif key == "prompt":
            whisper.initial_prompt = (
                Prompt.ask(
                    "Initial prompt (blank to clear)",
                    default=whisper.initial_prompt or "",
                    show_choices=False,
                )
                or None
            )


def _configure_llm(config: TransubConfig) -> None:
    llm = config.llm
    while True:
        console.clear()
        summary_lines = [
            f"Provider: {llm.provider} | Model: {llm.model}",
            f"Target: {llm.target_language} | Batch size: {llm.batch_size}",
            f"Temperature: {llm.temperature} | Max retries: {llm.max_retries} | Timeout: {llm.request_timeout}s",
            f"API base: {llm.api_base or '(default)'} | API key env: {llm.api_key_env} | Style: {llm.style or '(none)'}",
        ]
        console.print(Panel("\\n".join(summary_lines), title="Translation LLM", border_style="cyan"))

        options = [
            ("Change provider", "provider"),
            ("Change model", "model"),
            ("Set API base", "base"),
            ("Set API key env", "key"),
            ("Set batch size", "batch"),
            ("Set temperature", "temp"),
            ("Set max retries", "retries"),
            ("Set request timeout", "timeout"),
            ("Set target language", "target"),
            ("Set style descriptor", "style"),
        ]
        for idx, (label, _) in enumerate(options, start=1):
            console.print(f"{idx}. {label}")
        console.print("0. Back")
        choice = Prompt.ask("Select an option", default="0")
        if choice == "0":
            return
        try:
            _, key = options[int(choice) - 1]
        except (ValueError, IndexError):
            _wait_for_enter("Selection out of range. Press Enter to continue...")
            continue

        if key == "provider":
            llm.provider = Prompt.ask("Provider name", default=llm.provider)
        elif key == "model":
            llm.model = Prompt.ask("Model id", default=llm.model)
        elif key == "base":
            llm.api_base = (
                Prompt.ask(
                    "API base URL (blank for default)",
                    default=llm.api_base or "",
                    show_choices=False,
                )
                or None
            )
        elif key == "key":
            llm.api_key_env = Prompt.ask("API key environment variable", default=llm.api_key_env)
        elif key == "batch":
            llm.batch_size = _prompt_int(
                "Lines per translation batch",
                default=llm.batch_size,
                minimum=1,
                maximum=50,
            )
        elif key == "temp":
            llm.temperature = _prompt_float("Temperature", llm.temperature)
        elif key == "retries":
            llm.max_retries = _prompt_int(
                "Max retries",
                default=llm.max_retries,
                minimum=0,
                maximum=10,
            )
        elif key == "timeout":
            llm.request_timeout = _prompt_float(
                "Request timeout (seconds)",
                llm.request_timeout,
            )
        elif key == "target":
            llm.target_language = Prompt.ask(
                "Target language code",
                default=llm.target_language,
            )
        elif key == "style":
            llm.style = (
                Prompt.ask(
                    "Style descriptor (blank to clear)",
                    default=llm.style or "",
                    show_choices=False,
                )
                or None
            )


def _configure_pipeline(config: TransubConfig) -> None:
    pipeline = config.pipeline
    while True:
        console.clear()
        summary_lines = [
            f"Format: {pipeline.output_format} | Audio: {pipeline.audio_format}",
            f"Output dir: {pipeline.output_dir}",
            f"Keep temp audio: {_fmt_bool(pipeline.keep_temp_audio)} | Save English: {_fmt_bool(pipeline.save_source_subtitles)}",
            f"Max line chars: {pipeline.max_chars_per_line} (min {pipeline.min_chars_per_line})",
            f"Translated max: {pipeline.translation_max_chars_per_line} (min {pipeline.translation_min_chars_per_line})",
            f"Timing trim: {pipeline.timing_trim_seconds}s | Offset: {pipeline.timing_offset_seconds}s | Minimum duration: {pipeline.min_line_duration}s",
            f"Trim punctuation: {_fmt_bool(pipeline.remove_trailing_punctuation)} | CJK spacing: {_fmt_bool(pipeline.normalize_cjk_spacing)}",
            f"Refine English export: {_fmt_bool(pipeline.refine_source_subtitles)}",
        ]
        console.print(Panel("\\n".join(summary_lines), title="Pipeline & output", border_style="cyan"))

        options = [
            ("Set output format", "format"),
            ("Set audio format", "audio"),
            ("Set output directory", "dir"),
            ("Toggle keep temp audio", "keep"),
            ("Toggle save English subtitles", "save"),
            ("Set max characters per line", "max_line"),
            ("Set min characters per line", "min_line"),
            ("Set translated max characters", "tmax"),
            ("Set translated min characters", "tmin"),
            ("Set timing trim seconds", "trim"),
            ("Set timing offset seconds", "offset"),
            ("Set minimum line duration", "duration"),
            ("Toggle remove trailing punctuation", "punct"),
            ("Toggle CJK-Latin spacing", "spacing"),
            ("Toggle refine English export", "refine_en"),
            ("Edit translation system prompt", "prompt"),
        ]
        for idx, (label, _) in enumerate(options, start=1):
            console.print(f"{idx}. {label}")
        console.print("0. Back")
        choice = Prompt.ask("Select an option", default="0")
        if choice == "0":
            return
        try:
            _, key = options[int(choice) - 1]
        except (ValueError, IndexError):
            _wait_for_enter("Selection out of range. Press Enter to continue...")
            continue

        if key == "format":
            pipeline.output_format = Prompt.ask(
                "Output format",
                choices=["srt", "vtt"],
                default=pipeline.output_format,
            )
        elif key == "audio":
            pipeline.audio_format = Prompt.ask(
                "Intermediate audio format",
                choices=["wav", "mp3", "flac", "m4a", "ogg"],
                default=pipeline.audio_format,
            )
        elif key == "dir":
            pipeline.output_dir = Prompt.ask(
                "Output directory",
                default=pipeline.output_dir,
                show_choices=False,
            )
        elif key == "keep":
            pipeline.keep_temp_audio = Confirm.ask(
                "Keep intermediate audio file?",
                default=pipeline.keep_temp_audio,
            )
        elif key == "save":
            pipeline.save_source_subtitles = Confirm.ask(
                "Save English subtitles?",
                default=pipeline.save_source_subtitles,
            )
        elif key == "max_line":
            pipeline.max_chars_per_line = _prompt_int(
                "Max characters per subtitle line",
                default=pipeline.max_chars_per_line,
                minimum=20,
                maximum=160,
            )
            if pipeline.min_chars_per_line > pipeline.max_chars_per_line:
                pipeline.min_chars_per_line = pipeline.max_chars_per_line
        elif key == "min_line":
            pipeline.min_chars_per_line = _prompt_int(
                "Min characters per subtitle line",
                default=pipeline.min_chars_per_line,
                minimum=10,
                maximum=pipeline.max_chars_per_line,
            )
        elif key == "tmax":
            pipeline.translation_max_chars_per_line = _prompt_int(
                "Max characters per translated line",
                default=pipeline.translation_max_chars_per_line or 36,
                minimum=10,
                maximum=160,
            )
            if (
                pipeline.translation_min_chars_per_line
                and pipeline.translation_min_chars_per_line > pipeline.translation_max_chars_per_line
            ):
                pipeline.translation_min_chars_per_line = pipeline.translation_max_chars_per_line
        elif key == "tmin":
            max_chars = pipeline.translation_max_chars_per_line or 160
            pipeline.translation_min_chars_per_line = _prompt_int(
                "Min characters per translated line",
                default=pipeline.translation_min_chars_per_line or 4,
                minimum=1,
                maximum=max_chars,
            )
        elif key == "trim":
            pipeline.timing_trim_seconds = _prompt_float(
                "Timing trim seconds",
                pipeline.timing_trim_seconds,
            )
        elif key == "offset":
            pipeline.timing_offset_seconds = _prompt_float(
                "Timing offset seconds (positive delays subtitles)",
                pipeline.timing_offset_seconds,
            )
        elif key == "duration":
            pipeline.min_line_duration = _prompt_float(
                "Minimum line duration (seconds)",
                pipeline.min_line_duration,
            )
        elif key == "punct":
            pipeline.remove_trailing_punctuation = Confirm.ask(
                "Remove trailing punctuation?",
                default=pipeline.remove_trailing_punctuation,
            )
        elif key == "spacing":
            pipeline.normalize_cjk_spacing = Confirm.ask(
                "Insert spaces between Chinese characters and Latin/digit text?",
                default=pipeline.normalize_cjk_spacing,
            )
        elif key == "refine_en":
            pipeline.refine_source_subtitles = Confirm.ask(
                "Re-refine English subtitles when exporting?",
                default=pipeline.refine_source_subtitles,
            )
        elif key == "prompt":
            edited = typer.edit(pipeline.prompt_preamble + "\n")
            pipeline.prompt_preamble = (edited or pipeline.prompt_preamble).strip()

def _fmt_bool(value: bool, true_label: str = "yes", false_label: str = "no") -> str:
    return true_label if value else false_label


def _prompt_float(question: str, default: float) -> float:
    while True:
        raw = Prompt.ask(question, default=str(default), show_choices=False)
        try:
            return float(raw)
        except ValueError:
            console.print("[red]Please enter a numeric value.[/]")


def _prompt_float_optional(question: str, current: Optional[float]) -> Optional[float]:
    default = "" if current is None else str(current)
    while True:
        raw = Prompt.ask(question, default=default, show_choices=False).strip()
        if not raw:
            return current
        if raw.lower() in {"none", "null"}:
            return None
        try:
            return float(raw)
        except ValueError:
            console.print("[red]Please enter a numeric value or 'none'.[/]")


def _prompt_int(question: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        raw = Prompt.ask(question, default=str(default))
        try:
            value = int(raw)
        except ValueError:
            console.print("[red]Please enter an integer value.[/]")
            continue
        if value < minimum or value > maximum:
            console.print(
                f"[red]Value must be between {minimum} and {maximum}.[/]"
            )
            continue
        return value


def _prompt_optional_int(
    question: str,
    minimum: int,
) -> Optional[int]:
    raw = Prompt.ask(question, default="")
    if not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        console.print("[red]Please enter an integer or leave blank.[/]")
        return _prompt_optional_int(question, minimum)
    if value < minimum:
        console.print(f"[red]Value must be at least {minimum}.[/]")
        return _prompt_optional_int(question, minimum)
    return value


def _prompt_non_empty(question: str, default: str) -> str:
    while True:
        raw = Prompt.ask(question, default=default)
        if raw.strip():
            return raw
        console.print("[red]This field cannot be empty. Please try again.[/]")


def _prompt_json_dict(question: str, default: str) -> dict:
    while True:
        raw = Prompt.ask(question, default=default)
        if not raw.strip():
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            console.print("[red]Please enter a valid JSON object.[/]")
            continue
        if not isinstance(value, dict):
            console.print("[red]A JSON object (key/value pairs) is required.[/]")
            continue
        return value


def _write_document(
    document: SubtitleDocument,
    target_dir: Path,
    stem: str,
    suffix: str,
    output_format: str,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{stem}{suffix}.{output_format}"
    output_path = target_dir / filename
    if output_format == "srt":
        output_path.write_text(document.to_srt(), encoding="utf-8")
    elif output_format == "vtt":
        output_path.write_text(document.to_vtt(), encoding="utf-8")
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unsupported output format: {output_format}")
    return output_path


def _language_suffix(language: Optional[str]) -> str:
    if not language:
        return ""
    normalized = language.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.strip("._")
    if not normalized:
        return ""
    return f".{normalized}"


def _cleanup_audio_file(work_dir: Path, audio_path: Optional[Path]) -> None:
    if not audio_path:
        return
    try:
        audio_path.unlink(missing_ok=True)
        if not any(work_dir.iterdir()):
            work_dir.rmdir()
    except OSError:
        pass


def _offer_failure_cleanup(
    work_dir: Path,
    state: PipelineState,
    audio_path: Optional[Path],
    translations_path: Optional[Path],
    interrupted: bool,
) -> None:
    cached_paths = []
    current_segments = state.get_segments_path()
    if audio_path and audio_path.exists():
        cached_paths.append(audio_path)
    if current_segments and current_segments.exists():
        cached_paths.append(current_segments)
    if translations_path and translations_path.exists():
        cached_paths.append(translations_path)
    if state.path.exists():
        cached_paths.append(state.path)

    if not cached_paths:
        return

    prompt_text = "Pipeline did not finish. Clear cached data before retrying?"
    if interrupted:
        prompt_text = "Interrupted detected. Clear cached data now?"
    try:
        should_cleanup = Confirm.ask(prompt_text, default=False)
    except Exception:
        return

    if not should_cleanup:
        console.print("[yellow]Cached data kept. Resume the run after fixing the issue.[/]")
        return

    if audio_path and audio_path.exists():
        _cleanup_audio_file(work_dir, audio_path)
    if current_segments and current_segments.exists():
        try:
            current_segments.unlink()
        except OSError:
            pass
    if translations_path and translations_path.exists():
        try:
            translations_path.unlink()
        except OSError:
            pass
    state.clear()
    try:
        if work_dir.exists() and not any(work_dir.iterdir()):
            work_dir.rmdir()
    except OSError:
        pass
    console.print("[green]Cached data cleared.[/]")


if __name__ == "__main__":
    app()
