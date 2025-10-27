from __future__ import annotations

import json
import shlex
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from importlib import metadata

import typer
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.progress import SpinnerColumn, Progress, TextColumn, TaskProgressColumn
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from .audio import AudioExtractionError, extract_audio
from .config import ConfigManager, TransubConfig, DEFAULT_TRANSLATION_PROMPT
from .logger import setup_logging
from .state import (
    PipelineState,
    load_translation_progress,
    persist_translation_progress,
)
from .subtitles import SubtitleDocument
from .transcribe import (
    TranscriptionError,
    transcribe_audio,
    check_dependencies,
    DEFAULT_OPENAI_TRANSCRIBE_URL,
)
from .translate import LLMTranslationError, translate_subtitles

THEME_COLOR = "#33c9b2"
THEME_VARIABLE_COLOR = "#f0b429"
THEME_VARIABLE_STYLE = f"bold {THEME_VARIABLE_COLOR}"
THEME_HIGHLIGHT = f"bold {THEME_COLOR}"

app = typer.Typer(add_completion=False, help="Transcribe and translate subtitles from videos.")
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


def _get_version() -> str:
    try:
        return metadata.version("transub")
    except metadata.PackageNotFoundError:  # pragma: no cover - dev installs
        return "unknown"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show Transub version and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        console.print(f"Transub { _get_version() }")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(app.get_help())
def _print_header(
    *,
    subtitle: str | None = None,
    video: Path | None = None,
    body: RenderableType | None = None,
) -> None:
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")

    title = Text("Transub", style=THEME_HIGHLIGHT)
    if subtitle:
        title.append("  ·  ", style="dim")
        title.append(subtitle, style="white")
    tagline = Text("video → subtitles (transcribe + translate)", style="bright_black")

    header.add_row(title, tagline)

    renderables: List[RenderableType] = [header]

    if video:
        video_table = _key_value_table([("source", str(video))])
        renderables.append(video_table)

    if body is not None:
        renderables.append(body)

    panel = Panel(
        Group(*renderables),
        border_style=THEME_COLOR,
        padding=(1, 2),
    )
    console.print(panel)


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
    _print_header(subtitle="configuration wizard")
    _run_wizard(manager, allow_overwrite=True)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    video: Path = typer.Argument(..., exists=True, readable=True, help="Path to the video file"),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Custom configuration file location"
    ),
    work_dir: Optional[Path] = typer.Option(
        None,
        "--work-dir",
        help="Working directory for intermediate files (defaults to ~/.cache/transub).",
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
    check_dependencies(config.whisper)

    work_dir = (work_dir or Path.home() / ".cache" / "transub").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    config_display = (config_path or ConfigManager.default_path()).resolve()
    mode_label = "transcribe-only" if transcribe_only else "full pipeline"
    source_lang = config.whisper.language or "auto"
    target_lang = config.llm.target_language or "auto"
    source_text = Text(source_lang, style=THEME_VARIABLE_STYLE)
    target_text = Text(target_lang, style=THEME_VARIABLE_STYLE)
    run_snapshot = _key_value_table(
        [
            ("config", str(config_display)),
            ("work_dir", str(work_dir)),
            ("mode", mode_label),
            ("source language", source_text),
            ("target language", target_text),
        ]
    )

    _print_header(subtitle="pipeline", video=video, body=run_snapshot)

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
                max_width=config.pipeline.max_display_width,
                min_width=config.pipeline.min_display_width,
                min_duration=config.pipeline.min_line_duration,
                max_cps=config.pipeline.max_cps,
                pause_threshold=config.pipeline.pause_threshold_seconds,
                silence_threshold=config.pipeline.silence_threshold_seconds,
                remove_silence=config.pipeline.remove_silence_segments,
                prefer_sentence_boundaries=config.pipeline.prefer_sentence_boundaries,
            )
            if config.pipeline.timing_offset_seconds != 0:
                refined_doc = refined_doc.apply_offset(config.pipeline.timing_offset_seconds)
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

        output_dir = (
            Path(config.pipeline.output_dir)
            if config.pipeline.output_dir is not None
            else video.parent
        )

        if transcribe_only:
            console.print(f"[{THEME_COLOR}]Transcribe-only mode: skipping translation.[/]")
            logger.info("Transcribe-only mode enabled; skipping translation stage.")

            transcript_doc = source_doc
            # Refine source subtitles if requested
            if config.pipeline.refine_source_subtitles:
                transcript_doc = source_doc.refine(
                    max_width=config.pipeline.translation_max_display_width or 30.0,
                    min_width=config.pipeline.translation_min_display_width or 15.0,
                    min_duration=config.pipeline.min_line_duration,
                    max_cps=config.pipeline.max_cps,
                    pause_threshold=config.pipeline.pause_threshold_seconds,
                    silence_threshold=config.pipeline.silence_threshold_seconds,
                    remove_silence=config.pipeline.remove_silence_segments,
                    prefer_sentence_boundaries=config.pipeline.prefer_sentence_boundaries,
                )

            source_suffix = _source_language_suffix(config.whisper.language)
            transcript_path = _write_document(
                document=transcript_doc,
                target_dir=output_dir,
                stem=video.stem,
                suffix=source_suffix,
                output_format=config.pipeline.output_format,
            )
            console.print(Panel.fit(f"✅ Transcription saved to {transcript_path}", style="green"))
            logger.info("Transcription exported to %s", transcript_path)

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
            return f"[{THEME_HIGHLIGHT}]Translating[/] {done}/{total_lines}"

        progress = Progress(
            SpinnerColumn(style=THEME_COLOR),
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
        
        # Refine translated subtitles using display width
        output_doc = translated_doc.refine(
            max_width=config.pipeline.translation_max_display_width or 30.0,
            min_width=config.pipeline.translation_min_display_width or 15.0,
            min_duration=config.pipeline.min_line_duration,
            max_cps=config.pipeline.max_cps,
            pause_threshold=config.pipeline.pause_threshold_seconds,
            silence_threshold=config.pipeline.silence_threshold_seconds,
            remove_silence=config.pipeline.remove_silence_segments,
            prefer_sentence_boundaries=config.pipeline.prefer_sentence_boundaries,
        )

        if config.pipeline.remove_trailing_punctuation:
            output_doc = output_doc.remove_trailing_punctuation()
        if config.pipeline.simplify_cjk_punctuation:
            output_doc = output_doc.simplify_cjk_punctuation()
        if config.pipeline.normalize_cjk_spacing:
            output_doc = output_doc.normalize_cjk_spacing()

        output_suffix = _language_suffix(config.llm.target_language)
        output_path = _write_document(
            document=output_doc,
            target_dir=output_dir,
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
            if config.pipeline.refine_source_subtitles:
                source_output_doc = source_doc.refine(
                    max_width=config.pipeline.translation_max_display_width or 30.0,
                    min_width=config.pipeline.translation_min_display_width or 15.0,
                    min_duration=config.pipeline.min_line_duration,
                    max_cps=config.pipeline.max_cps,
                    pause_threshold=config.pipeline.pause_threshold_seconds,
                    silence_threshold=config.pipeline.silence_threshold_seconds,
                    remove_silence=config.pipeline.remove_silence_segments,
                    prefer_sentence_boundaries=config.pipeline.prefer_sentence_boundaries,
                )
            source_suffix = _source_language_suffix(config.whisper.language)
            source_path = _write_document(
                document=source_output_doc,
                target_dir=output_dir,
                stem=video.stem,
                suffix=source_suffix,
                output_format=config.pipeline.output_format,
            )
            console.print(f"Source subtitles saved to [italic]{source_path}[/]")
            logger.info("Source subtitles saved to %s", source_path)

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

    dirty = False
    while True:
        console.clear()
        _print_header(
            subtitle="configuration editor",
            body=_config_summary_table(config, manager.path),
        )
        console.print()

        console.print("[bold]Select an option:[/]")
        menu = Table(
            show_header=False,
            box=box.SIMPLE_HEAD,
            expand=False,
            padding=(0, 1),
        )
        menu.add_column(style=THEME_HIGHLIGHT, justify="center", width=4)
        menu.add_column(style="white", justify="left")
        menu.add_row("0", "Save & exit")
        menu.add_row("Q", "Exit without saving")
        for idx, (label, _) in enumerate(option_handlers, start=1):
            menu.add_row(str(idx), label)
        console.print(menu)

        choice = Prompt.ask("Your choice", default="0").strip()
        if not choice:
            choice = "0"

        if choice.lower() in {"q", "quit", "exit"}:
            if dirty and not Confirm.ask("Discard unsaved changes?", default=False):
                continue
            console.clear()
            console.print(Panel.fit("Changes discarded.", style="yellow"))
            return

        if choice == "0":
            manager.save(config)
            dirty = False
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
            console.print(
                Panel(
                    json.dumps(
                        config.model_dump(mode="json"),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    title="Raw configuration",
                )
            )
            _wait_for_enter()
        else:
            handler(config)
            dirty = True


BACK_TOKENS = {"back", "b"}


class BackAction(Exception):
    """Signal that the wizard should move back one step."""


@dataclass
class WizardStep:
    key: str
    heading: str
    description: Callable[[TransubConfig], Optional[str]]
    handler: Callable[[TransubConfig], None]
    condition: Callable[[TransubConfig], bool] = lambda _: True


def _is_back(value: str) -> bool:
    return value.strip().lower() in BACK_TOKENS



def _wizard_step_body(title: str, description: Optional[str]) -> RenderableType:
    lines: List[RenderableType] = [Text(title, style="bold white")]
    if description:
        lines.append(Text(description, style="white"))
    lines.append(Text("Type 'back' to return to the previous step.", style="dim"))
    return Padding(Group(*lines), (0, 0, 0, 0))


def _wizard_ask_choice(prompt_text: str, choices: List[str], default: str) -> str:
    choice_display = "/".join(choices)
    normalized = [choice.lower() for choice in choices]
    while True:
        raw = Prompt.ask(
            f"{prompt_text} ({choice_display})",
            default=default,
            show_choices=False,
        )
        if raw is None:
            raw = ""
        if _is_back(raw):
            raise BackAction()
        value = raw.strip()
        if value.lower() in normalized:
            return choices[normalized.index(value.lower())]
        console.print(
            "[red]Please choose one of the listed options, or type 'back' to return.[/]"
        )


def _wizard_ask_text(
    prompt_text: str,
    *,
    default: str | None = None,
    allow_blank: bool = True,
    to_none: bool = False,
) -> Optional[str]:
    while True:
        raw = Prompt.ask(
            f"{prompt_text}",
            default=default,
            show_choices=False,
        )
        if raw is None:
            raw = ""
        if _is_back(raw):
            raise BackAction()
        value = raw.strip()
        if not value and not allow_blank:
            console.print("[red]This field cannot be empty.[/]")
            continue
        if to_none:
            return value or None
        return value if value or allow_blank else default


def _wizard_ask_bool(prompt_text: str, default: bool) -> bool:
    default_hint = "Y/n" if default else "y/N"
    default_value = "y" if default else "n"
    while True:
        raw = Prompt.ask(
            f"{prompt_text} ({default_hint})",
            default=default_value,
            show_choices=False,
        )
        if raw is None:
            raw = default_value
        if _is_back(raw):
            raise BackAction()
        value = raw.strip().lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        if not value:
            return default
        console.print("[red]Please enter y or n.[/]")


def _wizard_ask_int(
    prompt_text: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    while True:
        raw = Prompt.ask(
            f"{prompt_text} [{minimum}-{maximum}]",
            default=str(default),
            show_choices=False,
        )
        if raw is None:
            raw = str(default)
        if _is_back(raw):
            raise BackAction()
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


def _wizard_ask_optional_int(
    prompt_text: str,
    *,
    current: Optional[int],
    minimum: int,
) -> Optional[int]:
    default = "" if current is None else str(current)
    while True:
        raw = Prompt.ask(
            f"{prompt_text} (leave blank for auto)",
            default=default,
            show_choices=False,
        )
        if raw is None:
            raw = ""
        if _is_back(raw):
            raise BackAction()
        value = raw.strip()
        if not value:
            return None
        try:
            parsed = int(value)
        except ValueError:
            console.print("[red]Please enter an integer or leave blank.[/]")
            continue
        if parsed < minimum:
            console.print(f"[red]Value must be at least {minimum}.[/]")
            continue
        return parsed


def _wizard_ask_json_dict(
    prompt_text: str,
    *,
    default: dict,
) -> dict:
    default_text = json.dumps(default or {}, ensure_ascii=False)
    while True:
        raw = Prompt.ask(
            f"{prompt_text} (JSON)",
            default=default_text,
            show_choices=False,
        )
        if raw is None:
            raw = default_text
        if _is_back(raw):
            raise BackAction()
        value = raw.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            console.print("[red]Please enter a valid JSON object.[/]")
            continue
        if not isinstance(parsed, dict):
            console.print("[red]The value must be a JSON object (key/value pairs).[/]")
            continue
        return parsed


def _wizard_step_whisper_backend(config: TransubConfig) -> None:
    choices = ["local", "api", "cpp", "mlx"]
    backend = _wizard_ask_choice(
        "Select backend",
        choices,
        config.whisper.backend,
    )
    config.whisper.backend = backend


def _wizard_step_whisper_model(config: TransubConfig) -> None:
    suggestions = WHISPER_MODEL_SUGGESTIONS.get(config.whisper.backend, [])
    default_model = config.whisper.model or (suggestions[0] if suggestions else "base")
    model = _wizard_ask_text(
        "Model ID or path",
        default=default_model,
        allow_blank=False,
    )
    assert model
    config.whisper.model = model


def _wizard_step_whisper_device(config: TransubConfig) -> None:
    device = _wizard_ask_text(
        "Preferred device",
        default=config.whisper.device or "",
        to_none=True,
    )
    config.whisper.device = device


def _wizard_step_whisper_language(config: TransubConfig) -> None:
    language = _wizard_ask_text(
        "Source language",
        default=config.whisper.language or "en",
        to_none=True,
    )
    config.whisper.language = language


def _wizard_step_whisper_segmentation(config: TransubConfig) -> None:
    config.whisper.tune_segmentation = _wizard_ask_bool(
        "Enable segmentation tuning?",
        config.whisper.tune_segmentation,
    )


def _wizard_step_whisper_initial_prompt(config: TransubConfig) -> None:
    initial_prompt = _wizard_ask_text(
        "Initial prompt for Whisper (optional)",
        default=config.whisper.initial_prompt or "",
        to_none=True,
    )
    config.whisper.initial_prompt = initial_prompt


def _wizard_step_whisper_api(config: TransubConfig) -> None:
    default_url = config.whisper.api_url or DEFAULT_OPENAI_TRANSCRIBE_URL
    api_url = _wizard_ask_text(
        "API URL",
        default=default_url,
        allow_blank=False,
    )
    api_key_env = _wizard_ask_text(
        "API key environment variable",
        default=config.whisper.api_key_env,
        allow_blank=False,
    )
    config.whisper.api_url = api_url
    config.whisper.api_key_env = api_key_env


def _wizard_step_whisper_cpp(config: TransubConfig) -> None:
    config.whisper.cpp_binary = _wizard_ask_text(
        "whisper.cpp executable",
        default=config.whisper.cpp_binary,
        allow_blank=False,
    )
    config.whisper.cpp_model_path = _wizard_ask_text(
        "Model file path (.bin/.gguf)",
        default=config.whisper.cpp_model_path or "",
        allow_blank=False,
    )
    config.whisper.cpp_threads = _wizard_ask_optional_int(
        "Thread count",
        current=config.whisper.cpp_threads,
        minimum=1,
    )
    extra_args = _wizard_ask_text(
        "Additional arguments (space separated)",
        default=" ".join(config.whisper.cpp_extra_args) if config.whisper.cpp_extra_args else "",
    )
    config.whisper.cpp_extra_args = shlex.split(extra_args) if extra_args else []


def _wizard_step_whisper_mlx(config: TransubConfig) -> None:
    config.whisper.mlx_model_dir = _wizard_ask_text(
        "Model directory",
        default=config.whisper.mlx_model_dir or "",
        to_none=True,
    )
    config.whisper.mlx_dtype = _wizard_ask_text(
        "dtype (auto/float16/float32)",
        default=config.whisper.mlx_dtype or "",
        to_none=True,
    )
    config.whisper.mlx_device = _wizard_ask_text(
        "Device (auto/mps/cpu)",
        default=config.whisper.mlx_device or "",
        to_none=True,
    )
    extra_args = _wizard_ask_json_dict(
        "Extra arguments",
        default=config.whisper.mlx_extra_args or {},
    )
    config.whisper.mlx_extra_args = extra_args


def _wizard_step_llm_provider(config: TransubConfig) -> None:
    config.llm.provider = _wizard_ask_text(
        "LLM provider",
        default=config.llm.provider,
        allow_blank=False,
    )
    config.llm.model = _wizard_ask_text(
        "LLM model",
        default=config.llm.model,
        allow_blank=False,
    )
    api_base = _wizard_ask_text(
        "API base URL (leave blank for default)",
        default=config.llm.api_base or "",
        to_none=True,
    )
    config.llm.api_base = api_base
    config.llm.api_key_env = _wizard_ask_text(
        "API key environment variable",
        default=config.llm.api_key_env,
        allow_blank=False,
    )


def _wizard_step_llm_batch_and_style(config: TransubConfig) -> None:
    config.llm.batch_size = _wizard_ask_int(
        "Lines per translation batch",
        default=config.llm.batch_size,
        minimum=1,
        maximum=50,
    )
    config.llm.target_language = _wizard_ask_text(
        "Target language",
        default=config.llm.target_language,
        allow_blank=False,
    )
    style = _wizard_ask_text(
        "Style description (optional)",
        default=config.llm.style or "",
        to_none=True,
    )
    config.llm.style = style


def _wizard_step_pipeline_output(config: TransubConfig) -> None:
    config.pipeline.output_format = _wizard_ask_choice(
        "Subtitle format",
        ["srt", "vtt"],
        config.pipeline.output_format,
    )
    config.pipeline.audio_format = _wizard_ask_choice(
        "Intermediate audio format",
        ["wav", "mp3", "flac", "m4a", "ogg"],
        config.pipeline.audio_format,
    )
    output_dir_input = _wizard_ask_text(
        "Output directory (leave blank to use video file directory)",
        default=config.pipeline.output_dir or "",
        allow_blank=True,
    )
    config.pipeline.output_dir = output_dir_input if output_dir_input else None
    config.pipeline.keep_temp_audio = _wizard_ask_bool(
        "Keep extracted audio file?",
        config.pipeline.keep_temp_audio,
    )
    config.pipeline.save_source_subtitles = _wizard_ask_bool(
        "Save source language subtitles?",
        config.pipeline.save_source_subtitles,
    )
    config.pipeline.simplify_cjk_punctuation = _wizard_ask_bool(
        "Simplify CJK punctuation? (Replace commas/periods with spaces, common for Chinese subtitles)",
        config.pipeline.simplify_cjk_punctuation,
    )




def _wizard_step_prompt(config: TransubConfig) -> None:
    if _wizard_ask_bool("Open an editor to modify the translation prompt?", False):
        edited = typer.edit(config.pipeline.prompt_preamble + "\n")
        if edited:
            config.pipeline.prompt_preamble = edited.strip()
        return
    custom = _wizard_ask_text(
        "Custom system prompt (leave blank to keep default)",
        default=config.pipeline.prompt_preamble,
        allow_blank=True,
    )
    if custom and custom.strip():
        config.pipeline.prompt_preamble = custom.strip()


def _prompt_for_config() -> TransubConfig:
    config = TransubConfig()
    steps: List[WizardStep] = [
        WizardStep(
            "whisper-backend",
            "Whisper Backend",
            lambda _cfg: "Choose where Whisper will run.",
            _wizard_step_whisper_backend,
        ),
        WizardStep(
            "whisper-model",
            "Whisper Model",
            lambda cfg: (
                "Suggested models: " + ", ".join(WHISPER_MODEL_SUGGESTIONS.get(cfg.whisper.backend, []))
            )
            if WHISPER_MODEL_SUGGESTIONS.get(cfg.whisper.backend)
            else "Provide the model identifier or path.",
            _wizard_step_whisper_model,
        ),
        WizardStep(
            "whisper-device",
            "Compute Device",
            lambda _cfg: "Optional: cuda / cpu / mps. Leave blank to auto-detect.",
            _wizard_step_whisper_device,
        ),
        WizardStep(
            "whisper-language",
            "Language Hint",
            lambda _cfg: "Enter a language code (for example, en) or leave blank for auto detection.",
            _wizard_step_whisper_language,
        ),
        WizardStep(
            "whisper-tune",
            "Segmentation Tuning",
            lambda _cfg: "Enable this to reduce fragmented subtitles.",
            _wizard_step_whisper_segmentation,
        ),
        WizardStep(
            "whisper-initial",
            "Initial Prompt",
            lambda _cfg: "Optional text to help Whisper with terminology or role-playing.",
            _wizard_step_whisper_initial_prompt,
        ),
        WizardStep(
            "whisper-api",
            "Whisper API",
            lambda _cfg: "Configure your remote transcription endpoint.",
            _wizard_step_whisper_api,
            lambda cfg: cfg.whisper.backend == "api",
        ),
        WizardStep(
            "whisper-cpp",
            "whisper.cpp Options",
            lambda _cfg: "Specify the executable, model path, and extra parameters.",
            _wizard_step_whisper_cpp,
            lambda cfg: cfg.whisper.backend == "cpp",
        ),
        WizardStep(
            "whisper-mlx",
            "mlx-whisper Options",
            lambda _cfg: "Provide optional model directory and advanced arguments.",
            _wizard_step_whisper_mlx,
            lambda cfg: cfg.whisper.backend == "mlx",
        ),
        WizardStep(
            "llm-provider",
            "Translation Model",
            lambda _cfg: "Configure which LLM performs translation.",
            _wizard_step_llm_provider,
        ),
        WizardStep(
            "llm-batch",
            "Translation Parameters",
            lambda _cfg: "Adjust batch size, target language, and style.",
            _wizard_step_llm_batch_and_style,
        ),
        WizardStep(
            "pipeline-output",
            "Output Settings",
            lambda _cfg: "Pick subtitle format, output directory, and caching options.",
            _wizard_step_pipeline_output,
        ),
        WizardStep(
            "prompt",
            "Translation Prompt",
            lambda _cfg: "Edit the system prompt or keep the default template.",
            _wizard_step_prompt,
        ),
    ]

    idx = 0
    history: List[int] = []

    while idx < len(steps):
        step = steps[idx]
        if not step.condition(config):
            idx += 1
            continue

        active_indices = [i for i, s in enumerate(steps) if s.condition(config)]
        step_position = active_indices.index(idx) + 1
        total_steps = len(active_indices)

        console.clear()
        description = step.description(config)
        body = _wizard_step_body(step.heading, description)
        _print_header(subtitle=f"setup {step_position}/{total_steps}", body=body)

        try:
            step.handler(config)
            history.append(idx)
            idx += 1
        except BackAction:
            if history:
                idx = history.pop()
                continue
            console.print("[yellow]You are already at the first step; cannot go back further.[/]")
            _wait_for_enter()
            idx = 0

    return config


def _run_wizard(manager: ConfigManager, allow_overwrite: bool) -> None:
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
    console.print()
    console.print(
        "[dim]💡 Tip: You can adjust advanced options (display width, pause detection, etc.) "
        "anytime with [bold cyan]transub configure[/bold cyan][/dim]"
    )


def _key_value_table(rows: List[tuple[str, RenderableType]]) -> Table:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(style="white", ratio=1)
    for key, value in rows:
        table.add_row(key, value)
    return table


def _config_summary_table(config: TransubConfig, path: Path) -> Table:
    whisper = config.whisper
    llm = config.llm
    pipeline = config.pipeline
    rows = [
        ("config", str(path)),
        (
            "whisper",
            (
                f"backend={whisper.backend} | model={whisper.model} | "
                f"device={whisper.device or 'auto'} | language={whisper.language or 'auto'}"
            ),
        ),
        (
            "llm",
            (
                f"provider={llm.provider} | model={llm.model} | target={llm.target_language} | batch={llm.batch_size}"
            ),
        ),
        (
            "pipeline",
            (
                f"format={pipeline.output_format} | audio={pipeline.audio_format} | dir={pipeline.output_dir or '(video dir)'} | "
                f"keep_audio={_fmt_bool(pipeline.keep_temp_audio)} | save_en={_fmt_bool(pipeline.save_source_subtitles)}"
            ),
        ),
    ]
    return _key_value_table(rows)


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
            Panel("\\n".join(summary_lines), title="Whisper backend & model", border_style=THEME_COLOR)
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
            Panel("\\n".join(summary_lines), title="Whisper advanced parameters", border_style=THEME_COLOR)
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
        console.print(Panel("\\n".join(summary_lines), title="Translation LLM", border_style=THEME_COLOR))

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
            f"Output dir: {pipeline.output_dir or '(video file directory)'}",
            f"Keep temp audio: {_fmt_bool(pipeline.keep_temp_audio)} | Save source subtitles: {_fmt_bool(pipeline.save_source_subtitles)}",
            f"Source display width: {pipeline.max_display_width} (min {pipeline.min_display_width})",
            f"Translation display width: {pipeline.translation_max_display_width} (min {pipeline.translation_min_display_width})",
            f"Timing offset: {pipeline.timing_offset_seconds}s | Minimum duration: {pipeline.min_line_duration}s",
            f"Pause threshold: {pipeline.pause_threshold_seconds}s | Silence threshold: {pipeline.silence_threshold_seconds}s",
            f"Remove silence: {_fmt_bool(pipeline.remove_silence_segments)} | Prefer sentence boundaries: {_fmt_bool(pipeline.prefer_sentence_boundaries)}",
            f"Trim punctuation: {_fmt_bool(pipeline.remove_trailing_punctuation)} | Simplify CJK punctuation: {_fmt_bool(pipeline.simplify_cjk_punctuation)}",
            f"CJK spacing: {_fmt_bool(pipeline.normalize_cjk_spacing)} | Refine source export: {_fmt_bool(pipeline.refine_source_subtitles)}",
        ]
        console.print(Panel("\\n".join(summary_lines), title="Pipeline & output", border_style=THEME_COLOR))

        options = [
            ("Set output format", "format"),
            ("Set audio format", "audio"),
            ("Set output directory", "dir"),
            ("Toggle keep temp audio", "keep"),
            ("Toggle save source subtitles", "save"),
            ("Set max display width", "max_width"),
            ("Set min display width", "min_width"),
            ("Set translated max display width", "tmax_width"),
            ("Set translated min display width", "tmin_width"),
            ("Set pause threshold", "pause"),
            ("Set silence threshold", "silence"),
            ("Toggle remove silence segments", "remove_silence"),
            ("Toggle prefer sentence boundaries", "sentence_boundaries"),
            ("Set timing offset seconds", "offset"),
            ("Set minimum line duration", "duration"),
            ("Toggle remove trailing punctuation", "punct"),
            ("Toggle simplify CJK punctuation", "simplify_punct"),
            ("Toggle CJK-Latin spacing", "spacing"),
            ("Toggle refine source export", "refine_en"),
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
            dir_input = Prompt.ask(
                "Output directory (blank = video file directory)",
                default=pipeline.output_dir or "",
                show_choices=False,
            )
            pipeline.output_dir = dir_input if dir_input else None
        elif key == "keep":
            pipeline.keep_temp_audio = Confirm.ask(
                "Keep intermediate audio file?",
                default=pipeline.keep_temp_audio,
            )
        elif key == "save":
            pipeline.save_source_subtitles = Confirm.ask(
                "Save source subtitles?",
                default=pipeline.save_source_subtitles,
            )
        elif key == "max_width":
            pipeline.max_display_width = _prompt_float(
                "Max display width for source subtitles (42.0 = industry standard for English)",
                default=pipeline.max_display_width,
            )
            if pipeline.min_display_width > pipeline.max_display_width:
                pipeline.min_display_width = pipeline.max_display_width
        elif key == "min_width":
            pipeline.min_display_width = _prompt_float(
                "Min display width for source subtitles",
                default=pipeline.min_display_width,
            )
        elif key == "tmax_width":
            pipeline.translation_max_display_width = _prompt_float(
                "Max display width for translated subtitles (30.0 suitable for CJK)",
                default=pipeline.translation_max_display_width or 30.0,
            )
            if (
                pipeline.translation_min_display_width
                and pipeline.translation_min_display_width > pipeline.translation_max_display_width
            ):
                pipeline.translation_min_display_width = pipeline.translation_max_display_width
        elif key == "tmin_width":
            pipeline.translation_min_display_width = _prompt_float(
                "Min display width for translated subtitles",
                default=pipeline.translation_min_display_width or 15.0,
            )
        elif key == "pause":
            pipeline.pause_threshold_seconds = _prompt_float(
                "Pause threshold seconds (minimum gap to consider as natural pause)",
                pipeline.pause_threshold_seconds,
            )
        elif key == "silence":
            pipeline.silence_threshold_seconds = _prompt_float(
                "Silence threshold seconds (minimum gap to consider as silence)",
                pipeline.silence_threshold_seconds,
            )
        elif key == "remove_silence":
            pipeline.remove_silence_segments = Confirm.ask(
                "Remove silence segments?",
                default=pipeline.remove_silence_segments,
            )
        elif key == "sentence_boundaries":
            pipeline.prefer_sentence_boundaries = Confirm.ask(
                "Prefer splitting at sentence/phrase boundaries?",
                default=pipeline.prefer_sentence_boundaries,
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
        elif key == "simplify_punct":
            pipeline.simplify_cjk_punctuation = Confirm.ask(
                "Simplify CJK punctuation (replace commas and periods with spaces)?",
                default=pipeline.simplify_cjk_punctuation,
            )
        elif key == "spacing":
            pipeline.normalize_cjk_spacing = Confirm.ask(
                "Insert spaces between CJK characters and Latin/digit text?",
                default=pipeline.normalize_cjk_spacing,
            )
        elif key == "refine_en":
            pipeline.refine_source_subtitles = Confirm.ask(
                "Re-refine source subtitles when exporting?",
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


def _source_language_suffix(language: Optional[str]) -> str:
    if not language or language.strip().lower() in {"", "auto"}:
        return ".src"
    suffix = _language_suffix(language)
    return suffix or ".src"


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
