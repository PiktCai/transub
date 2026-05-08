"""
HTTP backend for the transub desktop application.

Exposes the Python pipeline over localhost HTTP + SSE so the Electron
frontend can call it without spawning CLI subprocesses.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .auth import AuthManager
from .cache import clear_cache, get_cache_stats
from .config import ConfigManager, TransubConfig
from .optimize import optimize_subtitles
from .state import PipelineState, load_translation_progress, persist_translation_progress
from .subtitles import SubtitleDocument
from .transcribe import TranscriptionError, check_dependencies, prepare_transcription_model, transcribe_audio
from .translate import translate_subtitles

logger = logging.getLogger("transub.server")

_sse_subscribers: list[asyncio.Queue] = []
_active_pipeline: Optional[threading.Thread] = None
_active_pipeline_lock = threading.Lock()
_pipeline_status: dict = {"state": "idle"}
_pipeline_cancelled = False
_server_loop: Optional[asyncio.AbstractEventLoop] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _server_loop
    _server_loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="Transub", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _broadcast(event: str, data: dict) -> None:
    for q in _sse_subscribers:
        await q.put({"event": event, "data": data})


def _sync_broadcast(event: str, data: dict) -> None:
    if _server_loop and _server_loop.is_running():
        asyncio.run_coroutine_threadsafe(_broadcast(event, data), _server_loop)


def _set_pipeline_status(**updates: object) -> None:
    with _active_pipeline_lock:
        _pipeline_status.update(updates)


def _run_log_path(work_dir: Path, video: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in video.stem)
    return work_dir / "logs" / f"{timestamp}-{safe_stem}-{uuid4().hex[:8]}.log"


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamped = datetime.now().strftime("%H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamped}] {message}\n")


def _get_config(config_path: Optional[str] = None) -> TransubConfig:
    manager = ConfigManager(config_path or ConfigManager.default_path())
    if manager.exists():
        return manager.load()
    return TransubConfig()


def _get_auth_manager() -> AuthManager:
    return AuthManager()


class RunRequest(BaseModel):
    video_path: str
    transcribe_only: bool = False
    config_path: Optional[str] = None
    work_dir: Optional[str] = None


class ConfigUpdate(BaseModel):
    config: dict
    config_path: Optional[str] = None


class AuthUpdate(BaseModel):
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class PrepareModelRequest(BaseModel):
    config_path: Optional[str] = None


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.3.0"}


@app.get("/api/status")
async def status():
    with _active_pipeline_lock:
        return dict(_pipeline_status)


@app.get("/api/config")
async def get_config(config_path: Optional[str] = None):
    config = _get_config(config_path)
    return config.model_dump(mode="json")


@app.put("/api/config")
async def update_config(body: ConfigUpdate):
    manager = ConfigManager(body.config_path or ConfigManager.default_path())
    try:
        config = TransubConfig.model_validate(body.config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    manager.save(config)
    return {"status": "ok"}


@app.get("/api/auth")
async def get_auth():
    manager = _get_auth_manager()
    all_auth = manager.load_all()
    result = {}
    for provider, data in all_auth.items():
        if isinstance(data, dict):
            result[provider] = {
                "has_key": bool(data.get("api_key")),
                "api_base": data.get("api_base", ""),
            }
    return result


@app.put("/api/auth/{provider}")
async def save_auth(provider: str, body: AuthUpdate):
    manager = _get_auth_manager()
    manager.save_provider(provider, api_key=body.api_key, api_base=body.api_base)
    return {"status": "ok"}


@app.delete("/api/auth/{provider}")
async def delete_auth(provider: str):
    manager = _get_auth_manager()
    manager.save_provider(provider, api_key="", api_base="")
    return {"status": "ok"}


@app.post("/api/prepare-model")
async def prepare_model(body: PrepareModelRequest):
    config = _get_config(body.config_path)
    try:
        check_dependencies(config.whisper)
        message = prepare_transcription_model(config.whisper)
        return {"status": "ok", "message": message}
    except TranscriptionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/run")
async def run_pipeline(body: RunRequest):
    global _active_pipeline, _pipeline_cancelled
    with _active_pipeline_lock:
        if _active_pipeline and _active_pipeline.is_alive():
            raise HTTPException(status_code=409, detail="A pipeline is already running.")
        _pipeline_cancelled = False

    video = Path(body.video_path)
    if not video.exists():
        raise HTTPException(status_code=400, detail=f"Video not found: {body.video_path}")

    config = _get_config(body.config_path)
    work_dir = Path(body.work_dir) if body.work_dir else Path.home() / ".cache" / "transub"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = _run_log_path(work_dir, video)
    run_id = uuid4().hex
    _append_log(log_path, f"Starting run {run_id}")
    _append_log(log_path, f"Input: {video}")
    _append_log(log_path, f"Mode: {'transcribe-only' if body.transcribe_only else 'full pipeline'}")

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(run_id, video, config, work_dir, body.transcribe_only, log_path),
        daemon=True,
    )
    with _active_pipeline_lock:
        _active_pipeline = thread
        _pipeline_status.update({
            "state": "running",
            "run_id": run_id,
            "video": str(video),
            "log_path": str(log_path),
            "progress": 0,
        })
    thread.start()

    return {"status": "started", "video": str(video), "run_id": run_id, "log_path": str(log_path)}


@app.post("/api/cancel")
async def cancel_pipeline():
    global _pipeline_cancelled
    _pipeline_cancelled = True
    return {"status": "cancelling"}


@app.get("/api/stream")
async def stream():
    global _server_loop
    _server_loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _sse_subscribers.append(queue)

    async def generate():
        try:
            while True:
                data = await queue.get()
                yield f"event: {data['event']}\ndata: {json.dumps(data['data'])}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/cache/stats")
async def cache_stats():
    return get_cache_stats()


@app.delete("/api/cache")
async def clear_cache_endpoint():
    count = clear_cache()
    return {"cleared": count}


def _run_pipeline_thread(
    run_id: str,
    video: Path,
    config: TransubConfig,
    work_dir: Path,
    transcribe_only: bool,
    log_path: Path,
) -> None:
    global _pipeline_cancelled, _active_pipeline

    def emit(event: str, data: dict, *, log: Optional[str] = None) -> None:
        payload = {"run_id": run_id, "log_path": str(log_path), **data}
        if log or data.get("message") or data.get("error"):
            _append_log(log_path, log or str(data.get("message") or data.get("error")))
        if event == "progress":
            _set_pipeline_status(
                state="running",
                stage=payload.get("stage"),
                message=payload.get("message"),
                progress=payload.get("percent", _pipeline_status.get("progress", 0)),
            )
        elif event == "done":
            _set_pipeline_status(
                state="done" if payload.get("success") else "error",
                message=payload.get("error") or payload.get("output_path"),
                progress=100 if payload.get("success") else _pipeline_status.get("progress", 0),
            )
        _sync_broadcast(event, payload)

    def check_cancelled() -> bool:
        return _pipeline_cancelled

    try:
        state_path = work_dir / f"{video.stem}_state.json"
        state = PipelineState.load(state_path, video)

        audio_path = state.get_audio_path()
        segments_path = state.get_segments_path()

        if check_cancelled():
            emit("done", {"success": False, "error": "Cancelled"})
            return

        if not (audio_path and audio_path.exists()):
            emit("progress", {"stage": "extracting", "message": "Extracting audio...", "percent": 8})
            from .audio import extract_audio
            audio_path = extract_audio(video, config.pipeline, work_dir)
            audio_path = audio_path.resolve()
            state.set_audio_path(audio_path)
            emit("progress", {"stage": "extracting", "message": "Audio extracted.", "percent": 18})

        if check_cancelled():
            emit("done", {"success": False, "error": "Cancelled"})
            return

        if segments_path and segments_path.exists():
            with segments_path.open("r", encoding="utf-8") as fh:
                segment_payload = json.load(fh)
            source_doc = SubtitleDocument.from_serialized(segment_payload)
            emit("progress", {"stage": "transcribing", "message": "Loaded cached transcription.", "percent": 50})
        else:
            emit("progress", {"stage": "transcribing", "message": "Transcribing audio...", "percent": 24})

            def handle_transcription_progress(update: Dict[str, object]) -> None:
                duration = float(update.get("duration") or 0.0)
                position = float(update.get("position") or 0.0)
                segment_count = int(update.get("segment_count") or 0)
                if duration > 0:
                    percent = 24 + min(24, round((position / duration) * 24))
                    message = (
                        f"Transcribed {segment_count} segments "
                        f"({format_seconds(position)} / {format_seconds(duration)})."
                    )
                else:
                    percent = min(48, 24 + min(segment_count, 24))
                    message = f"Transcribed {segment_count} segments."
                emit("progress", {"stage": "transcribing", "message": message, "percent": percent})

            raw_doc = transcribe_audio(audio_path, config.whisper, progress_callback=handle_transcription_progress)

            if check_cancelled():
                emit("done", {"success": False, "error": "Cancelled"})
                return

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

            emit("progress", {"stage": "optimizing", "message": "Optimizing transcription...", "percent": 42})
            try:
                refined_doc = optimize_subtitles(refined_doc, config.llm, config.pipeline, mode="asr")
            except Exception as e:
                _append_log(log_path, f"ASR optimization skipped: {e}")

            segments_path = work_dir / f"{video.stem}_segments.json"
            segments_path.write_text(
                json.dumps(refined_doc.to_serializable(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            state.mark_transcription(segments_path, len(refined_doc.lines))
            source_doc = refined_doc
            emit("progress", {"stage": "transcribing", "message": f"Transcription complete ({len(refined_doc.lines)} lines).", "percent": 52})

        output_dir = (
            Path(config.pipeline.output_dir)
            if config.pipeline.output_dir is not None
            else video.parent
        )

        if transcribe_only:
            source_suffix = _source_language_suffix(config.whisper.language)
            transcript_path = _write_document(source_doc, output_dir, video.stem, source_suffix, config.pipeline.output_format)
            emit("done", {"success": True, "output_path": str(transcript_path), "lines": len(source_doc.lines)})
            return

        if check_cancelled():
            emit("done", {"success": False, "error": "Cancelled"})
            return

        translations_path = state.translation_progress_path(work_dir / f"{video.stem}_translations.json")
        existing_translations = load_translation_progress(translations_path)

        def handle_progress(new_items: Dict[str, str]) -> None:
            existing_translations.update(new_items)
            persist_translation_progress(translations_path, existing_translations)
            state.mark_lines_completed(new_items.keys())
            emit("progress", {
                "stage": "translating",
                "message": f"Translated {len(existing_translations)}/{len(source_doc.lines)} lines.",
                "done": len(existing_translations),
                "total": len(source_doc.lines),
                "percent": 58 + round((len(existing_translations) / max(len(source_doc.lines), 1)) * 28),
            })

        emit("progress", {"stage": "translating", "message": "Translating subtitles...", "percent": 58})
        translated_doc, usage_stats = translate_subtitles(
            source_doc, config.llm, config.pipeline,
            existing_translations=existing_translations,
            progress_callback=handle_progress,
        )

        if check_cancelled():
            emit("done", {"success": False, "error": "Cancelled"})
            return

        emit("progress", {"stage": "polishing", "message": "Polishing translation...", "percent": 88})
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

        try:
            output_doc = optimize_subtitles(output_doc, config.llm, config.pipeline, mode="polish")
        except Exception as e:
            _append_log(log_path, f"Translation polishing skipped: {e}")

        if config.pipeline.remove_trailing_punctuation:
            output_doc = output_doc.remove_trailing_punctuation()
        if config.pipeline.normalize_cjk_spacing:
            output_doc = output_doc.normalize_cjk_spacing()

        output_suffix = _language_suffix(config.llm.target_language)
        output_path = _write_document(output_doc, output_dir, video.stem, output_suffix, config.pipeline.output_format)

        if config.pipeline.save_source_subtitles:
            source_suffix = _source_language_suffix(config.whisper.language)
            _write_document(source_doc, output_dir, video.stem, source_suffix, config.pipeline.output_format)

        state.clear()
        emit("done", {
            "success": True,
            "output_path": str(output_path),
            "lines": len(output_doc.lines),
            "tokens": usage_stats,
        })

    except Exception as e:
        logger.exception("Pipeline failed")
        emit("done", {"success": False, "error": str(e)})
    finally:
        with _active_pipeline_lock:
            _active_pipeline = None


def _write_document(document: SubtitleDocument, target_dir: Path, stem: str, suffix: str, fmt: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{stem}{suffix}.{fmt}"
    content = document.to_srt() if fmt == "srt" else document.to_vtt()
    path.write_text(content, encoding="utf-8")
    return path


def _source_language_suffix(language: Optional[str]) -> str:
    if not language or language == "auto":
        return ".src"
    return f".{language}"


def _language_suffix(language: str) -> str:
    if not language or language == "auto":
        return ""
    return f".{language}"


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remaining:02d}"
    return f"{minutes:d}:{remaining:02d}"


def start_server(host: str = "127.0.0.1", port: int = 18789) -> None:
    import uvicorn
    logger.info("Starting Transub server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
