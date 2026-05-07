from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ConfigManager, TransubConfig


def load_batch_tasks(file_path: str) -> List[Dict[str, str]]:
    """Load batch tasks from CSV or JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {file_path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            tasks = json.load(f)
            if isinstance(tasks, list):
                return tasks
            return []

    if path.suffix.lower() == ".csv":
        tasks = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tasks.append(dict(row))
        return tasks

    raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .json")


def save_batch_status(file_path: str, tasks: List[Dict[str, str]]) -> None:
    """Save batch task status back to file."""
    path = Path(file_path)

    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        return

    if path.suffix.lower() == ".csv":
        if not tasks:
            return
        fieldnames = list(tasks[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tasks)
        return


def validate_batch_task(task: Dict[str, str]) -> Tuple[bool, str]:
    """Validate a single batch task."""
    video = task.get("video") or task.get("file") or task.get("path")
    if not video:
        return False, "Missing video file path"

    video_path = Path(video)
    if not video_path.exists():
        return False, f"Video file not found: {video}"

    return True, ""


def get_task_config(task: Dict[str, str], base_config: TransubConfig) -> TransubConfig:
    """Create a config override for a specific task."""
    config = base_config.model_copy(deep=True)

    if target_lang := task.get("target_language") or task.get("target"):
        config.llm.target_language = target_lang

    if source_lang := task.get("source_language") or task.get("source"):
        config.whisper.language = source_lang

    if output_dir := task.get("output_dir") or task.get("output"):
        config.pipeline.output_dir = output_dir

    return config


def get_task_video(task: Dict[str, str]) -> str:
    """Extract video path from task."""
    return task.get("video") or task.get("file") or task.get("path", "")


__all__ = [
    "load_batch_tasks",
    "save_batch_status",
    "validate_batch_task",
    "get_task_config",
    "get_task_video",
]
