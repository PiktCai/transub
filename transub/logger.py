from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "transub"


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure rotating file logging for the CLI run."""

    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger() -> logging.Logger:
    """Return the configured Transub logger."""

    return logging.getLogger(_LOGGER_NAME)


__all__ = ["setup_logging", "get_logger"]

