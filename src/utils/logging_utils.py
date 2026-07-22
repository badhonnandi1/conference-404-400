"""Logging helpers for command-line preprocessing workflows."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: Path, level: str = "INFO", verbose: bool = False) -> logging.Logger:
    """Configure console and file logging, returning the project logger."""

    log_dir.mkdir(parents=True, exist_ok=True)
    numeric_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("video_authentication")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / "preprocessing.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
