"""Application-wide logging configuration."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import DATA_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Attach console and rotating-file handlers to the root logger once."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            DATA_DIR / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Read-only deployment filesystems are expected; console logging still applies.
        root.warning("File logging unavailable; continuing with console logging only.")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    configure_logging()
    return logging.getLogger(name)
