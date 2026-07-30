"""
Configures logging for the full application.
Writes to both console (INFO) and a timestamped log file (DEBUG).
Called once in main.py before anything else runs.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(logs_dir: Path, run_mode: str) -> None:
    """
    Configure root logger with console and file handlers.

    Console: INFO level, concise format for monitoring progress.
    File:    DEBUG level, verbose format for post-run inspection.

    Args:
        logs_dir:  Directory to write log files into.
        run_mode:  Included in log filename for easy identification.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{run_mode}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing handlers (e.g. from pytest or notebooks)
    root.handlers.clear()

    # --- Console handler ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(NoTracebackFormatter(    
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # --- File handler ---
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_verbose_formatter())
    root.addHandler(file_handler)

    logging.info("Logging initialised. Log file: %s", log_file)

class NoTracebackFormatter(logging.Formatter):
    """Console formatter that suppresses exception tracebacks to keep output readable."""
    def format(self, record):
        record = logging.makeLogRecord(record.__dict__)  # copy to avoid mutating original
        record.exc_info = None
        record.exc_text = None
        return super().format(record)


def _verbose_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )