"""Logging setup: console + logs/app.log + logs/trades.log + logs/errors.log.

* ``gold_trader``      -> app.log (INFO+), errors.log (ERROR+), console
* ``gold_trader.trades`` -> trades.log (INFO), also propagated to app.log
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
APP_FILE = "app.log"
TRADES_FILE = "trades.log"
ERRORS_FILE = "errors.log"
FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None, level: str = "INFO"
) -> None:
    """Create the log files and attach handlers (idempotent per call)."""
    directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(FORMAT)

    root = logging.getLogger("gold_trader")
    root.setLevel(numeric_level)
    root.propagate = False
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    root.addHandler(console)

    app_handler = logging.FileHandler(directory / APP_FILE, encoding="utf-8")
    app_handler.setFormatter(formatter)
    app_handler.setLevel(logging.INFO)
    root.addHandler(app_handler)

    error_handler = logging.FileHandler(directory / ERRORS_FILE, encoding="utf-8")
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root.addHandler(error_handler)

    trades = logging.getLogger("gold_trader.trades")
    trades.setLevel(logging.INFO)
    trades.propagate = True
    for handler in list(trades.handlers):
        handler.close()
        trades.removeHandler(handler)
    trades_handler = logging.FileHandler(directory / TRADES_FILE, encoding="utf-8")
    trades_handler.setFormatter(formatter)
    trades_handler.setLevel(logging.INFO)
    trades.addHandler(trades_handler)


def get_logger(name: str = "gold_trader") -> logging.Logger:
    """Application logger (app.log + console + errors.log)."""
    return logging.getLogger(name)


def get_trades_logger() -> logging.Logger:
    """Trade audit logger (trades.log + app.log)."""
    return logging.getLogger("gold_trader.trades")


def get_errors_logger() -> logging.Logger:
    """Error logger (errors.log + console)."""
    return logging.getLogger("gold_trader.errors")
