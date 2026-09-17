"""Access to MT5 numeric constants with documented fallback values.

The fallback values equal the official MetaTrader5 Python API constants;
they are used only when the package cannot be imported (non-Windows
machines) so the integration layer stays importable for unit tests.
"""
from __future__ import annotations

from .connection import MT5_AVAILABLE, _mt5


def const(name: str, fallback: int) -> int:
    """Return the MT5 constant ``name`` (or its documented fallback)."""
    if MT5_AVAILABLE and _mt5 is not None:
        value = getattr(_mt5, name, None)
        if isinstance(value, int):
            return value
    return fallback
