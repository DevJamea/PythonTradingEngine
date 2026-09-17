"""Access to MT5 numeric constants with documented fallback values.

The fallback values equal the official MetaTrader5 Python API constants;
they are used only when the package cannot be imported (non-Windows
machines) so the integration layer stays importable for unit tests.
"""
from __future__ import annotations

from .connection import MT5_AVAILABLE, _mt5


def const(name: str, fallback: int) -> int:
    """Return the MT5 constant ``name`` (or its documented fallback).

    The Python MT5 package uses TRADE_ACTION_* names, while some code
    historically used ORDER_ACTION_*. This helper tries both prefixes
    to stay compatible with real terminals and unit-test fallbacks.
    """
    if MT5_AVAILABLE and _mt5 is not None:
        # Direct name
        value = getattr(_mt5, name, None)
        if isinstance(value, int):
            return value
        # Try TRADE_ACTION_* <-> ORDER_ACTION_* aliases
        aliases = []
        if name.startswith("ORDER_ACTION_"):
            aliases.append(name.replace("ORDER_ACTION_", "TRADE_ACTION_"))
        if name.startswith("TRADE_ACTION_"):
            aliases.append(name.replace("TRADE_ACTION_", "ORDER_ACTION_"))
        # DELETE vs REMOVE alias
        if name in ("ORDER_ACTION_DELETE", "TRADE_ACTION_DELETE"):
            aliases.extend(["TRADE_ACTION_REMOVE", "ORDER_ACTION_REMOVE"])
        if name in ("TRADE_ACTION_REMOVE", "ORDER_ACTION_REMOVE"):
            aliases.extend(["TRADE_ACTION_DELETE", "ORDER_ACTION_DELETE"])
        for alt in aliases:
            alt_val = getattr(_mt5, alt, None)
            if isinstance(alt_val, int):
                return alt_val
    return fallback
