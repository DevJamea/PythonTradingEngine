"""Time helpers (UTC-based and deterministic)."""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Tuple

#: Supported timeframes and their duration in seconds.
TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
}


class TimeParseError(ValueError):
    """Raised when an 'HH:MM' value is malformed."""


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def parse_hhmm(value: str) -> Tuple[int, int]:
    """Parse ``HH:MM`` into ``(hour, minute)``."""
    try:
        parts = value.strip().split(":")
        if len(parts) != 2:
            raise ValueError
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise TimeParseError(f"invalid time of day: {value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise TimeParseError(f"time of day out of range: {value!r}")
    return hour, minute


def is_within_trading_hours(now: datetime, start: str, end: str) -> bool:
    """Check whether ``now`` falls inside ``start``..``end`` (UTC).

    Both bounds are inclusive. ``start > end`` is interpreted as a window
    that crosses midnight (e.g. 20:00-02:00).
    """
    start_hour, start_minute = parse_hhmm(start)
    end_hour, end_minute = parse_hhmm(end)
    current = time(now.hour, now.minute)
    start_t = time(start_hour, start_minute)
    end_t = time(end_hour, end_minute)
    if start_t <= end_t:
        return start_t <= current <= end_t
    return current >= start_t or current <= end_t


def timeframe_seconds(timeframe: str) -> int:
    """Seconds in one candle of ``timeframe`` (case-insensitive)."""
    key = timeframe.upper()
    if key not in TIMEFRAME_SECONDS:
        supported = ", ".join(sorted(TIMEFRAME_SECONDS))
        raise ValueError(f"unsupported timeframe {timeframe!r}; expected one of: {supported}")
    return TIMEFRAME_SECONDS[key]
