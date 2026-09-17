"""Unit tests for time helpers (deterministic)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gold_trader.utils.time_utils import (
    TimeParseError,
    is_within_trading_hours,
    parse_hhmm,
    timeframe_seconds,
)


def make_dt(hour: int, minute: int) -> datetime:
    return datetime(2026, 9, 17, hour, minute, tzinfo=timezone.utc)


def test_parse_hhmm():
    assert parse_hhmm("07:30") == (7, 30)
    assert parse_hhmm("23:59") == (23, 59)


def test_parse_hhmm_invalid():
    for bad in ("25:00", "12:60", "12", "ab:cd", ""):
        with pytest.raises(TimeParseError):
            parse_hhmm(bad)


def test_within_normal_window():
    assert is_within_trading_hours(make_dt(12, 0), "08:00", "17:00")
    assert is_within_trading_hours(make_dt(8, 0), "08:00", "17:00")  # inclusive start
    assert is_within_trading_hours(make_dt(17, 0), "08:00", "17:00")  # inclusive end
    assert not is_within_trading_hours(make_dt(7, 59), "08:00", "17:00")
    assert not is_within_trading_hours(make_dt(17, 1), "08:00", "17:00")


def test_window_crossing_midnight():
    assert is_within_trading_hours(make_dt(23, 0), "20:00", "02:00")
    assert is_within_trading_hours(make_dt(1, 0), "20:00", "02:00")
    assert is_within_trading_hours(make_dt(0, 0), "20:00", "02:00")
    assert not is_within_trading_hours(make_dt(12, 0), "20:00", "02:00")


def test_timeframe_seconds():
    assert timeframe_seconds("M1") == 60
    assert timeframe_seconds("M15") == 900
    assert timeframe_seconds("h4") == 14400
    with pytest.raises(ValueError):
        timeframe_seconds("M7")
