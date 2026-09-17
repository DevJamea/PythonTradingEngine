"""Candlestick pattern detection.

Every detector returns a structured :class:`CandlePattern` instead of a
bare string. ``strength`` is a value in [0, 1] computed with an explicit
mathematical formula when a defensible one exists, otherwise it is 0.0
(never a random number). Strength formulas:

* bullish/bearish candle : body / range
* doji                   : 1 - body / range
* hammer                 : lower wick / range
* shooting star          : upper wick / range
* engulfing (both)       : body of the engulfing candle / sum of the two bodies
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class Candle:
    """One OHLC candle (open/high/low/close as floats)."""

    open: float
    high: float
    low: float
    close: float

    @property
    def body(self) -> float:
        """Absolute body size."""
        return abs(self.close - self.open)

    @property
    def total_range(self) -> float:
        """High - low."""
        return self.high - self.low

    @property
    def upper_wick(self) -> float:
        """Distance from body top to the high."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Distance from body bottom to the low."""
        return min(self.open, self.close) - self.low

    @classmethod
    def from_mapping(cls, row) -> "Candle":
        """Build a Candle from a pandas row (needs o/h/l/c columns)."""
        return cls(
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
        )


@dataclass(frozen=True)
class CandlePattern:
    """Structured pattern result (no free text only)."""

    pattern: str
    detected: bool
    strength: float
    description: str = ""


PATTERN_NAMES: Tuple[str, ...] = (
    "bullish_candle",
    "bearish_candle",
    "doji",
    "hammer",
    "shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",
)

#: Maximum fraction of the range allowed for the small (opposite) wick
#: of hammer / shooting star candles.
_SMALL_WICK_FRACTION = 0.25


def detect_bullish_candle(candle: Candle) -> CandlePattern:
    """Bullish candle: close strictly above open. Strength = body/range."""
    if candle.total_range <= 0 or candle.close <= candle.open:
        return CandlePattern("bullish_candle", False, 0.0, "close not above open")
    strength = candle.body / candle.total_range
    return CandlePattern(
        "bullish_candle", True, strength,
        f"body {candle.body:.5f} of range {candle.total_range:.5f}",
    )


def detect_bearish_candle(candle: Candle) -> CandlePattern:
    """Bearish candle: close strictly below open. Strength = body/range."""
    if candle.total_range <= 0 or candle.close >= candle.open:
        return CandlePattern("bearish_candle", False, 0.0, "close not below open")
    strength = candle.body / candle.total_range
    return CandlePattern(
        "bearish_candle", True, strength,
        f"body {candle.body:.5f} of range {candle.total_range:.5f}",
    )


def detect_doji(candle: Candle, body_fraction: float = 0.10) -> CandlePattern:
    """Doji: body <= body_fraction * range. Strength = 1 - body/range."""
    if candle.total_range <= 0:
        return CandlePattern("doji", False, 0.0, "zero-range candle")
    if candle.body <= body_fraction * candle.total_range:
        strength = 1.0 - candle.body / candle.total_range
        return CandlePattern("doji", True, strength, "indecisive candle")
    return CandlePattern(
        "doji", False, 0.0,
        f"body {candle.body:.5f} > {body_fraction:.2f} * range",
    )


def detect_hammer(candle: Candle) -> CandlePattern:
    """Hammer: long lower wick (>= 2x body), small upper wick.

    Strength = lower wick / range.
    """
    if candle.total_range <= 0 or candle.body <= 0:
        return CandlePattern("hammer", False, 0.0, "not a valid hammer shape")
    long_lower = candle.lower_wick >= 2.0 * candle.body
    small_upper = candle.upper_wick <= _SMALL_WICK_FRACTION * candle.total_range
    if long_lower and small_upper:
        strength = candle.lower_wick / candle.total_range
        return CandlePattern("hammer", True, strength, "long lower wick")
    return CandlePattern("hammer", False, 0.0, "wick/body ratio not met")


def detect_shooting_star(candle: Candle) -> CandlePattern:
    """Shooting star: long upper wick (>= 2x body), small lower wick.

    Strength = upper wick / range.
    """
    if candle.total_range <= 0 or candle.body <= 0:
        return CandlePattern("shooting_star", False, 0.0, "not a valid shooting star shape")
    long_upper = candle.upper_wick >= 2.0 * candle.body
    small_lower = candle.lower_wick <= _SMALL_WICK_FRACTION * candle.total_range
    if long_upper and small_lower:
        strength = candle.upper_wick / candle.total_range
        return CandlePattern("shooting_star", True, strength, "long upper wick")
    return CandlePattern("shooting_star", False, 0.0, "wick/body ratio not met")


def detect_bullish_engulfing(prev: Candle, cur: Candle) -> CandlePattern:
    """Bullish engulfing: bearish prev, bullish cur whose body fully
    engulfs the previous body and is strictly larger.

    Strength = cur body / (cur body + prev body), in (0.5, 1).
    """
    prev_bearish = prev.close < prev.open
    cur_bullish = cur.close > cur.open
    engulfs = cur.open <= prev.close and cur.close >= prev.open
    larger_body = cur.body > prev.body
    if prev_bearish and cur_bullish and engulfs and larger_body:
        strength = cur.body / (cur.body + prev.body)
        return CandlePattern(
            "bullish_engulfing", True, strength,
            f"body {cur.body:.5f} engulfs prev body {prev.body:.5f}",
        )
    return CandlePattern("bullish_engulfing", False, 0.0, "engulfing conditions not met")


def detect_bearish_engulfing(prev: Candle, cur: Candle) -> CandlePattern:
    """Bearish engulfing: bullish prev, bearish cur whose body fully
    engulfs the previous body and is strictly larger.

    Strength = cur body / (cur body + prev body), in (0.5, 1).
    """
    prev_bullish = prev.close > prev.open
    cur_bearish = cur.close < cur.open
    engulfs = cur.open >= prev.close and cur.close <= prev.open
    larger_body = cur.body > prev.body
    if prev_bullish and cur_bearish and engulfs and larger_body:
        strength = cur.body / (cur.body + prev.body)
        return CandlePattern(
            "bearish_engulfing", True, strength,
            f"body {cur.body:.5f} engulfs prev body {prev.body:.5f}",
        )
    return CandlePattern("bearish_engulfing", False, 0.0, "engulfing conditions not met")


def detect_patterns_pair(
    prev: Candle, cur: Candle, doji_body_fraction: float = 0.10
) -> Dict[str, CandlePattern]:
    """Detect all MVP patterns for the (prev, cur) candle pair."""
    return {
        "bullish_candle": detect_bullish_candle(cur),
        "bearish_candle": detect_bearish_candle(cur),
        "doji": detect_doji(cur, doji_body_fraction),
        "hammer": detect_hammer(cur),
        "shooting_star": detect_shooting_star(cur),
        "bullish_engulfing": detect_bullish_engulfing(prev, cur),
        "bearish_engulfing": detect_bearish_engulfing(prev, cur),
    }


def latest_patterns(
    df: pd.DataFrame, doji_body_fraction: float = 0.10
) -> Dict[str, CandlePattern]:
    """Detect patterns on the last two closed candles of ``df``."""
    if len(df) < 2:
        raise ValueError("at least 2 closed candles are required")
    return detect_patterns_pair(
        Candle.from_mapping(df.iloc[-2]),
        Candle.from_mapping(df.iloc[-1]),
        doji_body_fraction,
    )
