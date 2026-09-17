"""Trend classification from EMA alignment.

Rules (all inputs explicit -> trivially unit-testable):

* UPTREND:   ema_fast > ema_medium > ema_slow (and close > ema_fast)
* DOWNTREND: ema_fast < ema_medium < ema_slow (and close < ema_fast)
* SIDEWAYS:  anything else, including missing (NaN) indicator values

These are classification rules, not a claim about profitability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from ..models import Trend
from .indicators import ema, latest_valid


@dataclass(frozen=True)
class TrendResult:
    """Structured trend result."""

    trend: Trend
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


def _clean(value: Optional[float]) -> Optional[float]:
    """Normalize None/NaN to None."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def classify_trend_values(
    close: Optional[float],
    ema_fast: Optional[float],
    ema_medium: Optional[float],
    ema_slow: Optional[float],
    require_close: bool = True,
) -> TrendResult:
    """Classify the trend from scalar EMA values (pure function)."""
    close_v = _clean(close)
    fast_v = _clean(ema_fast)
    medium_v = _clean(ema_medium)
    slow_v = _clean(ema_slow)
    details: Dict[str, Any] = {
        "close": close_v,
        "ema_fast": fast_v,
        "ema_medium": medium_v,
        "ema_slow": slow_v,
    }
    if None in (close_v, fast_v, medium_v, slow_v):
        return TrendResult(Trend.SIDEWAYS, "insufficient indicator history (NaN)", details)

    if fast_v > medium_v > slow_v:
        if not require_close or close_v > fast_v:
            return TrendResult(
                Trend.UPTREND,
                "EMA alignment fast>medium>slow with close above fast EMA",
                details,
            )
    if fast_v < medium_v < slow_v:
        if not require_close or close_v < fast_v:
            return TrendResult(
                Trend.DOWNTREND,
                "EMA alignment fast<medium<slow with close below fast EMA",
                details,
            )
    return TrendResult(Trend.SIDEWAYS, "EMAs not aligned", details)


def classify_trend(
    df: pd.DataFrame,
    ema_fast_period: int,
    ema_medium_period: int,
    ema_slow_period: int,
    require_close: bool = True,
) -> TrendResult:
    """Convenience wrapper: compute EMAs on ``df`` and classify the last bar."""
    close = df["close"].astype(float)
    return classify_trend_values(
        latest_valid(close),
        latest_valid(ema(close, ema_fast_period)),
        latest_valid(ema(close, ema_medium_period)),
        latest_valid(ema(close, ema_slow_period)),
        require_close,
    )
