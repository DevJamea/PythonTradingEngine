"""Technical indicators implemented with pandas/numpy only.

All indicators are *causal*: the value at index ``i`` depends exclusively
on rows ``0..i``. No indicator ever looks at future candles. NaN handling:
values before the warm-up period are NaN; use :func:`latest_valid` to read
the last usable value.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (span = period, no look-ahead).

    The first ``period - 1`` values are NaN (min_periods).
    """
    if period < 1:
        raise ValueError("EMA period must be >= 1")
    return series.ewm(span=int(period), adjust=False, min_periods=int(period)).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    Uses Wilder smoothing (alpha = 1/period). Result is in [0, 100];
    a fully flat segment (no gains and no losses) maps to 50 (neutral).
    """
    if period < 1:
        raise ValueError("RSI period must be >= 1")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    return out.mask(flat, 50.0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max(H-L, |H-prevClose|, |L-prevClose|)."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Wilder's Average True Range (alpha = 1/period)."""
    if period < 1:
        raise ValueError("ATR period must be >= 1")
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def latest_valid(series: pd.Series) -> Optional[float]:
    """Last non-NaN value of the series, or None if the series has none."""
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])
