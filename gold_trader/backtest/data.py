"""Candle loading / resampling / period splitting for offline research.

Pure functions over pandas frames -- no network, no I/O beyond one explicit
``read_csv``. Every transformation is causal and deterministic, which is what
lets the backtest results be reproduced from the same file.

The gold study data format (gzip CSV) is::

    time,open,high,low,close,tick_volume,spread,spread_p90,spread_max

``spread`` is the real ``ask - bid`` distance averaged over the minutes inside
the bar, in price units (USD per ounce). It is produced by
:func:`build_m1_from_bid_ask` + :func:`resample_candles` from a broker/Dukascopy
bid+ask dump, so the cost model can price each bar with the spread that was
actually quoted then, instead of one global constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

CANDLE_COLUMNS = ("time", "open", "high", "low", "close")
TIMEFRAME_RULES = {
    "M1": "1min",
    "M3": "3min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def normalize_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a candle frame (``time`` tz-aware UTC, floats).

    Raises :class:`ValueError` when a required column is missing, the frame is
    empty, timestamps are not unique/ascending, or a bar violates
    ``high >= max(open, close) >= min(open, close) >= low``. A silently broken
    OHLC frame produces a silently flattering backtest, so this refuses.
    """
    missing = [column for column in CANDLE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"candle frame is missing columns: {missing}")
    if len(df) == 0:
        raise ValueError("candle frame is empty")
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    if out[list(CANDLE_COLUMNS[1:])].isna().any().any():
        raise ValueError("candle frame contains NaN/invalid OHLC values")
    out = out.sort_values("time").reset_index(drop=True)
    if out["time"].duplicated().any():
        raise ValueError("candle frame contains duplicate timestamps")
    bad_range = (
        (out["high"] < out[["open", "close"]].max(axis=1))
        | (out["low"] > out[["open", "close"]].min(axis=1))
        | (out["high"] <= 0)
    )
    if bool(bad_range.any()):
        raise ValueError(
            f"candle frame contains {int(bad_range.sum())} rows with impossible "
            "OHLC geometry (high < max(open,close) or low > min(open,close))"
        )
    return out


def load_candles_csv(path: str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read a gzip/plain CSV candle dump and normalise it."""
    df = pd.read_csv(path)
    if columns is not None:
        keep = [c for c in df.columns if c in set(columns)]
        df = df[keep]
    return normalize_candles(df)


def build_m1_from_bid_ask(
    bid: pd.DataFrame, ask: pd.DataFrame
) -> pd.DataFrame:
    """Join minute bid/ask OHLC into one M1 frame (bid = chart price).

    ``bid``/``ask`` both need ``time, open, high, low, close``. The result
    carries the bid OHLC (what a chart shows) plus the per-minute ``spread``
    (``ask.close - bid.close``), which is what a round trip actually costs.
    """
    b = normalize_candles(bid)[list(CANDLE_COLUMNS)]
    a = normalize_candles(ask)[list(CANDLE_COLUMNS)].rename(
        columns={c: f"{c}_ask" for c in ("open", "high", "low", "close")}
    )
    merged = b.merge(a, on="time", how="inner").sort_values("time").reset_index(drop=True)
    if merged.empty:
        raise ValueError("bid and ask frames share no timestamps")
    merged["spread"] = (merged["close_ask"] - merged["close"]).clip(lower=0.0)
    return merged[list(CANDLE_COLUMNS) + ["spread"]]


def resample_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate an M1 frame (with optional ``spread``) into ``timeframe``.

    OHLC follows the usual open/high/low/close aggregation; ``tick_volume``
    counts the minutes that actually exist (gold has no exchange volume) and
    ``spread`` is averaged over the bar with ``spread_max`` kept as well, so a
    study can price fills with the mean and stress-test with the worst minute.
    """
    rule = TIMEFRAME_RULES.get(timeframe.upper())
    if rule is None:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; expected one of "
            f"{', '.join(TIMEFRAME_RULES)}"
        )
    src = normalize_candles(df)
    indexed = src.set_index("time")
    groups = indexed.resample(rule)
    bars = pd.DataFrame(
        {
            "open": groups["open"].first(),
            "high": groups["high"].max(),
            "low": groups["low"].min(),
            "close": groups["close"].last(),
            "tick_volume": groups["close"].count(),
        }
    )
    if "spread" in indexed.columns:
        bars["spread"] = groups["spread"].mean()
        bars["spread_max"] = groups["spread"].max()
    bars = bars[bars["tick_volume"] > 0].reset_index().rename(columns={"index": "time"})
    # a resampled bar can only be as valid as its source geometry
    return normalize_candles(bars)


def spread_from_points(df: pd.DataFrame, point: float) -> pd.Series:
    """Convert MT5's ``spread`` column (integer *points*) to price units.

    ``copy_rates_*`` reports the spread as a whole number of points, while
    every cost in this project is a price distance. Mixing the two silently
    would multiply the assumed cost by 100 or divide it by 100, so the
    conversion is explicit and lives here.
    """
    if "spread" not in df.columns:
        raise ValueError("candle frame has no 'spread' column to convert")
    if point <= 0:
        raise ValueError("symbol point size must be > 0")
    values = pd.to_numeric(df["spread"], errors="coerce").astype(float) * float(point)
    return values.clip(lower=0.0)


def split_period(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Half-open slice ``[start, end)`` on the ``time`` column (inclusive start)."""
    out = normalize_candles(df)
    if start is not None:
        out = out[out["time"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        out = out[out["time"] < pd.Timestamp(end, tz="UTC")]
    return out.reset_index(drop=True)


def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Calendar-month ``[first, first_of_next)`` pairs from ``start`` to ``end``.

    Used by walk-forward validation so every fold is a whole month and no fold
    leaks a partial month into its neighbour.
    """
    first = pd.Timestamp(start, tz="UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last = pd.Timestamp(end, tz="UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    windows: list[tuple[str, str]] = []
    cursor = first
    while cursor < last:
        nxt = cursor + pd.offsets.MonthBegin(1)
        windows.append((cursor.isoformat(), nxt.isoformat()))
        cursor = nxt
    return windows


@dataclass(frozen=True)
class DataQuality:
    """Summary of what a candle dump actually contains (for the report)."""

    rows: int
    first_time: pd.Timestamp
    last_time: pd.Timestamp
    years: tuple[int, ...]
    mean_spread: Optional[float]
    p90_spread: Optional[float]
    max_spread: Optional[float]
    duplicate_minutes: int


def assess_quality(df: pd.DataFrame) -> DataQuality:
    """Describe the loaded frame; the report must never quote data unseen."""
    src = df
    spread = None
    if "spread" in src.columns:
        spread = pd.to_numeric(src["spread"], errors="coerce").dropna()
        spread = spread[spread > 0]
    times = pd.to_datetime(src["time"], utc=True)
    return DataQuality(
        rows=int(len(src)),
        first_time=times.min(),
        last_time=times.max(),
        years=tuple(int(y) for y in sorted(times.dt.year.unique())),
        mean_spread=None if spread is None or spread.empty else float(spread.mean()),
        p90_spread=None if spread is None or spread.empty else float(spread.quantile(0.90)),
        max_spread=None if spread is None or spread.empty else float(spread.max()),
        duplicate_minutes=int(times.duplicated().sum()),
    )
