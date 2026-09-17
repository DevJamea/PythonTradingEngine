"""OHLCV market data from the MT5 terminal.

Supported timeframes: M1, M5, M15, M30, H1, H4 (default M15 via config).
Signals must be produced from *closed* candles only: use
:func:`drop_unclosed_candle` before any decision, and never act on the
still-forming last candle.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from ..utils.time_utils import timeframe_seconds
from ._constants import const
from .connection import MT5DataError, require_mt5

logger = logging.getLogger("gold_trader.mt5.market_data")

#: Documented MT5 timeframe constants (used only when the package cannot
#: be imported, e.g. on non-Windows machines).
#: Official MT5 Python values: M1=1, M5=5, M15=15, M30=30, H1=16385, H4=16388
_TIMEFRAME_FALLBACKS: Dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
}


@dataclass(frozen=True)
class TickData:
    """Current quote for one symbol."""

    bid: float
    ask: float
    last: float
    time: datetime

    @property
    def spread(self) -> float:
        """Spread in price units (ask - bid)."""
        return self.ask - self.bid

    @property
    def is_usable(self) -> bool:
        """True when bid > 0 and ask > 0 and ask >= bid."""
        return is_tick_usable(self)


def is_tick_usable(tick: Optional[Any]) -> bool:
    """Check if market pricing is available and valid from tick data.

    For Forex/CFD instruments (such as Gold/XAU), 'last' can legitimately
    be 0 while valid Bid and Ask quotes exist. Usable market pricing exists
    when bid > 0, ask > 0, and ask >= bid.
    """
    if tick is None:
        return False
    try:
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    import math
    if not math.isfinite(bid) or not math.isfinite(ask):
        return False
    return bid > 0 and ask > 0 and ask >= bid


def _mt5_timeframe(timeframe: str) -> int:
    key = timeframe.upper()
    if key not in _TIMEFRAME_FALLBACKS:
        supported = ", ".join(sorted(_TIMEFRAME_FALLBACKS))
        raise ValueError(f"unsupported timeframe {timeframe!r}; expected: {supported}")
    return const(f"TIMEFRAME_{key}", _TIMEFRAME_FALLBACKS[key])


def get_candles(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Fetch the last ``count`` candles as an OHLCV DataFrame.

    Columns: time, open, high, low, close, tick_volume, spread, real_volume.
    The last row may still be forming -- filter it with
    :func:`drop_unclosed_candle`.
    """
    mt5_api = require_mt5()
    tf = _mt5_timeframe(timeframe)
    rates = mt5_api.copy_rates_from_pos(symbol, tf, 0, int(count))
    if rates is None or len(rates) == 0:
        raise MT5DataError(
            f"copy_rates_from_pos({symbol!r}, {timeframe}, {count}) returned no "
            f"data: {mt5_api.last_error()}"
        )
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for column in ("open", "high", "low", "close"):
        df[column] = df[column].astype(float)
    df["tick_volume"] = df["tick_volume"].astype("int64")
    df["real_volume"] = df["real_volume"].astype("int64")
    if "spread" not in df.columns:
        df["spread"] = 0
    return df[
        ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    ]


def drop_unclosed_candle(
    df: pd.DataFrame, timeframe: str, now: Optional[datetime] = None
) -> pd.DataFrame:
    """Remove the still-forming candle so only closed candles remain.

    A candle is closed when the current time has reached
    ``candle_time + timeframe``. ``now`` is injectable for deterministic
    tests.
    """
    if len(df) == 0:
        return df
    seconds = timeframe_seconds(timeframe)
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc), tz="UTC")
    last_time = pd.Timestamp(df["time"].iloc[-1], tz="UTC")
    if now_ts < last_time + pd.Timedelta(seconds=seconds):
        return df.iloc[:-1].copy()
    return df


def get_tick(symbol: str) -> TickData:
    """Current bid/ask quote; raises MT5DataError when the market is closed."""
    mt5_api = require_mt5()
    tick = mt5_api.symbol_info_tick(symbol)
    if not is_tick_usable(tick):
        last_err = mt5_api.last_error() if hasattr(mt5_api, "last_error") else "unknown"
        raise MT5DataError(
            f"no valid tick for {symbol!r} (market closed or symbol not "
            f"selected): {last_err}"
        )
    raw_last = getattr(tick, "last", 0.0)
    last_val = float(raw_last) if raw_last is not None else 0.0
    raw_time = getattr(tick, "time", 0)
    if isinstance(raw_time, datetime):
        tick_time = raw_time if raw_time.tzinfo else raw_time.replace(tzinfo=timezone.utc)
    else:
        tick_time = datetime.fromtimestamp(float(raw_time or 0), tz=timezone.utc)
    return TickData(
        bid=float(tick.bid),
        ask=float(tick.ask),
        last=last_val,
        time=tick_time,
    )
