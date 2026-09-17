"""Signal generation, fully decoupled from MT5.

A signal is produced from *closed* candles only and depends exclusively
on data available at the decision moment (row ``i`` uses rows ``0..i``).

MVP strategy (intentionally simple, fully configurable via ``Config``):

* BUY  : trend UPTREND  + one of ``buy_patterns``  + RSI < overbought + valid ATR
* SELL : trend DOWNTREND + one of ``sell_patterns`` + RSI > oversold + valid ATR
* otherwise NO_TRADE

This is NOT claimed to be a profitable strategy -- it is a clear,
testable baseline that can be swapped without touching the rest of the
system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from ..config import Config
from ..models import Signal, TradeSignal, Trend
from .candles import Candle, detect_patterns_pair
from .indicators import atr as atr_indicator
from .indicators import ema, rsi
from .trend import TrendResult, classify_trend_values

REQUIRED_COLUMNS = ("time", "open", "high", "low", "close")


@dataclass(frozen=True)
class IndicatorFrame:
    """Precomputed indicator series for one OHLC frame."""

    close: pd.Series
    ema_fast: pd.Series
    ema_medium: pd.Series
    ema_slow: pd.Series
    rsi: pd.Series
    atr: pd.Series


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"candle frame is missing columns: {missing}")


def compute_indicators(df: pd.DataFrame, cfg: Config) -> IndicatorFrame:
    """Compute all indicator series used by the strategy (causal)."""
    _validate_columns(df)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    return IndicatorFrame(
        close=close,
        ema_fast=ema(close, cfg.ema_fast),
        ema_medium=ema(close, cfg.ema_medium),
        ema_slow=ema(close, cfg.ema_slow),
        rsi=rsi(close, cfg.rsi_period),
        atr=atr_indicator(high, low, close, cfg.atr_period),
    )


def _opt(value) -> Optional[float]:
    """Convert a numpy/pandas scalar to float, mapping NaN -> None."""
    number = float(value)
    return None if math.isnan(number) else number


def evaluate_at(
    index: int, df: pd.DataFrame, ind: IndicatorFrame, cfg: Config
) -> TradeSignal:
    """Evaluate the strategy at closed candle ``index``.

    Only rows ``0..index`` are used (pattern pair = index-1 / index).
    The backtest engine calls this with precomputed indicators; because
    every indicator is causal, the result is identical to recomputing on
    the visible window (covered by a unit test).
    """
    if index < 1 or index >= len(df):
        raise ValueError(f"index {index} out of range for {len(df)} candles")
    close = float(ind.close.iloc[index])
    fast = _opt(ind.ema_fast.iloc[index])
    medium = _opt(ind.ema_medium.iloc[index])
    slow = _opt(ind.ema_slow.iloc[index])
    rsi_value = _opt(ind.rsi.iloc[index])
    atr_value = _opt(ind.atr.iloc[index])

    trend = classify_trend_values(close, fast, medium, slow, cfg.trend_require_close)
    patterns = detect_patterns_pair(
        Candle.from_mapping(df.iloc[index - 1]),
        Candle.from_mapping(df.iloc[index]),
        cfg.doji_body_fraction,
    )
    detected = [name for name, pattern in patterns.items() if pattern.detected]
    details: Dict[str, Any] = {
        "trend": trend.trend.value,
        "rsi": rsi_value,
        "atr": atr_value,
        "patterns": detected,
        "close": close,
    }
    if rsi_value is None or atr_value is None or atr_value <= 0:
        return TradeSignal(Signal.NO_TRADE, "indicators not ready (RSI/ATR invalid)", details)
    return _decide(trend, patterns, rsi_value, atr_value, cfg, details)


def _decide(
    trend: TrendResult,
    patterns: Dict[str, Any],
    rsi_value: float,
    atr_value: float,
    cfg: Config,
    details: Dict[str, Any],
) -> TradeSignal:
    """Apply the configurable BUY/SELL rules."""
    detected = {name for name, pattern in patterns.items() if pattern.detected}

    if trend.trend is Trend.UPTREND:
        if rsi_value >= cfg.rsi_overbought:
            return TradeSignal(
                Signal.NO_TRADE,
                f"uptrend but RSI {rsi_value:.2f} >= overbought {cfg.rsi_overbought}",
                details,
            )
        buy_hits = sorted(detected.intersection(cfg.buy_patterns))
        if buy_hits:
            return TradeSignal(
                Signal.BUY,
                f"uptrend + {', '.join(buy_hits)} + RSI {rsi_value:.2f} < "
                f"{cfg.rsi_overbought} + ATR {atr_value:.2f}",
                details,
            )
        return TradeSignal(
            Signal.NO_TRADE,
            f"uptrend but no buy pattern (detected: {sorted(detected) or 'none'})",
            details,
        )

    if trend.trend is Trend.DOWNTREND:
        if rsi_value <= cfg.rsi_oversold:
            return TradeSignal(
                Signal.NO_TRADE,
                f"downtrend but RSI {rsi_value:.2f} <= oversold {cfg.rsi_oversold}",
                details,
            )
        sell_hits = sorted(detected.intersection(cfg.sell_patterns))
        if sell_hits:
            return TradeSignal(
                Signal.SELL,
                f"downtrend + {', '.join(sell_hits)} + RSI {rsi_value:.2f} > "
                f"{cfg.rsi_oversold} + ATR {atr_value:.2f}",
                details,
            )
        return TradeSignal(
            Signal.NO_TRADE,
            f"downtrend but no sell pattern (detected: {sorted(detected) or 'none'})",
            details,
        )

    return TradeSignal(
        Signal.NO_TRADE, f"trend {trend.trend.value}: {trend.reason}", details
    )


def generate_signal(df: pd.DataFrame, cfg: Config) -> TradeSignal:
    """Compute indicators and evaluate the last closed candle."""
    if len(df) < 2:
        raise ValueError("at least 2 closed candles are required")
    if len(df) < cfg.min_candles_for_signal:
        return TradeSignal(
            Signal.NO_TRADE,
            f"insufficient candles ({len(df)} < {cfg.min_candles_for_signal})",
            {"length": len(df)},
        )
    ind = compute_indicators(df, cfg)
    return evaluate_at(len(df) - 1, df, ind, cfg)
