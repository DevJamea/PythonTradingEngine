"""Simple support/resistance discovery from recent swing points.

Swing points are pivots: a high that is the maximum (or a low that is the
minimum) over a symmetric window of ``pivot_window`` candles on each side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..config import Config
from ..models import Signal


@dataclass(frozen=True)
class SwingPoint:
    """A pivot high/low with its position in the frame."""

    index: int
    price: float
    kind: str  # "high" | "low"


@dataclass(frozen=True)
class SLTPPlan:
    """Computed stop loss / take profit with provenance."""

    sl: float
    tp: float
    sl_source: str
    tp_source: str


def find_swing_points(
    df: pd.DataFrame, pivot_window: int = 5, recent: Optional[int] = None
) -> List[SwingPoint]:
    """Find swing highs/lows.

    ``recent`` limits the result to the last ``recent`` candles (used as a
    lookback for level selection).
    """
    if pivot_window < 1:
        raise ValueError("pivot_window must be >= 1")
    n = len(df)
    if n < 2 * pivot_window + 1:
        return []
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    points: list[SwingPoint] = []
    for i in range(pivot_window, n - pivot_window):
        window_high = highs[i - pivot_window: i + pivot_window + 1]
        if highs[i] >= window_high.max():
            points.append(SwingPoint(i, float(highs[i]), "high"))
        window_low = lows[i - pivot_window: i + pivot_window + 1]
        if lows[i] <= window_low.min():
            points.append(SwingPoint(i, float(lows[i]), "low"))
    if recent is not None:
        cutoff = max(0, n - recent)
        points = [p for p in points if p.index >= cutoff]
    return points


def nearest_support(
    df: pd.DataFrame,
    price: float,
    lookback: int = 100,
    pivot_window: int = 5,
) -> Optional[float]:
    """Highest swing low below ``price`` within the recent lookback."""
    points = [
        p
        for p in find_swing_points(df, pivot_window, recent=lookback)
        if p.kind == "low" and p.price < price
    ]
    return max(p.price for p in points) if points else None


def nearest_resistance(
    df: pd.DataFrame,
    price: float,
    lookback: int = 100,
    pivot_window: int = 5,
) -> Optional[float]:
    """Lowest swing high above ``price`` within the recent lookback."""
    points = [
        p
        for p in find_swing_points(df, pivot_window, recent=lookback)
        if p.kind == "high" and p.price > price
    ]
    return min(p.price for p in points) if points else None


def build_sl_tp(
    direction: Signal,
    entry: float,
    atr_value: float,
    df: Optional[pd.DataFrame] = None,
    cfg: Optional[Config] = None,
) -> SLTPPlan:
    """ATR-based SL/TP, optionally refined with market levels.

    Base rules (always):
        BUY : SL = entry - ATR * atr_sl_multiplier ; TP = entry + ATR * atr_tp_multiplier
        SELL: SL = entry + ATR * atr_sl_multiplier ; TP = entry - ATR * atr_tp_multiplier

    Optional refinements (off by default, need ``df`` and ``cfg``):
        * use_levels_for_sl: BUY SL tightened to (support - buffer), SELL SL
          tightened to (resistance + buffer) -- only when that is closer to
          entry (never widens the stop).
        * use_levels_for_tp: BUY TP capped at (resistance - buffer), SELL TP
          capped at (support + buffer) -- only when that keeps TP on the
          right side of entry.
    """
    if atr_value is None or not math.isfinite(atr_value) or atr_value <= 0:
        raise ValueError("a valid positive ATR value is required")
    if entry <= 0:
        raise ValueError("entry price must be positive")
    if direction not in (Signal.BUY, Signal.SELL):
        raise ValueError("direction must be Signal.BUY or Signal.SELL")

    sl_source = "atr"
    tp_source = "atr"
    buffer = 0.0
    if cfg is not None:
        buffer = atr_value * cfg.level_buffer_atr

    is_buy = direction is Signal.BUY
    if is_buy:
        sl = entry - atr_value * (cfg.atr_sl_multiplier if cfg else 1.5)
        tp = entry + atr_value * (cfg.atr_tp_multiplier if cfg else 2.5)
    else:
        sl = entry + atr_value * (cfg.atr_sl_multiplier if cfg else 1.5)
        tp = entry - atr_value * (cfg.atr_tp_multiplier if cfg else 2.5)

    if cfg is not None and df is not None:
        if cfg.use_levels_for_sl:
            if is_buy:
                support = nearest_support(
                    df, entry, cfg.level_lookback, cfg.level_pivot_window
                )
                if support is not None:
                    candidate = support - buffer
                    if candidate > sl:
                        sl, sl_source = candidate, "support_buffer"
            else:
                resistance = nearest_resistance(
                    df, entry, cfg.level_lookback, cfg.level_pivot_window
                )
                if resistance is not None:
                    candidate = resistance + buffer
                    if candidate < sl:
                        sl, sl_source = candidate, "resistance_buffer"
        if cfg.use_levels_for_tp:
            if is_buy:
                resistance = nearest_resistance(
                    df, entry, cfg.level_lookback, cfg.level_pivot_window
                )
                if resistance is not None:
                    candidate = resistance - buffer
                    if entry < candidate < tp:
                        tp, tp_source = candidate, "resistance_buffer"
            else:
                support = nearest_support(
                    df, entry, cfg.level_lookback, cfg.level_pivot_window
                )
                if support is not None:
                    candidate = support + buffer
                    if tp < candidate < entry:
                        tp, tp_source = candidate, "support_buffer"

    return SLTPPlan(sl=sl, tp=tp, sl_source=sl_source, tp_source=tp_source)
