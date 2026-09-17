"""Strategy layer: candle patterns, indicators, trend, signals, levels.

Everything in this package is pure pandas/numpy: no MT5 imports, no I/O,
no randomness -- fully unit-testable and reusable by the backtest engine.
"""
from .candles import (
    Candle,
    CandlePattern,
    detect_patterns_pair,
    latest_patterns,
)
from .indicators import atr, ema, latest_valid, rsi, true_range
from .levels import (
    SLTPPlan,
    SwingPoint,
    build_sl_tp,
    find_swing_points,
    nearest_resistance,
    nearest_support,
)
from .signals import (
    IndicatorFrame,
    compute_indicators,
    evaluate_at,
    generate_signal,
)
from .trend import TrendResult, classify_trend, classify_trend_values

__all__ = [
    "Candle",
    "CandlePattern",
    "detect_patterns_pair",
    "latest_patterns",
    "atr",
    "ema",
    "latest_valid",
    "rsi",
    "true_range",
    "SLTPPlan",
    "SwingPoint",
    "build_sl_tp",
    "find_swing_points",
    "nearest_resistance",
    "nearest_support",
    "IndicatorFrame",
    "compute_indicators",
    "evaluate_at",
    "generate_signal",
    "TrendResult",
    "classify_trend",
    "classify_trend_values",
]
