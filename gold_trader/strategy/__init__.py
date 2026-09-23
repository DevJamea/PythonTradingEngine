"""Strategy layer: candle patterns, indicators, trend, signals, levels.

Two decision engines coexist here and never override each other:

* :mod:`~gold_trader.strategy.signals` -- the original EMA+RSI+candle
  trend-following baseline (still the default, ``Config.active_strategy``
  = ``"signals"``);
* :mod:`~gold_trader.strategy.scalping` -- an opt-in short-horizon
  mean-reversion scalper with a mandatory cost (spread) gate
  (``Config.active_strategy`` = ``"scalping"``, and it does nothing while
  ``Config.scalping_enabled`` is False).

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
from .scalping import (
    ScalpingIndicators,
    ScalpingSLTP,
    bollinger_bands,
    build_scalping_sl_tp,
    compute_scalping_indicators,
    cost_gate_result,
    evaluate_scalping_at,
    effective_spread,
    generate_scalping_signal,
    percentile_rank,
    required_tp_distance,
    reversal_confirmed,
    spread_gate_result,
    volatility_gate_result,
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
    "ScalpingIndicators",
    "ScalpingSLTP",
    "bollinger_bands",
    "build_scalping_sl_tp",
    "compute_scalping_indicators",
    "cost_gate_result",
    "effective_spread",
    "evaluate_scalping_at",
    "generate_scalping_signal",
    "percentile_rank",
    "required_tp_distance",
    "reversal_confirmed",
    "spread_gate_result",
    "volatility_gate_result",
    "TrendResult",
    "classify_trend",
    "classify_trend_values",
]
