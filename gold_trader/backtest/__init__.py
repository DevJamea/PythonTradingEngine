"""Backtesting layer: engines (event loop over closed candles), metrics,
cost model and offline data helpers.

Two engines coexist, mirroring the two strategy engines:

* :class:`~gold_trader.backtest.engine.BacktestEngine` -- the original
  fixed-spread simulation of the trend baseline (untouched methodology);
* :class:`~gold_trader.backtest.scalping_engine.ScalpingBacktestEngine` -- a
  fill-geometry simulation with a real per-bar spread for the scalper.

:mod:`~gold_trader.backtest.analysis` adds the validation layer on top of
either result (walk-forward folds, Monte Carlo, Sharpe/Sortino, spread stress).
"""
from .cost_model import CostModel, fixed_round_trip_spread, spread_summary
from .engine import BacktestEngine, BacktestResult
from .metrics import BacktestMetrics, TradeResult, compute_metrics, format_metrics
from .scalping_engine import (
    ScalpingBacktestEngine,
    ScalpingBacktestResult,
    ScalpingTradeRecord,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestMetrics",
    "CostModel",
    "ScalpingBacktestEngine",
    "ScalpingBacktestResult",
    "ScalpingTradeRecord",
    "TradeResult",
    "compute_metrics",
    "fixed_round_trip_spread",
    "format_metrics",
    "spread_summary",
]
