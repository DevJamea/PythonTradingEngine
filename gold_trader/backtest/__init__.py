"""Backtesting layer: engine (event loop over closed candles) and metrics."""
from .engine import BacktestEngine, BacktestResult
from .metrics import BacktestMetrics, TradeResult, compute_metrics, format_metrics

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestMetrics",
    "TradeResult",
    "compute_metrics",
    "format_metrics",
]
