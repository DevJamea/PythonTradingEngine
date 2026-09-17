"""Backtest metrics: pure, deterministic trade statistics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class TradeResult:
    """One simulated trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # "BUY" | "SELL"
    entry: float
    exit_price: float
    sl: float
    tp: float
    volume: float
    pnl: float
    reason: str  # "stop_loss" | "take_profit" | "end_of_data"
    equity_after: float


@dataclass(frozen=True)
class BacktestMetrics:
    """Summary statistics for a backtest run."""

    total_trades: int
    wins: int
    losses: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    average_win: float
    average_loss: float
    final_equity: float


def compute_metrics(
    trades: Sequence[TradeResult], initial_balance: float
) -> BacktestMetrics:
    """Compute all headline metrics from a trade list.

    Conventions: a trade with pnl > 0 is a win; pnl <= 0 counts as a loss
    (conservative, includes breakeven trades). ``profit_factor`` is ``inf``
    when there are no losing trades and gross profit is positive.
    """
    pnls = [t.pnl for t in trades]
    total = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = total - wins
    win_rate = wins / total if total else 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(p for p in pnls if p < 0)
    net_profit = sum(pnls)

    equity_curve = [float(initial_balance)] + [t.equity_after for t in trades]
    peak = equity_curve[0]
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            drawdown = peak - equity
            max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_pct = max(max_drawdown_pct, drawdown / peak)

    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    else:
        profit_factor = math.inf if gross_profit > 0 else 0.0

    losing_count = sum(1 for p in pnls if p < 0)
    average_win = gross_profit / wins if wins else 0.0
    average_loss = gross_loss / losing_count if losing_count else 0.0

    return BacktestMetrics(
        total_trades=total,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        profit_factor=profit_factor,
        average_win=average_win,
        average_loss=average_loss,
        final_equity=equity_curve[-1],
    )


def format_metrics(metrics: BacktestMetrics, initial_balance: float) -> str:
    """Human-readable multi-line report for CLI output."""
    if math.isinf(metrics.profit_factor):
        pf = "inf"
    else:
        pf = f"{metrics.profit_factor:.2f}"
    lines = [
        "=== Backtest results ===",
        f"Initial balance:  {initial_balance:,.2f}",
        f"Final equity:     {metrics.final_equity:,.2f}",
        f"Total trades:     {metrics.total_trades}",
        f"Wins / Losses:    {metrics.wins} / {metrics.losses}",
        f"Win rate:         {metrics.win_rate:.2%}",
        f"Gross profit:     {metrics.gross_profit:,.2f}",
        f"Gross loss:       {metrics.gross_loss:,.2f}",
        f"Net profit:       {metrics.net_profit:,.2f}",
        f"Max drawdown:     {metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2%})",
        f"Profit factor:    {pf}",
        f"Average win:      {metrics.average_win:,.2f}",
        f"Average loss:     {metrics.average_loss:,.2f}",
        "NOTE: past/indicated results do not guarantee future profits.",
    ]
    return "\n".join(lines)
