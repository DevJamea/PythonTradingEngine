"""Performance statistics for research backtests (§7).

Reports BOTH R-based and account-equity performance. Conventions (documented):
* win = pnl_net > 0 (breakeven counts as a loss — conservative);
* expectancy_R / average_R = mean net-R per trade (after costs);
* Sharpe/Sortino are computed on per-trade net-R and annualised by observed
  trade frequency: Sharpe = sqrt(N_year) * mean(R)/std(R, ddof=1), where
  N_year = n_trades * 365.25 / span_days. Sortino uses downside deviation of
  R below 0. These are screening statistics, not fund-grade ratios;
* MaxDD in money and percent, drawdown duration in days (peak-to-recovery on
  the trade equity curve; unrecovered DD runs to the last trade);
* Ulcer Index = sqrt(mean(squared % drawdown)) on the trade equity curve;
* annualised return from (final/initial)^(365.25/span_days) - 1;
* aux t-stat = mean(R)/(std(R)/sqrt(n)) — AUXILIARY ONLY, never a gate
  (trade returns are not guaranteed independent).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerfMetrics:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    average_r: float
    median_r: float
    std_r: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_pct: float
    drawdown_duration_days: float
    ulcer_index: float
    consecutive_losses_max: int
    avg_trade_duration_bars: float
    avg_trade_duration_hours: float
    annualised_return: float
    return_over_maxdd: float
    total_costs: float
    cost_per_trade: float
    final_equity: float
    span_days: float
    t_stat_aux: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                d[k] = str(v)
        return d


def _pf(gross_profit: float, gross_loss: float) -> float:
    if gross_loss < 0:
        return gross_profit / abs(gross_loss)
    return math.inf if gross_profit > 0 else 0.0


def compute_performance(trades: list, initial_balance: float,
                        span_days: float) -> PerfMetrics:
    n = len(trades)
    pnls = np.array([t.pnl_net for t in trades], dtype=float) if n else np.zeros(0)
    rs = np.array([t.r_net for t in trades], dtype=float) if n else np.zeros(0)
    wins = int((pnls > 0).sum()) if n else 0
    losses = n - wins
    win_rate = wins / n if n else 0.0
    gross_profit = float(pnls[pnls > 0].sum()) if n else 0.0
    gross_loss = float(pnls[pnls < 0].sum()) if n else 0.0
    net_profit = float(pnls.sum()) if n else 0.0
    exp_r = float(rs.mean()) if n else 0.0
    med_r = float(np.median(rs)) if n else 0.0
    std_r = float(rs.std(ddof=1)) if n > 1 else 0.0

    span_days = max(span_days, 1e-9)
    n_year = n * 365.25 / span_days if span_days > 0 else 0.0
    if n > 1 and std_r > 0:
        sharpe = math.sqrt(n_year) * exp_r / std_r
    else:
        sharpe = 0.0
    downside = rs[rs < 0]
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = math.sqrt(n_year) * exp_r / float(downside.std(ddof=1))
    else:
        sortino = 0.0

    equity = [float(initial_balance)]
    for t in trades:
        equity.append(float(t.equity_after))
    equity_a = np.array(equity)
    peak = np.maximum.accumulate(equity_a)
    dd = peak - equity_a
    dd_pct = np.where(peak > 0, dd / peak, 0.0)
    max_dd = float(dd.max()) if n else 0.0
    max_dd_pct = float(dd_pct.max()) if n else 0.0
    ulcer = float(np.sqrt(np.mean(dd_pct ** 2))) if n else 0.0

    # drawdown duration: longest peak-to-recovery in days (trade-time based)
    dd_dur_days = 0.0
    if n:
        exit_times = pd.to_datetime([t.exit_time for t in trades], utc=True)
        start_t = pd.to_datetime(trades[0].entry_time, utc=True)
        times = [start_t] + list(exit_times)
        run_start: int | None = None
        best = 0.0
        cur_peak = equity_a[0]
        peak_t = times[0]
        for k in range(1, len(equity_a)):
            if equity_a[k] >= cur_peak:
                if run_start is not None:
                    best = max(best, (times[k] - peak_t).total_seconds() / 86400.0)
                    run_start = None
                cur_peak = equity_a[k]
                peak_t = times[k]
            elif run_start is None:
                run_start = k
        if run_start is not None:  # never recovered
            best = max(best, (times[-1] - peak_t).total_seconds() / 86400.0)
        dd_dur_days = best

    consec = cur = 0
    for p in pnls:
        cur = cur + 1 if p <= 0 else 0
        consec = max(consec, cur)

    dur_bars = float(np.mean([t.duration_bars for t in trades])) if n else 0.0
    final_equity = float(equity_a[-1])
    ann_ret = ((final_equity / initial_balance) ** (365.25 / span_days) - 1.0
               if initial_balance > 0 and final_equity > 0 else 0.0)
    romad = (net_profit / max_dd) if max_dd > 0 else 0.0
    total_costs = float(sum(t.cost_money for t in trades)) if n else 0.0
    t_stat = (exp_r / (std_r / math.sqrt(n))) if n > 1 and std_r > 0 else 0.0

    return PerfMetrics(
        total_trades=n, wins=wins, losses=losses, win_rate=win_rate,
        profit_factor=_pf(gross_profit, gross_loss),
        expectancy_r=exp_r, average_r=exp_r, median_r=med_r, std_r=std_r,
        gross_profit=gross_profit, gross_loss=gross_loss, net_profit=net_profit,
        sharpe=sharpe, sortino=sortino, max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct, drawdown_duration_days=dd_dur_days,
        ulcer_index=ulcer, consecutive_losses_max=consec,
        avg_trade_duration_bars=dur_bars,
        avg_trade_duration_hours=dur_bars * 0.25,
        annualised_return=ann_ret, return_over_maxdd=romad,
        total_costs=total_costs, cost_per_trade=(total_costs / n if n else 0.0),
        final_equity=final_equity, span_days=span_days, t_stat_aux=t_stat,
    )
