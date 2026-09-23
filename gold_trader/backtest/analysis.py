"""Validation layer for backtest results: walk-forward, Monte Carlo, ratios.

This module never re-decides anything a strategy decided -- it only measures.
That separation is deliberate: the same three measurements (walk-forward folds,
Monte Carlo resampling, risk ratios) are applied to the trend baseline and to
the scalper, so the two can be compared with one yardstick and neither can be
flattered by a different measuring tape.

Definitions used throughout (stated so a report cannot be quietly re-defined
later):

* **Profit factor** -- gross profit / abs(gross loss). ``< 1`` means the system
  lost money. A strategy only counts as *working* above
  :data:`MIN_PROFIT_FACTOR`.
* **Sharpe / Sortino** -- computed from the *per-trade return* series (each
  trade's P/L over the equity before that trade), annualised with
  ``trades per year`` derived from the actual entry timestamps of the run.
  Per-trade (not bar-by-bar) because a scalper's intra-day equity path is not a
  stationary series; annualising a per-trade series by calendar bars would
  invent volatility it never had.
* **Walk-forward** -- rolling train/test windows inside the *development*
  period only. Parameters may be chosen on the train window of each fold; the
  test window is never used for choosing.
* **Isolated period** -- the last stretch of data that the study may look at
  exactly once, after everything else is fixed. No selection, no tuning.

Randomness appears only inside :func:`monte_carlo_simulation`, and it is
seeded, so every run of the same report is byte-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..backtest.metrics import BacktestMetrics, TradeResult, compute_metrics

#: Acceptance bar used by every verdict in this project. A strategy passes only
#: if it clears BOTH on the isolated period: profit factor and Sharpe.
MIN_PROFIT_FACTOR = 1.10
MIN_SHARPE = 0.50
#: Monte Carlo "ruin" = equity falls below this fraction of the starting
#: balance (a 50% drawdown is treated as the end of the account, not a dip).
RUIN_FRACTION = 0.50


# ---------------------------------------------------------------------------
# ratios
# ---------------------------------------------------------------------------

def per_trade_returns(
    trades: Sequence[TradeResult], initial_balance: float
) -> List[float]:
    """P/L of each trade divided by the equity *before* that trade."""
    returns: List[float] = []
    equity = float(initial_balance)
    for trade in trades:
        if equity > 0:
            returns.append(float(trade.pnl) / equity)
        equity = trade.equity_after
    return returns


def span_years(trades: Sequence[TradeResult]) -> float:
    """Elapsed years covered by the trade list (0.0 when fewer than 2 trades)."""
    if len(trades) < 2:
        return 0.0
    start = pd.Timestamp(min(t.entry_time for t in trades))
    end = pd.Timestamp(max(t.exit_time for t in trades))
    seconds = (end - start).total_seconds()
    if seconds <= 0:
        return 0.0
    return seconds / (365.25 * 24 * 3600)


def trades_per_year(trades: Sequence[TradeResult]) -> float:
    """Activity rate used to annualise a per-trade ratio (0.0 if undefined)."""
    years = span_years(trades)
    if years <= 0:
        return 0.0
    return len(trades) / years


def sharpe_ratio(returns: Sequence[float], periods_per_year: float) -> float:
    """Annualised Sharpe of a return series (0.0 when it cannot be defined)."""
    values = np.asarray(list(returns), dtype=float)
    if values.size < 2 or periods_per_year <= 0:
        return 0.0
    std = float(values.std(ddof=1))
    if std <= 0 or not math.isfinite(std):
        return 0.0
    return float(values.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(returns: Sequence[float], periods_per_year: float) -> float:
    """Annualised Sortino ratio (downside deviation = RMS of negative returns)."""
    values = np.asarray(list(returns), dtype=float)
    if values.size < 2 or periods_per_year <= 0:
        return 0.0
    downside = values[values < 0.0]
    if downside.size == 0:
        return math.inf if float(values.mean()) > 0 else 0.0
    downside_rms = float(np.sqrt(np.mean(downside**2)))
    if downside_rms <= 0 or not math.isfinite(downside_rms):
        return 0.0
    return float(values.mean() / downside_rms * math.sqrt(periods_per_year))


@dataclass(frozen=True)
class RiskRatios:
    """Sharpe/Sortino + the activity assumptions behind them."""

    sharpe: float
    sortino: float
    trades_per_year: float
    span_years: float
    mean_trade_return: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "trades_per_year": self.trades_per_year,
            "span_years": self.span_years,
            "mean_trade_return": self.mean_trade_return,
        }


def risk_ratios(trades: Sequence[TradeResult], initial_balance: float) -> RiskRatios:
    """Compute the risk ratios of one run (per-trade basis, see module docs)."""
    returns = per_trade_returns(trades, initial_balance)
    ppy = trades_per_year(trades)
    return RiskRatios(
        sharpe=sharpe_ratio(returns, ppy),
        sortino=sortino_ratio(returns, ppy),
        trades_per_year=ppy,
        span_years=span_years(trades),
        mean_trade_return=float(np.mean(returns)) if returns else 0.0,
    )


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    """Pass/fail against :data:`MIN_PROFIT_FACTOR` and :data:`MIN_SHARPE`."""

    passed: bool
    profit_factor: float
    sharpe: float
    reasons: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "profit_factor": self.profit_factor,
            "sharpe": self.sharpe,
            "reasons": list(self.reasons),
        }


def verdict(
    metrics: BacktestMetrics,
    ratios: RiskRatios,
    min_profit_factor: float = MIN_PROFIT_FACTOR,
    min_sharpe: float = MIN_SHARPE,
) -> Verdict:
    """Strict verdict: no trade-count or "activity" credit, only these two."""
    reasons: List[str] = []
    if metrics.total_trades == 0:
        reasons.append("no trades were taken (nothing to judge)")
    if not math.isfinite(metrics.profit_factor) or metrics.profit_factor < min_profit_factor:
        reasons.append(
            f"profit factor {metrics.profit_factor:.3f} < required {min_profit_factor:.2f}"
        )
    if ratios.sharpe < min_sharpe:
        reasons.append(f"sharpe {ratios.sharpe:.3f} < required {min_sharpe:.2f}")
    return Verdict(
        passed=not reasons,
        profit_factor=float(metrics.profit_factor),
        sharpe=float(ratios.sharpe),
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonteCarloReport:
    """Distribution of outcomes when the same trade set is re-ordered."""

    runs: int
    block_size: int
    median_net: float
    p05_net: float
    p95_net: float
    probability_of_loss: float
    probability_of_ruin: float
    median_max_drawdown: float
    p95_max_drawdown: float
    worst_case_equity: float
    positive_net_share: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "runs": self.runs,
            "block_size": self.block_size,
            "median_net": self.median_net,
            "p05_net": self.p05_net,
            "p95_net": self.p95_net,
            "probability_of_loss": self.probability_of_loss,
            "probability_of_ruin": self.probability_of_ruin,
            "median_max_drawdown": self.median_max_drawdown,
            "p95_max_drawdown": self.p95_max_drawdown,
            "worst_case_equity": self.worst_case_equity,
            "positive_net_share": self.positive_net_share,
        }


def monte_carlo_simulation(
    trades: Sequence[TradeResult],
    initial_balance: float,
    runs: int = 1000,
    block_size: int = 5,
    seed: int = 20260923,
) -> MonteCarloReport:
    """Moving-block bootstrap over the trade P/Ls of one run.

    Blocks (not iid draws) keep losing streaks intact -- a scalper's damage is
    concentrated in sequences, and shuffling trades independently would erase
    exactly that risk. Each run re-samples until it has as many trades as the
    original, then the equity path is replayed to measure net P/L, max drawdown
    and ruin (equity below ``RUIN_FRACTION`` of the start).

    Deterministic for a given ``seed``; returns an all-zero report when there
    are too few trades to resample (never a silent 100%-safe answer).
    """
    pnls = np.asarray([float(t.pnl) for t in trades], dtype=float)
    if pnls.size < max(block_size, 2) or initial_balance <= 0:
        return MonteCarloReport(0, block_size, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rng = np.random.default_rng(int(seed))
    n = int(pnls.size)
    block = max(1, int(block_size))
    blocks_needed = int(math.ceil(n / block))
    max_start = n - block  # a block always starts on a real trade

    nets = np.empty(runs, dtype=float)
    draws = np.empty((runs, n), dtype=float)
    for r in range(int(runs)):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        path = np.concatenate([pnls[s : s + block] for s in starts])[:n]
        draws[r] = path
        nets[r] = float(path.sum())

    equity_paths = initial_balance + np.cumsum(draws, axis=1)
    curves = np.hstack([np.full((draws.shape[0], 1), float(initial_balance)), equity_paths])
    # per-run equity path: peak of the running curve, drawdown, lowest trough
    running_peak = np.maximum.accumulate(curves, axis=1)
    drawdowns = np.max(running_peak - curves, axis=1)
    lowest = np.min(curves, axis=1)

    ruin = float(np.mean(lowest < initial_balance * (1.0 - RUIN_FRACTION)))
    return MonteCarloReport(
        runs=int(runs),
        block_size=block,
        median_net=float(np.median(nets)),
        p05_net=float(np.percentile(nets, 5)),
        p95_net=float(np.percentile(nets, 95)),
        probability_of_loss=float(np.mean(nets <= 0.0)),
        probability_of_ruin=ruin,
        median_max_drawdown=float(np.median(drawdowns)),
        p95_max_drawdown=float(np.percentile(drawdowns, 95)),
        worst_case_equity=float(lowest.min()),
        positive_net_share=float(np.mean(nets > 0.0)),
    )


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

EngineFactory = Callable[[Any], Any]


@dataclass(frozen=True)
class WalkForwardFold:
    """One train/test fold of a walk-forward study."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen: Dict[str, Any]
    train_profit_factor: float
    train_trades: int
    test_profit_factor: float
    test_trades: int
    test_net_profit: float
    note: str = ""


@dataclass(frozen=True)
class WalkForwardReport:
    """Per-fold detail plus the stitched out-of-sample result."""

    folds: Tuple[WalkForwardFold, ...]
    oos_metrics: BacktestMetrics
    oos_trades: Tuple[TradeResult, ...]
    skipped_folds: Tuple[str, ...] = ()

    @property
    def folds_passed(self) -> int:
        """How many test windows had profit factor above 1.0."""
        return sum(1 for f in self.folds if f.test_profit_factor > 1.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "folds": len(self.folds),
            "folds_profitable": self.folds_passed,
            "oos_trades": self.oos_metrics.total_trades,
            "oos_profit_factor": self.oos_metrics.profit_factor,
            "oos_net_profit": self.oos_metrics.net_profit,
            "skipped_folds": list(self.skipped_folds),
        }


def _safe_pf(metrics: BacktestMetrics) -> float:
    """Profit factor for ranking, with the two degenerate cases handled.

    A fold with zero trades scores 0.0 -- it must never "win" by having no
    losses. An infinite profit factor (no losing trade at all) is a real,
    maximal score and is kept as such; the callers display it as ``inf``.
    """
    if metrics.total_trades == 0:
        return 0.0
    return float(metrics.profit_factor)


def walk_forward_study(
    df: pd.DataFrame,
    base_config: Any,
    param_grid: Sequence[Dict[str, Any]],
    engine_factory: EngineFactory,
    train_months: int = 6,
    test_months: int = 2,
    step_months: Optional[int] = None,
    start: str = "2021-01-01",
    end: str = "2024-01-01",
    spec: Any = None,
) -> WalkForwardReport:
    """Rolling walk-forward over ``[start, end)`` with grid selection per fold.

    For every fold: score *every* candidate configuration on the train window,
    keep the one with the best profit factor (ties -> the earlier candidate,
    i.e. the simpler/default one), then evaluate it on the following
    ``test_months`` and collect only those out-of-sample trades. The stitched
    OOS trade list -- not the best fold -- is the answer.

    The window rolls forward by ``step_months`` (default: ``test_months``), so
    each month of the development period is scored out of sample once instead
    of only the one test window that follows each training block.
    """
    if train_months < 1 or test_months < 1:
        raise ValueError("train_months and test_months must be >= 1")
    if not param_grid:
        raise ValueError("param_grid must not be empty")
    step = int(test_months if step_months is None else step_months)
    if step < 1:
        raise ValueError("step_months must be >= 1")
    src = df.copy()
    src["time"] = pd.to_datetime(src["time"], utc=True)
    window_start = pd.Timestamp(start, tz="UTC")
    window_end = pd.Timestamp(end, tz="UTC")

    folds: List[WalkForwardFold] = []
    oos_trades: List[TradeResult] = []
    skipped: List[str] = []

    cursor = window_start
    while True:
        train_end = cursor + pd.DateOffset(months=int(train_months))
        test_end = train_end + pd.DateOffset(months=int(test_months))
        if train_end >= window_end:
            break
        train = src[(src["time"] >= cursor) & (src["time"] < train_end)].reset_index(drop=True)
        test = src[(src["time"] >= train_end) & (src["time"] < test_end)].reset_index(drop=True)
        if len(test) == 0:
            skipped.append(f"{cursor.date()}: empty test window")
            cursor = train_end
            continue
        best_cfg = None
        best_pf = -1.0
        best_metrics: Optional[BacktestMetrics] = None
        chosen_overrides: Dict[str, Any] = {}
        for overrides in param_grid:
            candidate = replace(base_config, **overrides)
            metrics = engine_factory(candidate).run(train, spec=spec).metrics
            pf = _safe_pf(metrics)
            if pf > best_pf:
                best_pf, best_cfg, best_metrics, chosen_overrides = pf, candidate, metrics, dict(overrides)
        if best_cfg is None or best_metrics is None:  # pragma: no cover - defensive
            raise RuntimeError("walk-forward produced no candidate configuration")
        test_result = engine_factory(best_cfg).run(test, spec=spec)
        folds.append(
            WalkForwardFold(
                train_start=cursor,
                train_end=train_end,
                test_start=train_end,
                test_end=min(test_end, window_end),
                chosen=chosen_overrides,
                train_profit_factor=_safe_pf(best_metrics),
                train_trades=best_metrics.total_trades,
                test_profit_factor=_safe_pf(test_result.metrics),
                test_trades=test_result.metrics.total_trades,
                test_net_profit=float(test_result.metrics.net_profit),
            )
        )
        oos_trades.extend(test_result.trades)
        cursor = cursor + pd.DateOffset(months=step)

    oos = list(oos_trades)
    oos.sort(key=lambda t: t.exit_time)
    metrics = compute_metrics(oos, float(base_config.backtest_initial_balance))
    return WalkForwardReport(
        folds=tuple(folds),
        oos_metrics=metrics,
        oos_trades=tuple(oos),
        skipped_folds=tuple(skipped),
    )


# ---------------------------------------------------------------------------
# scenarios / formatting
# ---------------------------------------------------------------------------

def scenario_from_config(base_config: Any, **overrides: Any) -> Any:
    """Immutable config override helper used to build stress scenarios."""
    return replace(base_config, **overrides)


def format_ratios(ratios: RiskRatios, metrics: BacktestMetrics, label: str = "") -> str:
    """One-line, aligned summary for console/report tables."""
    pf = "  inf" if not math.isfinite(metrics.profit_factor) else f"{metrics.profit_factor:.2f}"
    prefix = f"{label:<18}" if label else ""
    return (
        f"{prefix}trades={metrics.total_trades:>5}  win={metrics.win_rate:6.2%}  "
        f"PF={pf:>6}  net={metrics.net_profit:12,.2f}  "
        f"Sharpe={ratios.sharpe:6.2f}  Sortino={ratios.sortino:7.2f}"
    )


def monte_carlo_text(report: MonteCarloReport) -> str:
    """Human-readable Monte Carlo summary (Arabic-labelled output is added by
    the study script; this stays ASCII so it can be diffed in tests)."""
    if report.runs == 0:
        return "Monte Carlo: not enough trades to resample (no result)."
    return (
        f"Monte Carlo ({report.runs} runs, block={report.block_size}): "
        f"median net {report.median_net:,.2f} | p05 {report.p05_net:,.2f} | "
        f"p95 {report.p95_net:,.2f} | P(loss) {report.probability_of_loss:.1%} | "
        f"P(ruin -50%) {report.probability_of_ruin:.1%} | "
        f"p95 max DD {report.p95_max_drawdown:,.2f}"
    )


def format_walk_forward(report: WalkForwardReport) -> str:  # pragma: no cover - display helper
    """Fold table (used by the study script and by humans reading the log)."""
    lines = [
        "walk-forward folds (train -> test, selection on train only):",
        f"{'train':<12}{'test':<12}{'chosen':<46}{'PFtr':>6}{'PFte':>6}{'Nte':>6}",
    ]
    for fold in report.folds:
        chosen = ",".join(f"{k}={v}" for k, v in sorted(fold.chosen.items())) or "-"
        lines.append(
            f"{fold.train_start.date()}  {fold.test_start.date()}  "
            f"{chosen[:44]:<46}{fold.train_profit_factor:6.2f}"
            f"{fold.test_profit_factor:6.2f}{fold.test_trades:6d}"
        )
    lines.append(
        f"stitched OOS: trades={report.oos_metrics.total_trades} "
        f"PF={report.oos_metrics.profit_factor:.3f} "
        f"net={report.oos_metrics.net_profit:,.2f} "
        f"profitable folds={report.folds_passed}/{len(report.folds)}"
    )
    if report.skipped_folds:
        lines.append("skipped: " + "; ".join(report.skipped_folds))
    return "\n".join(lines)
