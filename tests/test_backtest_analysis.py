"""Unit tests for the validation layer: walk-forward, Monte Carlo, ratios.

The stub engine below makes these tests exact: the numbers come from the
analysis code, not from a market, so a methodology regression fails loudly.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from gold_trader.backtest.analysis import (
    MIN_PROFIT_FACTOR,
    MIN_SHARPE,
    RUIN_FRACTION,
    RiskRatios,
    WalkForwardFold,
    compute_metrics,
    format_ratios,
    format_walk_forward,
    monte_carlo_simulation,
    monte_carlo_text,
    per_trade_returns,
    risk_ratios,
    scenario_from_config,
    sharpe_ratio,
    sortino_ratio,
    span_years,
    trades_per_year,
    verdict,
    walk_forward_study,
)
from gold_trader.backtest.metrics import BacktestMetrics, TradeResult
from gold_trader.config import Config
from tests._helpers import make_cfg


def _trade(pnl: float, day: int, balance_before: float = 10_000.0) -> TradeResult:
    entry = pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(days=day)
    return TradeResult(
        entry_time=entry,
        exit_time=entry + pd.Timedelta(hours=1),
        side="BUY" if pnl >= 0 else "SELL",
        entry=2000.0,
        exit_price=2000.0 + pnl,
        sl=1995.0,
        tp=2010.0,
        volume=0.1,
        pnl=pnl,
        reason="take_profit" if pnl > 0 else "stop_loss",
        equity_after=balance_before + pnl,
    )


def _metrics(gross_profit: float, gross_loss: float, trades: int = 10) -> BacktestMetrics:
    return compute_metrics(
        [_trade(p, i) for i, p in enumerate([gross_profit / 5] * 5 + [gross_loss / 5] * 5)],
        10_000.0,
    )


# ---------------------------------------------------------------------------
# per-trade returns and ratios
# ---------------------------------------------------------------------------

class TestRatios:
    def test_per_trade_returns_use_equity_before_the_trade(self):
        trades = [_trade(100.0, 1), _trade(-50.0, 2, 10_100.0), _trade(1.0, 3, 10_050.0)]
        returns = per_trade_returns(trades, 10_000.0)
        assert returns[0] == pytest.approx(100.0 / 10_000.0)
        assert returns[1] == pytest.approx(-50.0 / 10_100.0)
        assert returns[2] == pytest.approx(1.0 / 10_050.0)

    def test_span_and_activity(self):
        trades = [_trade(1.0, 0), _trade(1.0, 364)]
        assert span_years(trades) == pytest.approx(364 / 365.25, rel=1e-3)
        assert trades_per_year(trades) == pytest.approx(2 / span_years(trades))
        assert span_years([_trade(1.0, 0)]) == 0.0
        assert trades_per_year([_trade(1.0, 0)]) == 0.0

    def test_sharpe_matches_the_formula(self):
        returns = [0.01, -0.02, 0.03, -0.005]
        arr = np.asarray(returns)
        expected = arr.mean() / arr.std(ddof=1) * math.sqrt(4.0)
        assert sharpe_ratio(returns, periods_per_year=4.0) == pytest.approx(expected)

    def test_sortino_only_punishes_downside(self):
        gains = [0.02, 0.01, 0.03, -0.01]
        arr = np.asarray(gains)
        downside = arr[arr < 0]
        expected = arr.mean() / float(np.sqrt(np.mean(downside**2))) * math.sqrt(9.0)
        assert sortino_ratio(gains, periods_per_year=9.0) == pytest.approx(expected)

    def test_sortino_is_infinite_without_a_single_loss(self):
        assert sortino_ratio([0.01, 0.02, 0.03], 4.0) == math.inf

    def test_degenerate_inputs_return_zero_not_crash(self):
        assert sharpe_ratio([], 4.0) == 0.0
        assert sharpe_ratio([0.01], 4.0) == 0.0
        assert sharpe_ratio([0.01, 0.01], 0.0) == 0.0
        assert sharpe_ratio([0.0, 0.0], 4.0) == 0.0  # zero variance
        assert sortino_ratio([0.0, 0.0], 4.0) == 0.0
        assert sortino_ratio([-0.01, 0.0], 4.0) != 0.0

    def test_risk_ratios_from_a_trade_list(self):
        trades = [_trade(100.0, i * 10) for i in range(4)] + [_trade(-40.0, 45 + i * 3) for i in range(3)]
        ratios = risk_ratios(trades, 10_000.0)
        assert isinstance(ratios, RiskRatios)
        assert ratios.trades_per_year > 0
        assert ratios.sortino > 0
        assert set(ratios.as_dict()) == {
            "sharpe",
            "sortino",
            "trades_per_year",
            "span_years",
            "mean_trade_return",
        }

    def test_empty_run_has_zero_ratios(self):
        ratios = risk_ratios([], 10_000.0)
        assert ratios.sharpe == 0.0 and ratios.sortino == 0.0
        assert ratios.span_years == 0.0


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

class TestVerdict:
    def test_defaults_are_the_documented_bar(self):
        assert MIN_PROFIT_FACTOR == pytest.approx(1.10)
        assert MIN_SHARPE == pytest.approx(0.50)
        assert RUIN_FRACTION == pytest.approx(0.50)

    def test_a_losing_strategy_fails_even_with_many_trades(self):
        metrics = _metrics(gross_profit=400.0, gross_loss=-500.0, trades=240)
        ratios = RiskRatios(sharpe=-0.4, sortino=-0.2, trades_per_year=90.0, span_years=2.6, mean_trade_return=-0.001)
        result = verdict(metrics, ratios)
        assert result.passed is False
        assert any("profit factor" in reason for reason in result.reasons)
        assert any("sharpe" in reason for reason in result.reasons)

    def test_a_profitable_but_choppy_strategy_still_fails(self):
        metrics = _metrics(gross_profit=500.0, gross_loss=-400.0)
        ratios = RiskRatios(sharpe=0.1, sortino=0.2, trades_per_year=50.0, span_years=2.0, mean_trade_return=0.0005)
        result = verdict(metrics, ratios)
        assert result.passed is False
        assert any("sharpe" in reason for reason in result.reasons)

    def test_a_strategy_that_clears_both_bars_passes(self):
        metrics = _metrics(gross_profit=900.0, gross_loss=-400.0)
        ratios = RiskRatios(sharpe=1.4, sortino=2.1, trades_per_year=50.0, span_years=2.0, mean_trade_return=0.004)
        assert verdict(metrics, ratios).passed is True

    def test_no_trades_is_never_a_pass(self):
        empty = compute_metrics([], 10_000.0)
        result = verdict(empty, RiskRatios(0.0, 0.0, 0.0, 0.0, 0.0))
        assert result.passed is False
        assert "no trades were taken" in result.reasons[0]

    def test_infinite_profit_factor_without_trades_cannot_pass(self):
        metrics = compute_metrics([], 10_000.0)
        assert metrics.profit_factor == 0.0
        assert verdict(metrics, RiskRatios(9.9, 9.9, 1.0, 1.0, 1.0)).passed is False

    def test_verdict_is_serialisable(self):
        result = verdict(_metrics(900.0, -100.0), RiskRatios(2.0, 3.0, 10.0, 1.0, 0.01))
        assert result.as_dict()["passed"] is True


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_is_deterministic_for_a_seed(self):
        trades = [_trade(p, i) for i, p in enumerate([100, -50, 80, -30, 40, -60, 20, 90])]
        first = monte_carlo_simulation(trades, 10_000.0, runs=50, block_size=3)
        second = monte_carlo_simulation(trades, 10_000.0, runs=50, block_size=3)
        assert first == second
        assert first.runs == 50

    def test_different_seed_explores_different_paths(self):
        trades = [_trade(p, i) for i, p in enumerate([100, -50, 80, -30, 40, -60, 20, 90])]
        a = monte_carlo_simulation(trades, 10_000.0, runs=25, block_size=2, seed=1)
        b = monte_carlo_simulation(trades, 10_000.0, runs=25, block_size=2, seed=2)
        assert (a.median_net, a.p95_net) != (b.median_net, b.p95_net)

    def test_a_losing_trade_set_is_reported_as_losing(self):
        trades = [_trade(-p, i) for i, p in enumerate([50, 60, 40, 55, 70, 45, 65, 30])]
        report = monte_carlo_simulation(trades, 10_000.0, runs=40, block_size=4)
        assert report.median_net < 0
        assert report.probability_of_loss == pytest.approx(1.0)
        assert report.positive_net_share == pytest.approx(0.0)

    def test_a_winning_set_barely_risks_ruin(self):
        trades = [_trade(500.0, i) for i in range(12)]
        report = monte_carlo_simulation(trades, 10_000.0, runs=30, block_size=3)
        assert report.probability_of_ruin == pytest.approx(0.0)
        assert report.median_net > 0

    def test_ruin_fraction_is_applied_to_the_equity_trough(self):
        # four -3000 trades on a 10k account: the trough is below 50% => ruin
        trades = [_trade(-3_000.0, i) for i in range(8)]
        report = monte_carlo_simulation(trades, 10_000.0, runs=20, block_size=8)
        assert report.probability_of_ruin == pytest.approx(1.0)
        assert report.worst_case_equity < 10_000.0 * (1 - RUIN_FRACTION)

    def test_too_few_trades_gives_an_explicit_empty_report(self):
        report = monte_carlo_simulation([_trade(10.0, 0), _trade(-5.0, 1)], 10_000.0, runs=50, block_size=5)
        assert report.runs == 0
        assert report.median_net == 0.0
        assert "not enough trades" in monte_carlo_text(report)

    def test_a_full_length_block_reproduces_the_original_path_exactly(self):
        """block_size == number of trades leaves one possible draw: the run itself."""
        pnls = [-100.0] * 4 + [10.0] * 46
        trades = [_trade(p, i) for i, p in enumerate(pnls)]
        report = monte_carlo_simulation(trades, 10_000.0, runs=20, block_size=len(pnls))
        assert report.median_net == pytest.approx(sum(pnls))
        assert report.p05_net == pytest.approx(report.p95_net)

    def test_drawdown_of_a_monotone_losing_set_equals_its_loss(self):
        trades = [_trade(-p, i) for i, p in enumerate([50, 60, 40, 55, 70, 45, 65, 30])]
        report = monte_carlo_simulation(trades, 10_000.0, runs=25, block_size=3)
        assert report.median_max_drawdown == pytest.approx(abs(report.median_net))

    def test_invalid_balance_is_refused(self):
        trades = [_trade(p, i) for i, p in enumerate([1.0] * 10)]
        assert monte_carlo_simulation(trades, 0.0, runs=10).runs == 0


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

class _StubResult:
    def __init__(self, metrics, trades):
        self.metrics = metrics
        self.trades = trades


class _StubEngine:
    """Scores a window as ``cfg.max_spread * bar_count`` (fully predictable).

    Records every frame it was asked to run so a test can prove the selection
    step never sees the test window's P/L and the test step runs once.
    """

    calls: list = []

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def run(self, df, initial_balance=None, spec=None) -> _StubResult:
        score = float(self.cfg.max_spread) * len(df)
        _StubEngine.calls.append((self.cfg.max_spread, len(df), self.cfg.atr_tp_multiplier))
        trades = [_trade(score / 10.0, i) for i in range(3)]
        if self.cfg.atr_tp_multiplier < 0:  # "losing" candidate
            trades = [_trade(-score / 10.0, i) for i in range(3)]
        return _StubResult(compute_metrics(trades, 10_000.0), trades)


def _synthetic_hours(n: int, start: str = "2021-01-01 00:00:00") -> pd.DataFrame:
    times = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "open": 2000.0,
            "high": 2001.0,
            "low": 1999.0,
            "close": 2000.5,
        }
    )


class TestWalkForward:
    def setup_method(self):
        _StubEngine.calls = []

    def test_grid_is_scored_on_train_only_then_applied_to_the_test_window(self):
        _StubEngine.calls = []
        df = _synthetic_hours(4000)  # ~41 days of M15
        report = walk_forward_study(
            df,
            make_cfg(),
            param_grid=[
                {"max_spread": 0.20, "atr_tp_multiplier": -1.0},  # loses on train
                {"max_spread": 0.60, "atr_tp_multiplier": 1.0},   # wins on train
            ],
            engine_factory=_StubEngine,
            train_months=1,
            test_months=1,
            start="2021-01-01",
            end="2021-03-15",
        )
        assert report.folds, "expected at least one fold"
        for fold in report.folds:
            assert fold.chosen == {"max_spread": 0.60, "atr_tp_multiplier": 1.0}
            assert fold.train_profit_factor > 0
            # one scoring run per candidate + one test run per fold
            window_calls = [c for c in _StubEngine.calls if c[0] in (0.20, 0.60)]
            assert len(window_calls) >= 3
        assert isinstance(report.folds[0], WalkForwardFold)

    def test_rolling_step_scores_each_month_out_of_sample_once(self):
        df = _synthetic_hours(300 * 96)  # ~300 days of continuous M15 bars
        one_month_step = walk_forward_study(
            df,
            make_cfg(),
            param_grid=[{"max_spread": 0.60, "atr_tp_multiplier": 1.0}],
            engine_factory=_StubEngine,
            train_months=3,
            test_months=1,
            step_months=1,
            start="2021-01-01",
            end="2021-07-01",
        )
        two_month_step = walk_forward_study(
            df,
            make_cfg(),
            param_grid=[{"max_spread": 0.60, "atr_tp_multiplier": 1.0}],
            engine_factory=_StubEngine,
            train_months=3,
            test_months=1,
            step_months=2,
            start="2021-01-01",
            end="2021-07-01",
        )
        assert len(one_month_step.folds) == 3
        assert len(two_month_step.folds) == 2
        # distinct, non-overlapping test windows
        starts = {f.test_start for f in one_month_step.folds}
        assert len(starts) == len(one_month_step.folds)

    def test_non_positive_step_is_refused(self):
        with pytest.raises(ValueError):
            walk_forward_study(
                _synthetic_hours(500),
                make_cfg(),
                param_grid=[{"max_spread": 0.6}],
                engine_factory=_StubEngine,
                train_months=1,
                test_months=1,
                step_months=0,
            )

    def test_stitched_oos_result_is_the_concatenation_of_test_windows(self):
        df = _synthetic_hours(4000)
        report = walk_forward_study(
            df,
            make_cfg(),
            param_grid=[{"max_spread": 0.60, "atr_tp_multiplier": 1.0}],
            engine_factory=_StubEngine,
            train_months=1,
            test_months=1,
            start="2021-01-01",
            end="2021-04-01",
        )
        expected_trades = 3 * len(report.folds)
        assert report.oos_metrics.total_trades == expected_trades
        assert len(report.oos_trades) == expected_trades
        assert report.oos_trades == tuple(sorted(report.oos_trades, key=lambda t: t.exit_time))

    def test_invalid_arguments_raise(self):
        df = _synthetic_hours(500)
        with pytest.raises(ValueError):
            walk_forward_study(df, make_cfg(), [], _StubEngine, train_months=1, test_months=1)
        with pytest.raises(ValueError):
            walk_forward_study(
                df, make_cfg(), [{"max_spread": 0.2}], _StubEngine, train_months=0, test_months=1
            )

    def test_windows_without_data_are_reported_not_hidden(self):
        # a frame covering one week only: no fold can complete
        df = _synthetic_hours(200)
        report = walk_forward_study(
            df,
            make_cfg(),
            param_grid=[{"max_spread": 0.60, "atr_tp_multiplier": 1.0}],
            engine_factory=_StubEngine,
            train_months=1,
            test_months=1,
            start="2021-01-01",
            end="2021-06-01",
        )
        assert report.folds == ()
        assert report.oos_metrics.total_trades == 0
        assert "profitable folds=0/0" in format_walk_forward(report)

    def test_report_summary_keys(self):
        df = _synthetic_hours(4000)
        report = walk_forward_study(
            df,
            make_cfg(),
            param_grid=[{"max_spread": 0.60, "atr_tp_multiplier": 1.0}],
            engine_factory=_StubEngine,
            train_months=1,
            test_months=1,
            start="2021-01-01",
            end="2021-03-01",
        )
        data = report.as_dict()
        assert data["folds"] == len(report.folds)
        assert data["folds_profitable"] == report.folds_passed

    def test_format_lists_the_chosen_parameters(self):
        df = _synthetic_hours(4000)
        report = walk_forward_study(
            df,
            make_cfg(),
            param_grid=[{"max_spread": 0.60, "atr_tp_multiplier": 1.0}],
            engine_factory=_StubEngine,
            train_months=1,
            test_months=1,
            start="2021-01-01",
            end="2021-03-01",
        )
        text = format_walk_forward(report)
        assert "max_spread=0.6" in text
        assert "stitched OOS" in text


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_scenario_from_config_is_a_pure_copy(self):
        base = Config()
        scenario = scenario_from_config(base, backtest_spread_stress_multiplier=1.5)
        assert base.backtest_spread_stress_multiplier == 1.0
        assert scenario.backtest_spread_stress_multiplier == pytest.approx(1.5)
        assert isinstance(scenario, Config)

    def test_format_ratios_is_one_line_and_contains_the_headlines(self):
        metrics = _metrics(900.0, -400.0)
        ratios = RiskRatios(1.2, 2.0, 40.0, 2.5, 0.001)
        line = format_ratios(ratios, metrics, label="scalping M5")
        assert "\n" not in line
        for token in ("PF=", "Sharpe=", "Sortino=", "trades="):
            assert token in line

    def test_format_ratios_handles_infinite_profit_factor(self):
        metrics = compute_metrics([_trade(10.0, 0)], 10_000.0)
        assert math.isinf(metrics.profit_factor)
        assert "inf" in format_ratios(RiskRatios(1.0, 1.0, 1.0, 1.0, 0.1), metrics)

    def test_monte_carlo_text_reports_the_headline_numbers(self):
        trades = [_trade(p, i) for i, p in enumerate([100, -50, 80, -30, 40, -60, 20, 90])]
        text = monte_carlo_text(monte_carlo_simulation(trades, 10_000.0, runs=25, block_size=4))
        for token in ("Monte Carlo", "median net", "P(loss)", "P(ruin"):
            assert token in text

