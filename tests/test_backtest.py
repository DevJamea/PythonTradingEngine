"""Unit tests for the backtest engine and metrics (deterministic)."""
from __future__ import annotations

import pandas as pd
import pytest

from gold_trader.backtest.engine import BacktestEngine
from gold_trader.backtest.metrics import (
    TradeResult,
    compute_metrics,
    format_metrics,
)
from gold_trader.models import default_gold_spec
from tests._helpers import build_df, make_cfg, make_uptrend_with_engulfing

SPEC = default_gold_spec()


def _ts(offset_days: int) -> pd.Timestamp:
    return pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(days=offset_days)


def make_trade(pnl: float, equity: float, day: int, side: str = "BUY") -> TradeResult:
    return TradeResult(
        entry_time=_ts(day),
        exit_time=_ts(day) + pd.Timedelta(days=1),
        side=side,
        entry=2000.0,
        exit_price=2001.0,
        sl=1995.0,
        tp=2010.0,
        volume=0.1,
        pnl=pnl,
        reason="take_profit",
        equity_after=equity,
    )


class TestMetrics:
    def test_known_values(self):
        trades = [make_trade(100.0, 10_100.0, 1), make_trade(-40.0, 10_060.0, 2), make_trade(10.0, 10_070.0, 3)]
        m = compute_metrics(trades, 10_000.0)
        assert m.total_trades == 3
        assert m.wins == 2
        assert m.losses == 1
        assert m.win_rate == pytest.approx(2 / 3)
        assert m.gross_profit == pytest.approx(110.0)
        assert m.gross_loss == pytest.approx(-40.0)
        assert m.net_profit == pytest.approx(70.0)
        assert m.max_drawdown == pytest.approx(40.0)
        assert m.max_drawdown_pct == pytest.approx(40.0 / 10_100.0)
        assert m.profit_factor == pytest.approx(110.0 / 40.0)
        assert m.average_win == pytest.approx(55.0)
        assert m.average_loss == pytest.approx(-40.0)
        assert m.final_equity == pytest.approx(10_070.0)

    def test_no_losses_gives_infinite_profit_factor(self):
        m = compute_metrics([make_trade(50.0, 10_050.0, 1)], 10_000.0)
        assert m.profit_factor == float("inf")
        assert m.average_loss == 0.0

    def test_empty_trade_list(self):
        m = compute_metrics([], 10_000.0)
        assert m.total_trades == 0
        assert m.win_rate == 0.0
        assert m.net_profit == 0.0
        assert m.max_drawdown == 0.0
        assert m.final_equity == 10_000.0

    def test_format_contains_headlines(self):
        m = compute_metrics([make_trade(10.0, 10_010.0, 1)], 10_000.0)
        text = format_metrics(m, 10_000.0)
        for token in ("Total trades", "Win rate", "Max drawdown", "Profit factor", "Net profit"):
            assert token in text


class TestEngine:
    def test_uptrend_scenario_trades_and_hits_take_profit(self):
        df = make_uptrend_with_engulfing()
        # extend with 40 more rising candles so the TP is reached
        extra = []
        price = float(df["close"].iloc[-1])
        for _ in range(40):
            o = price
            c = o + 0.8
            extra.append((o, c + 0.2, o - 0.2, c))
            price = c
        df = build_df(
            [
                (
                    float(df["open"].iloc[i]),
                    float(df["high"].iloc[i]),
                    float(df["low"].iloc[i]),
                    float(df["close"].iloc[i]),
                )
                for i in range(len(df))
            ]
            + extra
        )

        cfg = make_cfg(backtest_initial_balance=10_000.0)
        engine = BacktestEngine(cfg)
        result = engine.run(df, spec=SPEC)

        assert result.metrics.total_trades >= 1
        trade = result.trades[0]
        assert trade.side == "BUY"
        assert trade.reason == "take_profit"
        # entry must be the OPEN of the candle after the signal candle
        signal_index = len(make_uptrend_with_engulfing()) - 1
        assert trade.entry == pytest.approx(float(df["open"].iloc[signal_index + 1]))
        assert trade.pnl > 0
        assert result.final_equity > result.initial_balance

    def test_metrics_consistent_with_trades(self):
        df = make_uptrend_with_engulfing()
        extra = []
        price = float(df["close"].iloc[-1])
        for _ in range(40):
            o = price
            c = o + 0.8
            extra.append((o, c + 0.2, o - 0.2, c))
            price = c
        full = build_df(
            [
                (
                    float(df["open"].iloc[i]),
                    float(df["high"].iloc[i]),
                    float(df["low"].iloc[i]),
                    float(df["close"].iloc[i]),
                )
                for i in range(len(df))
            ]
            + extra
        )
        result = BacktestEngine(make_cfg()).run(full, spec=SPEC)
        assert result.metrics.net_profit == pytest.approx(sum(t.pnl for t in result.trades))
        assert result.metrics.final_equity == pytest.approx(
            result.initial_balance + sum(t.pnl for t in result.trades)
        )

    def test_deterministic_across_runs(self):
        df = make_uptrend_with_engulfing()
        result_a = BacktestEngine(make_cfg()).run(df, spec=SPEC)
        result_b = BacktestEngine(make_cfg()).run(df, spec=SPEC)
        assert result_a.metrics == result_b.metrics
        assert [t.pnl for t in result_a.trades] == [t.pnl for t in result_b.trades]

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            BacktestEngine(make_cfg()).run(build_df([(2000, 2001, 1999, 2000.5)] * 10))

    def test_stop_loss_is_hit_before_tp_when_both_touched(self):
        """Conservative rule: a candle spanning both SL and TP exits at SL."""
        rows = make_uptrend_with_engulfing()
        base = [
            (
                float(rows["open"].iloc[i]),
                float(rows["high"].iloc[i]),
                float(rows["low"].iloc[i]),
                float(rows["close"].iloc[i]),
            )
            for i in range(len(rows))
        ]
        # after the signal candle, add a wild candle that touches both SL and TP
        last = base[-1]
        entry = last[3] + 0.0  # entry = open of the NEXT candle
        extra = [
            (last[3], entry + 40.0, entry - 40.0, entry + 0.2),  # huge range
            (entry + 0.2, entry + 0.4, entry + 0.1, entry + 0.3),
        ]
        df = build_df(base + extra)
        result = BacktestEngine(make_cfg()).run(df, spec=SPEC)
        assert result.metrics.total_trades == 1
        assert result.trades[0].reason == "stop_loss"
        assert result.trades[0].exit_price == pytest.approx(result.trades[0].sl)
