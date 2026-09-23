"""Unit tests for the scalping cost model, data helpers and backtest engine."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gold_trader.backtest.cost_model import (
    CostModel,
    fixed_round_trip_spread,
    spread_summary,
)
from gold_trader.backtest.data import (
    assess_quality,
    build_m1_from_bid_ask,
    load_candles_csv,
    month_windows,
    normalize_candles,
    resample_candles,
    split_period,
    spread_from_points,
)
from gold_trader.backtest.scalping_engine import (
    ScalpingBacktestEngine,
    ScalpingBacktestResult,
)
from gold_trader.config import Config
from gold_trader.models import default_gold_spec
from tests._helpers import build_df, make_cfg, make_scalp_frame, scalp_cfg

SPEC = default_gold_spec()


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_fixed_spread_is_two_times_the_legacy_per_side_estimate(self):
        cfg = make_cfg(backtest_spread_cost=0.15)
        assert fixed_round_trip_spread(cfg) == pytest.approx(0.30)
        cost = CostModel.from_frame(pd.DataFrame({"x": [1]}), cfg)
        assert cost.spread_at(0) == pytest.approx(0.30)

    def test_recorded_spread_is_off_by_default(self):
        """The switch must be opt-in: the legacy model stays fixed-spread."""
        cfg = make_cfg(backtest_spread_cost=0.15)
        frame = pd.DataFrame({"spread": [0.9, 0.9, 0.9]})
        assert CostModel.from_frame(frame, cfg).spread_at(0) == pytest.approx(0.30)
        enabled = replace(cfg, backtest_use_recorded_spread=True)
        assert CostModel.from_frame(frame, enabled).spread_at(0) == pytest.approx(0.90)

    def test_variable_spread_is_used_when_wider_than_the_floor(self):
        cfg = replace(Config(), backtest_spread_cost=0.15, backtest_use_recorded_spread=True)
        cost = CostModel.from_frame(pd.DataFrame({"spread": [0.9, 0.2, float("nan")]}), cfg)
        assert cost.spread_at(0) == pytest.approx(0.9)
        # 0.2 is narrower than the 0.30 floor -> the floor is charged instead
        assert cost.spread_at(1) == pytest.approx(0.30)
        # invalid values never become a free fill
        assert cost.spread_at(2) == pytest.approx(0.30)

    def test_out_of_range_index_falls_back_to_the_floor(self):
        cost = CostModel(variable_spread=np.array([0.8]), fixed_spread=0.3)
        assert cost.spread_at(5) == pytest.approx(0.3)
        assert cost.spread_at(-1) == pytest.approx(0.3)

    def test_stress_multiplier_scales_every_spread(self):
        cost = CostModel(
            variable_spread=np.array([0.5, 1.0]), fixed_spread=0.3, stress_multiplier=1.5
        )
        assert cost.spread_at(0) == pytest.approx(0.75)
        assert cost.spread_at(1) == pytest.approx(1.5)

    def test_round_trip_price_cost_adds_slippage_on_both_sides(self):
        cost = CostModel(variable_spread=None, fixed_spread=0.30, slippage_per_side=0.05)
        assert cost.round_trip_price_cost(0) == pytest.approx(0.40)

    def test_commission_is_charged_per_side(self):
        cost = CostModel(fixed_spread=0.0, commission_per_lot=3.5)
        assert cost.commission_currency(0, 0.4) == pytest.approx(2 * 3.5 * 0.4)
        # 1 lot, 1.0 price distance = 100 USD, so spread+commission add up
        money = cost.round_trip_currency_cost(0, 1.0, SPEC)
        assert money == pytest.approx(7.0)  # spread 0 + 2 x 3.5 per lot

    def test_describe_mentions_the_mode(self):
        variable = CostModel(variable_spread=np.array([0.4]), fixed_spread=0.3)
        assert "real per-bar spread" in variable.describe()
        assert "fixed" in CostModel(variable_spread=None, fixed_spread=0.3).describe()

    def test_spread_summary_stats(self):
        summary = spread_summary([0.1, 0.2, 0.3, 0.4, float("nan"), 0.0])
        assert summary["count"] == 4
        assert summary["mean"] == pytest.approx(0.25)
        assert summary["max"] == pytest.approx(0.4)
        assert spread_summary([])["count"] == 0


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def _m1_frame(n=12, start="2026-01-05 00:00:00"):
    times = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "open": [2000.0 + i for i in range(n)],
            "high": [2000.5 + i for i in range(n)],
            "low": [1999.5 + i for i in range(n)],
            "close": [2000.2 + i for i in range(n)],
        }
    )


class TestDataHelpers:
    def test_normalize_accepts_a_valid_frame_and_sorts_it(self):
        frame = _m1_frame(5).iloc[::-1].reset_index(drop=True)
        out = normalize_candles(frame)
        assert out["time"].is_monotonic_increasing
        assert out["open"].iloc[0] == pytest.approx(2000.0)
        assert out["time"].iloc[0] < out["time"].iloc[-1]

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            normalize_candles(pd.DataFrame({"time": [1], "open": [1], "close": [1]}))

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_candles(_m1_frame(3).iloc[0:0])

    def test_duplicate_timestamps_raise(self):
        frame = _m1_frame(4)
        doubled = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate timestamps"):
            normalize_candles(doubled)

    def test_impossible_ohlc_geometry_raises(self):
        frame = _m1_frame(4)
        frame.loc[2, "high"] = frame.loc[2, "close"] - 1.0
        with pytest.raises(ValueError, match="impossible OHLC"):
            normalize_candles(frame)

    def test_nan_ohlc_raises(self):
        frame = _m1_frame(4)
        frame.loc[1, "low"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            normalize_candles(frame)

    def test_build_m1_from_bid_ask_computes_the_real_spread(self):
        bid = _m1_frame(4)
        ask = bid.copy()
        ask["open"] += 0.30
        ask["high"] += 0.30
        ask["low"] += 0.30
        ask["close"] += 0.40
        merged = build_m1_from_bid_ask(bid, ask)
        assert merged["spread"].tolist() == pytest.approx([0.4, 0.4, 0.4, 0.4])
        # the chart price stays the bid side
        assert merged["close"].iloc[0] == pytest.approx(bid["close"].iloc[0])

    def test_build_m1_from_bid_ask_clamps_a_crossed_market_to_zero(self):
        bid = _m1_frame(3)
        ask = bid.copy()
        ask["close"] -= 0.5  # impossible, but must not become a negative cost
        merged = build_m1_from_bid_ask(bid, ask)
        assert (merged["spread"] >= 0).all()

    def test_build_m1_requires_overlapping_times(self):
        with pytest.raises(ValueError, match="share no timestamps"):
            build_m1_from_bid_ask(_m1_frame(3), _m1_frame(3, start="2027-01-01 00:00:00"))

    def test_resample_m1_to_m5_aggregates_ohlc_and_spread(self):
        frame = _m1_frame(12)
        frame["spread"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        bars = resample_candles(frame, "M5")
        assert len(bars) == 3
        first = bars.iloc[0]
        assert first["open"] == pytest.approx(2000.0)
        assert first["high"] == pytest.approx(2004.5)
        assert first["low"] == pytest.approx(1999.5)
        assert first["close"] == pytest.approx(2004.2)
        assert first["tick_volume"] == 5
        assert first["spread"] == pytest.approx(np.mean([0.1, 0.2, 0.3, 0.4, 0.5]))
        assert first["spread_max"] == pytest.approx(0.5)

    def test_resample_skips_empty_buckets(self):
        frame = pd.concat(
            [_m1_frame(5), _m1_frame(5, start="2026-01-05 00:20:00")], ignore_index=True
        )
        frame["spread"] = 0.2
        bars = resample_candles(frame, "M5")
        assert len(bars) == 2  # the 00:15 bucket had no minutes at all

    def test_resample_rejects_unknown_timeframe(self):
        with pytest.raises(ValueError, match="unsupported timeframe"):
            resample_candles(_m1_frame(6), "M7")

    def test_split_period_is_half_open(self):
        frame = build_df([(2000.0, 2001.0, 1999.0, 2000.5)] * 48, start="2026-01-05 00:00:00")
        out = split_period(frame, "2026-01-05", "2026-01-05 06:00:00")
        assert len(out) == 24  # 6h of M15 candles, end exclusive
        again = split_period(out, None, None)
        assert len(again) == 24

    def test_month_windows_are_contiguous(self):
        windows = month_windows("2021-01-01", "2021-04-01")
        assert len(windows) == 3
        assert windows[0][0].startswith("2021-01-01")
        assert windows[0][1] == windows[1][0]

    def test_spread_from_points_converts_with_the_symbol_point(self):
        frame = build_df([(2000.0, 2001.0, 1999.0, 2000.5)] * 5)
        frame = frame.assign(spread=[10, 20, 30, 40, 50])
        values = spread_from_points(frame, point=0.01)
        assert values.tolist() == pytest.approx([0.10, 0.20, 0.30, 0.40, 0.50])

    def test_spread_from_points_guards(self):
        frame = build_df([(2000.0, 2001.0, 1999.0, 2000.5)] * 5).drop(columns=["spread"])
        with pytest.raises(ValueError, match="no 'spread' column"):
            spread_from_points(frame, 0.01)
        frame = frame.assign(spread=[10] * len(frame))
        with pytest.raises(ValueError, match="point size"):
            spread_from_points(frame, 0.0)

    def test_load_candles_csv_reads_a_gzip_dump(self, tmp_path):
        frame = _m1_frame(6)
        frame["spread"] = 0.25
        path = tmp_path / "candles.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        loaded = load_candles_csv(str(path))
        assert len(loaded) == 6
        assert loaded["spread"].iloc[0] == pytest.approx(0.25)

    def test_assess_quality_reports_the_truth(self):
        frame = _m1_frame(10)
        frame["spread"] = [0.2, 0.4] * 5
        quality = assess_quality(frame)
        assert quality.rows == 10
        assert quality.years == (2026,)
        assert quality.mean_spread == pytest.approx(0.3)
        assert quality.p90_spread == pytest.approx(0.4)
        assert quality.duplicate_minutes == 0


# ---------------------------------------------------------------------------
# scalping backtest engine
# ---------------------------------------------------------------------------

def _scenario_frame(signal_at: int, *, move: float, bars: int = 40, freq: str = "5min", side: str = "buy"):
    """Scalp fixture followed by a monotonic run (so TP or SL must be hit).

    The synthetic frame carries no ``spread`` column: every engine test passes
    an explicit :class:`CostModel`, so a cost can never creep in silently from
    a fixture default.
    """
    base = make_scalp_frame(side=side)
    rows = [
        (
            float(base["open"].iloc[i]),
            float(base["high"].iloc[i]),
            float(base["low"].iloc[i]),
            float(base["close"].iloc[i]),
        )
        for i in range(len(base))
    ]
    price = rows[-1][3]
    for _ in range(bars):
        o = price
        c = o + move
        rows.append((o, max(o, c) + 0.05, min(o, c) - 0.05, c))
        price = c
    return build_df(rows, start="2026-01-05 00:00:00", freq=freq).drop(
        columns=["spread", "real_volume"]
    )


def _flat_cost(
    length: int, spread: float, commission: float = 0.0, stress: float = 1.0
) -> CostModel:
    """A constant, explicit spread (and optional commission) for one scenario."""
    return CostModel(
        variable_spread=np.full(length, spread),
        fixed_spread=spread,
        stress_multiplier=stress,
        commission_per_lot=commission,
    )


def _engine_cfg(**overrides):
    """Scenario config: cost gate set so a 1-ATR target is acceptable."""
    base = dict(
        scalp_tp_atr_multiple=1.0,
        scalp_sl_atr_multiple=1.0,
        min_profit_to_spread_ratio=1.0,
        scalp_spread_safety_margin=0.0,
        scalp_max_spread=5.0,
        scalp_vol_min_percentile=0.0,
        scalp_vol_max_percentile=1.0,
    )
    base.update(overrides)
    return scalp_cfg(**base)


class TestScalpingEngine:
    def test_too_few_bars_raise(self):
        with pytest.raises(ValueError, match="need at least"):
            ScalpingBacktestEngine(_engine_cfg()).run(
                build_df([(2000.0, 2001.0, 1999.0, 2000.5)] * 10).drop(columns=["spread"])
            )

    def test_entry_is_priced_at_the_ask_not_the_bar_open(self):
        """The spread is charged structurally: a buy fills at bid + spread."""
        df = _scenario_frame(74, move=0.05)
        result = ScalpingBacktestEngine(_engine_cfg(), cost=_flat_cost(len(df), 0.06)).run(
            df, spec=SPEC
        )
        assert result.trades, "fixture must produce at least one trade"
        signal_index = len(make_scalp_frame()) - 1
        bid_open = float(df["open"].iloc[signal_index + 1])
        assert result.trades[0].entry == pytest.approx(bid_open + 0.06)

    def test_a_wider_spread_moves_both_levels_and_the_fill(self):
        """Cost enters through the geometry, not through a subtracted number.

        With a strong synthetic run both variants still reach their target, so
        the P/L per trade is the same distance: what the wider spread takes is
        the *probability* of ever reaching it (the target sits further away from
        the bid, and the stop sits closer to it). That is exactly the trap a
        scalper falls into, and the model must not hide it.
        """
        df = _scenario_frame(74, move=0.5)
        cfg = _engine_cfg()
        clean = ScalpingBacktestEngine(cfg, cost=_flat_cost(len(df), 0.0)).run(df, spec=SPEC)
        wide = ScalpingBacktestEngine(cfg, cost=_flat_cost(len(df), 0.20)).run(df, spec=SPEC)
        assert clean.trades and wide.trades
        assert clean.trades[0].reason == "take_profit"
        assert clean.trades[0].pnl > 0
        assert wide.trades[0].entry - clean.trades[0].entry == pytest.approx(0.20)
        assert wide.trades[0].tp - clean.trades[0].tp == pytest.approx(0.20)
        assert wide.metrics.net_profit <= clean.metrics.net_profit
        assert wide.total_spread_cost > clean.total_spread_cost

    def test_stop_loss_is_evaluated_first_within_a_bar(self):
        base = make_scalp_frame()
        rows = [
            (
                float(base["open"].iloc[i]),
                float(base["high"].iloc[i]),
                float(base["low"].iloc[i]),
                float(base["close"].iloc[i]),
            )
            for i in range(len(base))
        ]
        entry = rows[-1][3]
        rows.append((entry, entry + 10.0, entry - 10.0, entry))  # touches both
        rows.append((entry, entry + 0.1, entry - 0.1, entry))     # keeps a bar in the loop
        df = build_df(rows).drop(columns=["spread", "real_volume"])
        result = ScalpingBacktestEngine(_engine_cfg(), cost=_flat_cost(len(df), 0.05)).run(
            df, spec=SPEC
        )
        assert result.metrics.total_trades == 1
        assert result.trades[0].reason == "stop_loss"

    def test_short_position_is_filled_at_the_bid(self):
        df = _scenario_frame(74, move=-0.5, side="sell")
        cfg = _engine_cfg()
        result = ScalpingBacktestEngine(cfg, cost=_flat_cost(len(df), 0.06)).run(df, spec=SPEC)
        assert result.trades, "a falling market after the SELL signal must trade"
        trade = result.trades[0]
        signal_index = len(make_scalp_frame(side="sell")) - 1
        bid_open = float(df["open"].iloc[signal_index + 1])
        assert trade.side == "SELL"
        assert trade.entry == pytest.approx(bid_open)  # never bid + spread
        assert trade.tp < trade.entry and trade.sl > trade.entry

    def test_short_target_needs_an_extra_spread_of_room_on_the_bid_path(self):
        """The ask (bid + spread) must reach the target, so the bid must overshoot."""
        df = _scenario_frame(74, move=-0.5, side="sell")
        tight = ScalpingBacktestEngine(
            _engine_cfg(), cost=_flat_cost(len(df), 0.0)
        ).run(df, spec=SPEC)
        wide = ScalpingBacktestEngine(
            _engine_cfg(), cost=_flat_cost(len(df), 0.25)
        ).run(df, spec=SPEC)
        if tight.trades and wide.trades:
            first = wide.records[0]
            assert first.trade.tp < first.trade.entry
            # a wide spread must never make the short exit richer
            assert wide.metrics.average_win <= tight.metrics.average_win + 1e-9

    def test_daily_cap_limits_entries_per_utc_day(self):
        df = _scenario_frame(74, move=0.5, bars=200)
        result = ScalpingBacktestEngine(
            _engine_cfg(scalp_max_trades_per_day=1), cost=_flat_cost(len(df), 0.05)
        ).run(df, spec=SPEC)
        assert result.trades
        days = pd.Series([t.entry_time.date() for t in result.trades])
        assert days.value_counts().max() <= 1

    def test_two_trades_a_day_are_allowed_when_the_cap_is_two(self):
        df = _scenario_frame(74, move=0.5, bars=200)
        result = ScalpingBacktestEngine(
            _engine_cfg(scalp_max_trades_per_day=2), cost=_flat_cost(len(df), 0.05)
        ).run(df, spec=SPEC)
        days = pd.Series([t.entry_time.date() for t in result.trades])
        assert days.value_counts().max() <= 2

    def test_exits_are_only_sl_tp_or_end_of_data(self):
        """No break-even, no partial close: the reason vocabulary is closed."""
        df = _scenario_frame(74, move=0.5, bars=60)
        result = ScalpingBacktestEngine(_engine_cfg(), cost=_flat_cost(len(df), 0.05)).run(
            df, spec=SPEC
        )
        assert result.trades
        for trade in result.trades:
            assert trade.reason in {"take_profit", "stop_loss", "end_of_data"}
            assert trade.volume > 0

    def test_spread_cost_accounting_matches_the_trades(self):
        df = _scenario_frame(74, move=0.5, bars=60)
        cost = _flat_cost(len(df), 0.12)
        result = ScalpingBacktestEngine(_engine_cfg(), cost=cost).run(df, spec=SPEC)
        expected = sum(
            record.spread_price * record.trade.volume * SPEC.contract_size
            for record in result.records
        )
        assert result.total_spread_cost == pytest.approx(expected, rel=1e-6)
        assert result.total_spread_cost > 0

    def test_commission_lands_on_the_pnl(self):
        df = _scenario_frame(74, move=0.5, bars=60)
        cost = _flat_cost(len(df), 0.05)
        plain = ScalpingBacktestEngine(_engine_cfg(), cost=cost).run(df, spec=SPEC)
        charged = ScalpingBacktestEngine(
            _engine_cfg(), cost=_flat_cost(len(df), 0.05, commission=5.0)
        ).run(df, spec=SPEC)
        assert plain.trades and charged.trades
        assert len(charged.trades) == len(plain.trades)
        for with_fee, without in zip(charged.trades, plain.trades):
            assert with_fee.pnl == pytest.approx(without.pnl - 2 * 5.0 * with_fee.volume, abs=1e-6)
        assert charged.total_commission_cost == pytest.approx(
            sum(2 * 5.0 * t.volume for t in charged.trades), abs=1e-6
        )

    def test_stress_raises_every_cost_by_the_multiplier(self):
        """A +50% stress scenario must widen the actual fill by 50%."""
        df = _scenario_frame(74, move=0.05)
        cfg = _engine_cfg()
        base = ScalpingBacktestEngine(cfg, cost=_flat_cost(len(df), 0.06)).run(df, spec=SPEC)
        stressed = ScalpingBacktestEngine(
            cfg, cost=_flat_cost(len(df), 0.06, stress=1.5)
        ).run(df, spec=SPEC)
        assert base.trades and stressed.trades
        assert stressed.trades[0].entry - base.trades[0].entry == pytest.approx(0.03)

    def test_a_stressed_run_trades_no_more_than_the_base_run(self):
        """The cost gate refuses tight scalps when the spread widens: less harm,
        never more. (Trading "through" a doubled spread is what the gate is for.)"""
        df = _scenario_frame(74, move=0.5, bars=60)
        cfg = _engine_cfg()
        base = ScalpingBacktestEngine(cfg, cost=_flat_cost(len(df), 0.30)).run(df, spec=SPEC)
        stressed = ScalpingBacktestEngine(
            cfg, cost=_flat_cost(len(df), 0.30, stress=1.5)
        ).run(df, spec=SPEC)
        assert stressed.metrics.total_trades <= base.metrics.total_trades

    def test_cost_gate_veto_is_counted_and_blocks_everything(self):
        df = _scenario_frame(74, move=0.5, bars=30)
        cfg = _engine_cfg(min_profit_to_spread_ratio=3.0, scalp_tp_atr_multiple=0.01)
        result = ScalpingBacktestEngine(cfg, cost=_flat_cost(len(df), 0.30)).run(df, spec=SPEC)
        assert result.veto_counts.get("cost_gate", 0) > 0
        assert result.metrics.total_trades == 0

    def test_engine_is_deterministic(self):
        df = _scenario_frame(74, move=0.5, bars=40)
        first = ScalpingBacktestEngine(_engine_cfg(), cost=_flat_cost(len(df), 0.05)).run(df, spec=SPEC)
        second = ScalpingBacktestEngine(_engine_cfg(), cost=_flat_cost(len(df), 0.05)).run(df, spec=SPEC)
        assert [t.pnl for t in first.trades] == [t.pnl for t in second.trades]
        assert first.metrics == second.metrics

    def test_equity_is_consistent_with_the_trades(self):
        df = _scenario_frame(74, move=0.5, bars=40)
        result = ScalpingBacktestEngine(_engine_cfg(), cost=_flat_cost(len(df), 0.05)).run(df, spec=SPEC)
        assert result.final_equity == pytest.approx(
            result.initial_balance + sum(t.pnl for t in result.trades)
        )
        assert result.metrics.net_profit == pytest.approx(sum(t.pnl for t in result.trades))

    def test_sizing_uses_the_separate_scalping_risk(self):
        df = _scenario_frame(74, move=0.5, bars=40)
        small = ScalpingBacktestEngine(
            _engine_cfg(scalping_risk_per_trade=0.0005), cost=_flat_cost(len(df), 0.05)
        ).run(df, spec=SPEC)
        large = ScalpingBacktestEngine(
            _engine_cfg(scalping_risk_per_trade=0.005), cost=_flat_cost(len(df), 0.05)
        ).run(df, spec=SPEC)
        assert large.trades and small.trades
        assert large.trades[0].volume > small.trades[0].volume

    def test_result_as_dict_exposes_the_report_fields(self):
        df = _scenario_frame(74, move=0.5, bars=20)
        result = ScalpingBacktestEngine(_engine_cfg(), cost=_flat_cost(len(df), 0.05)).run(df, spec=SPEC)
        data = result.as_dict()
        for key in (
            "total_trades",
            "profit_factor",
            "net_profit",
            "total_spread_cost",
            "veto_counts",
            "cost_model",
        ):
            assert key in data
        assert isinstance(result, ScalpingBacktestResult)

    def test_broker_minimum_distance_is_never_simulated_away(self):
        """A scalp tighter than stops_level must be refused, not widened."""
        df = _scenario_frame(74, move=0.5, bars=20)
        result = ScalpingBacktestEngine(
            _engine_cfg(), cost=_flat_cost(len(df), 0.05)
        ).run(df, spec=SPEC)
        minimum = SPEC.min_stop_distance()
        assert minimum == pytest.approx(0.20)
        for trade in result.trades:
            assert abs(trade.entry - trade.tp) >= minimum - 1e-9
            assert abs(trade.entry - trade.sl) >= minimum - 1e-9

    def test_config_is_validated_before_the_run(self):
        df = _scenario_frame(74, move=0.5, bars=10)
        with pytest.raises(ValueError):
            ScalpingBacktestEngine(make_cfg(min_profit_to_spread_ratio=0.2)).run(df, spec=SPEC)

    def test_default_cost_model_uses_the_conservative_floor(self):
        """Without a spread column the engine still charges 2 x backtest_spread_cost."""
        df = _scenario_frame(74, move=0.5, bars=20)
        result = ScalpingBacktestEngine(_engine_cfg(backtest_spread_cost=0.15)).run(df, spec=SPEC)
        signal_index = len(make_scalp_frame()) - 1
        if result.trades:
            bid_open = float(df["open"].iloc[signal_index + 1])
            assert result.trades[0].entry == pytest.approx(bid_open + 0.30)
        assert "fixed" in result.cost_model
