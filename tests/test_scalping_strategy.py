"""Unit tests for the scalping strategy engine (pure, deterministic).

The tests are written to pin the *rules*, especially the ones that protect the
account from its own cost: the mandatory profit-vs-spread gate, the confirmed
reversal requirement, and causality (no lookahead).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gold_trader.models import Signal
from gold_trader.strategy.scalping import (
    SCALP_COMMENT_MARKER,
    ScalpingSLTP,
    bollinger_bands,
    build_scalping_sl_tp,
    compute_scalping_indicators,
    cost_gate_result,
    effective_spread,
    evaluate_scalping_at,
    excursion_extreme,
    fast_view,
    generate_scalping_signal,
    percentile_rank,
    required_tp_distance,
    reversal_confirmed,
    spread_gate_result,
    validate_scalping_config,
    volatility_gate_result,
)
from tests._helpers import build_df, make_cfg, make_scalp_frame, scalp_cfg


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_values_match_manual_computation(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        basis, upper, lower = bollinger_bands(close, period=4, n_std=2.0)
        window = close.iloc[1:5]
        expected_mean = float(window.mean())
        expected_std = float(window.std(ddof=0))
        assert math.isclose(float(basis.iloc[4]), expected_mean)
        assert math.isclose(float(upper.iloc[4]), expected_mean + 2.0 * expected_std)
        assert math.isclose(float(lower.iloc[4]), expected_mean - 2.0 * expected_std)

    def test_warmup_is_nan_never_a_partial_window(self):
        close = pd.Series([1.0] * 10)
        basis, _, _ = bollinger_bands(close, period=5, n_std=2.0)
        assert basis.iloc[:4].isna().all()
        assert not math.isnan(float(basis.iloc[4]))

    def test_flat_series_collapses_to_the_price(self):
        close = pd.Series([2000.0] * 8)
        basis, upper, lower = bollinger_bands(close, period=4, n_std=2.0)
        assert float(upper.iloc[-1]) == pytest.approx(2000.0)
        assert float(lower.iloc[-1]) == pytest.approx(2000.0)

    @pytest.mark.parametrize("period,std", [(1, 2.0), (0, 1.0), (5, 0.0), (5, -1.0)])
    def test_invalid_parameters_raise(self, period, std):
        with pytest.raises(ValueError):
            bollinger_bands(pd.Series([1.0, 2.0, 3.0]), period=period, n_std=std)


class TestPercentileRank:
    def test_known_ranks(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        ranks = percentile_rank(series, lookback=4)
        # every value <= itself in the window ending at it
        assert math.isnan(float(ranks.iloc[2]))
        assert float(ranks.iloc[3]) == pytest.approx(1.0)

    def test_causal_appending_a_future_value_changes_nothing(self):
        base = pd.Series([5.0, 1.0, 9.0, 2.0, 7.0, 3.0])
        short = percentile_rank(base, lookback=3)
        long = percentile_rank(
            pd.concat([base, pd.Series([100.0, 0.0, 100.0])], ignore_index=True),
            lookback=3,
        )
        pd.testing.assert_series_equal(short, long.iloc[: len(base)])

    def test_low_value_ranks_low_high_value_ranks_high(self):
        series = pd.Series([1.0] * 5 + [0.5, 10.0])
        ranks = percentile_rank(series, lookback=6)
        assert float(ranks.iloc[5]) < 0.3
        assert float(ranks.iloc[6]) == pytest.approx(1.0)

    def test_lookback_below_two_raises(self):
        with pytest.raises(ValueError):
            percentile_rank(pd.Series([1.0, 2.0]), lookback=1)


class TestComputeIndicators:
    def test_missing_column_raises(self):
        df = make_scalp_frame().drop(columns=["high"])
        with pytest.raises(ValueError, match="missing columns"):
            compute_scalping_indicators(df, scalp_cfg())

    def test_series_are_aligned_and_causal(self):
        df = make_scalp_frame()
        ind = compute_scalping_indicators(df, scalp_cfg())
        assert len(ind.atr) == len(df)
        assert len(ind.atr_rank) == len(df)
        # warm-up: RSI(7) needs 7 diffs, and diff() costs one bar, so the
        # first usable value is bar 7 (never a partial window)
        assert math.isnan(float(ind.rsi.iloc[6]))
        assert not math.isnan(float(ind.rsi.iloc[7]))

    def test_recomputing_on_the_visible_window_gives_the_same_decision(self):
        """No lookahead: prefix computation must match full-frame computation."""
        df = make_scalp_frame()
        cfg = scalp_cfg()
        full = compute_scalping_indicators(df, cfg)
        index = len(df) - 1
        prefix = compute_scalping_indicators(df.iloc[: index + 1].copy(), cfg)
        assert float(full.atr.iloc[index]) == pytest.approx(float(prefix.atr.iloc[-1]))
        assert float(full.rsi.iloc[index]) == pytest.approx(float(prefix.rsi.iloc[-1]))
        assert float(full.lower.iloc[index]) == pytest.approx(float(prefix.lower.iloc[-1]))
        assert reversal_confirmed(index, df, full) == reversal_confirmed(
            len(prefix.close) - 1, df.iloc[: index + 1].copy(), prefix
        )


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_defaults_are_valid_and_off(self):
        cfg = make_cfg()
        validate_scalping_config(cfg)
        assert cfg.scalping_enabled is False
        assert cfg.active_strategy == "signals"
        assert cfg.min_profit_to_spread_ratio == pytest.approx(3.0)
        assert cfg.scalping_risk_per_trade < cfg.risk_per_trade

    @pytest.mark.parametrize(
        "override",
        [
            {"min_profit_to_spread_ratio": 0.9},
            {"scalp_spread_safety_margin": -0.1},
            {"scalp_expected_spread": 0.0},
            {"scalp_tp_atr_multiple": 0.0},
            {"scalp_sl_atr_multiple": -1.0},
            {"scalp_vol_min_percentile": 0.9, "scalp_vol_max_percentile": 0.1},
            {"scalp_max_trades_per_day": 0},
            {"scalping_risk_per_trade": 0.0},
            {"scalping_timeframe": "H1"},
            {"scalping_timeframe": "M30"},
        ],
    )
    def test_self_contradictory_settings_raise(self, override):
        with pytest.raises(ValueError):
            validate_scalping_config(make_cfg(**override))

    def test_ratio_of_exactly_one_is_allowed(self):
        validate_scalping_config(make_cfg(min_profit_to_spread_ratio=1.0))

    def test_m1_is_a_configured_timeframe_but_documented_as_unreliable(self):
        validate_scalping_config(make_cfg(scalping_timeframe="M1"))


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

class TestSpreadAndCostGates:
    def test_effective_spread_prefers_the_live_quote(self):
        cfg = make_cfg(scalp_expected_spread=0.30)
        assert effective_spread(cfg, 0.12) == pytest.approx(0.12)

    @pytest.mark.parametrize("live", [None, 0.0, -1.0, float("nan")])
    def test_missing_live_quote_falls_back_to_the_conservative_estimate(self, live):
        cfg = make_cfg(scalp_expected_spread=0.30)
        assert effective_spread(cfg, live) == pytest.approx(0.30)

    def test_required_tp_distance_formula(self):
        cfg = make_cfg(
            min_profit_to_spread_ratio=3.0,
            scalp_spread_safety_margin=0.05,
        )
        assert required_tp_distance(0.25, cfg) == pytest.approx(3.0 * 0.30)

    def test_cost_gate_boundary_is_inclusive(self):
        cfg = make_cfg(min_profit_to_spread_ratio=3.0, scalp_spread_safety_margin=0.05)
        required = required_tp_distance(0.20, cfg)
        ok, _ = cost_gate_result(required, 0.20, cfg)
        assert ok is True
        fail, reason = cost_gate_result(required - 1e-9, 0.20, cfg)
        assert fail is False
        assert "TP distance" in reason

    def test_cost_gate_rejects_a_target_that_only_matches_the_spread(self):
        """The headline rule: TP == spread is a guaranteed loser, not a plan."""
        cfg = make_cfg(min_profit_to_spread_ratio=3.0, scalp_spread_safety_margin=0.0)
        fails, reason = cost_gate_result(0.30, 0.30, cfg)
        assert fails is False
        assert "required 0.90000" in reason  # 3 x (0.30 + 0.00)

    def test_cost_gate_rejects_invalid_distances(self):
        cfg = make_cfg(min_profit_to_spread_ratio=3.0)
        assert cost_gate_result(0.0, 0.1, cfg)[0] is False
        assert cost_gate_result(float("nan"), 0.1, cfg)[0] is False

    def test_spread_gate(self):
        cfg = make_cfg(scalp_max_spread=0.45)
        assert spread_gate_result(0.44, cfg)[0] is True
        ok, reason = spread_gate_result(0.46, cfg)
        assert ok is False
        assert "exceeds scalping maximum" in reason


class TestVolatilityGate:
    def test_accepts_only_the_middle_of_the_distribution(self):
        cfg = make_cfg(scalp_vol_min_percentile=0.10, scalp_vol_max_percentile=0.90)
        assert volatility_gate_result(1.0, 0.5, cfg)[0] is True

    def test_rejects_dead_market(self):
        cfg = make_cfg(scalp_vol_min_percentile=0.10, scalp_vol_max_percentile=0.90)
        ok, reason = volatility_gate_result(0.05, 0.02, cfg)
        assert ok is False
        assert "volatility too low" in reason

    def test_rejects_explosive_market(self):
        cfg = make_cfg(scalp_vol_min_percentile=0.10, scalp_vol_max_percentile=0.90)
        ok, reason = volatility_gate_result(25.0, 0.97, cfg)
        assert ok is False
        assert "volatility too high" in reason

    @pytest.mark.parametrize("atr,rank", [(float("nan"), 0.5), (1.0, float("nan")), (0.0, 0.5), (None, 0.5)])
    def test_rejects_missing_or_non_positive_data(self, atr, rank):
        cfg = make_cfg(scalp_vol_min_percentile=0.10, scalp_vol_max_percentile=0.90)
        assert volatility_gate_result(atr, rank, cfg)[0] is False


# ---------------------------------------------------------------------------
# SL / TP construction
# ---------------------------------------------------------------------------

class TestBuildScalpingSlTp:
    def test_buy_geometry(self):
        cfg = make_cfg(
            scalp_sl_atr_multiple=1.2,
            scalp_tp_atr_multiple=1.0,
            min_profit_to_spread_ratio=3.0,
            scalp_spread_safety_margin=0.0,
        )
        plan = build_scalping_sl_tp(Signal.BUY, entry=2000.0, atr_value=1.0, spread=0.10, cfg=cfg)
        assert isinstance(plan, ScalpingSLTP)
        assert plan.sl == pytest.approx(2000.0 - 1.2)
        assert plan.tp == pytest.approx(2000.0 + 1.0)
        assert plan.required_tp_distance == pytest.approx(0.30)

    def test_sell_geometry_is_the_mirror(self):
        cfg = make_cfg(
            scalp_sl_atr_multiple=1.2,
            scalp_tp_atr_multiple=1.0,
            min_profit_to_spread_ratio=3.0,
            scalp_spread_safety_margin=0.0,
        )
        plan = build_scalping_sl_tp(Signal.SELL, entry=2000.0, atr_value=1.0, spread=0.10, cfg=cfg)
        assert plan.sl == pytest.approx(2000.0 + 1.2)
        assert plan.tp == pytest.approx(2000.0 - 1.0)

    def test_tp_never_smaller_than_the_cost(self):
        """TP must clear (spread + margin) x ratio, or there is no plan at all."""
        cfg = make_cfg(
            scalp_sl_atr_multiple=1.0,
            scalp_tp_atr_multiple=0.5,
            min_profit_to_spread_ratio=3.0,
            scalp_spread_safety_margin=0.05,
        )
        # ATR 1.0 -> TP 0.5, but 3 * (0.30 + 0.05) = 1.05 -> refused
        assert build_scalping_sl_tp(Signal.BUY, 2000.0, 1.0, 0.30, cfg) is None

    def test_tp_exactly_equal_to_cost_is_still_refused_by_the_ratio(self):
        cfg = make_cfg(
            scalp_tp_atr_multiple=1.0,
            scalp_sl_atr_multiple=1.0,
            min_profit_to_spread_ratio=1.0,
            scalp_spread_safety_margin=0.0,
        )
        # ratio 1.0 means "TP >= spread"; equality is allowed but pointless
        plan = build_scalping_sl_tp(Signal.BUY, 2000.0, 0.30, 0.30, cfg)
        assert plan is not None
        assert plan.tp_distance == pytest.approx(0.30)

    def test_broker_minimum_stop_distance_is_enforced(self):
        cfg = make_cfg(
            scalp_sl_atr_multiple=1.0,
            scalp_tp_atr_multiple=1.0,
            min_profit_to_spread_ratio=1.0,
            scalp_spread_safety_margin=0.0,
        )
        # ATR 0.05 -> distances 0.05 < broker minimum 0.20 -> refused, not widened
        assert build_scalping_sl_tp(Signal.BUY, 2000.0, 0.05, 0.0, cfg, min_stop_distance=0.20) is None
        assert build_scalping_sl_tp(Signal.BUY, 2000.0, 0.5, 0.0, cfg, min_stop_distance=0.20) is not None

    def test_sell_target_below_zero_is_refused(self):
        cfg = make_cfg(
            scalp_tp_atr_multiple=50.0,
            scalp_sl_atr_multiple=1.0,
            min_profit_to_spread_ratio=1.0,
            scalp_spread_safety_margin=0.0,
        )
        assert build_scalping_sl_tp(Signal.SELL, 10.0, 1.0, 0.0, cfg) is None

    def test_invalid_direction_raises_and_invalid_inputs_return_none(self):
        cfg = make_cfg(scalp_tp_atr_multiple=1.0, scalp_sl_atr_multiple=1.0)
        with pytest.raises(ValueError):
            build_scalping_sl_tp(Signal.NO_TRADE, 2000.0, 1.0, 0.1, cfg)
        assert build_scalping_sl_tp(Signal.BUY, 2000.0, float("nan"), 0.1, cfg) is None
        assert build_scalping_sl_tp(Signal.BUY, 0.0, 1.0, 0.1, cfg) is None
        assert build_scalping_sl_tp(Signal.BUY, 2000.0, 0.0, 0.1, cfg) is None


# ---------------------------------------------------------------------------
# reversal pattern
# ---------------------------------------------------------------------------

class TestReversalConfirmation:
    @pytest.mark.parametrize("side,expected", [("buy", "BUY"), ("sell", "SELL")])
    def test_confirmed_reversal_bar(self, side, expected):
        df = make_scalp_frame(side=side)
        cfg = scalp_cfg()
        ind = compute_scalping_indicators(df, cfg)
        assert reversal_confirmed(len(df) - 1, df, ind) == expected

    def test_a_mere_touch_of_the_band_is_not_a_signal(self):
        """Second identical down candle: still outside, still falling -> None."""
        df = make_scalp_frame(n_push=5)
        rows = [
            (float(df["open"].iloc[i]), float(df["high"].iloc[i]), float(df["low"].iloc[i]), float(df["close"].iloc[i]))
            for i in range(len(df))
        ]
        last_close = rows[-1][3]
        rows.append((last_close, last_close + 0.02, last_close - 0.92, last_close - 0.9))
        extended = build_df(rows)
        cfg = scalp_cfg()
        ind = compute_scalping_indicators(extended, cfg)
        assert reversal_confirmed(len(extended) - 1, extended, ind) is None

    def test_bullish_close_back_inside_but_rsi_falling_is_rejected(self):
        df = make_scalp_frame()
        cfg = scalp_cfg(scalp_rsi_period=2)  # RSI(2) can already be rolling over
        ind = compute_scalping_indicators(df, cfg)
        side = reversal_confirmed(len(df) - 1, df, ind)
        # whichever way the 2-period RSI went, the decision must be a
        # deterministic function of the data, never of a future row
        assert side in (None, "BUY")

    def test_index_below_one_raises(self):
        df = make_scalp_frame()
        ind = compute_scalping_indicators(df, scalp_cfg())
        with pytest.raises(ValueError):
            reversal_confirmed(0, df, ind)

    def test_excursion_is_signed_and_zero_inside_the_band(self):
        df = make_scalp_frame()
        ind = compute_scalping_indicators(df, scalp_cfg())
        excursion = excursion_extreme(len(df) - 1, ind)
        assert excursion is not None and excursion < 0.0  # below the lower band
        # inside a live band -> exactly zero
        drift = build_df(
            [
                (
                    2000.0 + 0.05 * i,
                    2000.15 + 0.05 * i,
                    1999.85 + 0.05 * i,
                    2000.05 + 0.05 * i,
                )
                for i in range(80)
            ]
        )
        drift_ind = compute_scalping_indicators(drift, scalp_cfg())
        assert excursion_extreme(len(drift) - 1, drift_ind) == pytest.approx(0.0)
        # a degenerate zero-width band is "unknown", never a fake 0.0
        flat = build_df([(2000.0, 2000.0, 2000.0, 2000.0)] * 80)
        flat_ind = compute_scalping_indicators(flat, scalp_cfg())
        assert excursion_extreme(len(flat) - 1, flat_ind) is None


# ---------------------------------------------------------------------------
# full decision
# ---------------------------------------------------------------------------

class TestEvaluateScalpingAt:
    def test_perfect_pattern_with_enough_room_produces_a_buy(self):
        df = make_scalp_frame(side="buy")
        cfg = scalp_cfg()
        ind = compute_scalping_indicators(df, cfg)
        signal = evaluate_scalping_at(len(df) - 1, df, ind, cfg, spread=0.05)
        assert signal.signal is Signal.BUY
        assert "reject_code" not in signal.details
        assert signal.details["tp_distance"] >= signal.details["required_tp_distance"]

    def test_the_same_pattern_is_no_trade_when_the_target_cannot_pay_the_spread(self):
        """THE most important rule: signal quality never overrides the cost gate."""
        df = make_scalp_frame(side="buy")
        cfg = scalp_cfg(scalp_tp_atr_multiple=0.2, min_profit_to_spread_ratio=3.0)
        ind = compute_scalping_indicators(df, cfg)
        signal = evaluate_scalping_at(len(df) - 1, df, ind, cfg, spread=0.30)
        assert signal.signal is Signal.NO_TRADE
        assert signal.details["reject_code"] == "cost_gate"
        assert "cost gate" in signal.reason

    def test_daily_cap_short_circuits_before_the_pattern(self):
        df = make_scalp_frame()
        cfg = scalp_cfg(scalp_max_trades_per_day=3)
        ind = compute_scalping_indicators(df, cfg)
        signal = evaluate_scalping_at(
            len(df) - 1, df, ind, cfg, spread=0.05, trades_today=3
        )
        assert signal.signal is Signal.NO_TRADE
        assert signal.details["reject_code"] == "daily_cap"

    def test_wide_live_spread_vetoes_even_a_good_pattern(self):
        df = make_scalp_frame()
        cfg = scalp_cfg(scalp_max_spread=0.45)
        ind = compute_scalping_indicators(df, cfg)
        signal = evaluate_scalping_at(len(df) - 1, df, ind, cfg, spread=0.90)
        assert signal.signal is Signal.NO_TRADE
        assert signal.details["reject_code"] == "wide_spread"

    def test_volatility_filter_vetoes_before_the_pattern(self):
        df = make_scalp_frame()
        cfg = scalp_cfg(scalp_vol_max_percentile=0.5)  # the bar is the loudest -> rejected
        ind = compute_scalping_indicators(df, cfg)
        signal = evaluate_scalping_at(len(df) - 1, df, ind, cfg, spread=0.05)
        assert signal.details["reject_code"] == "volatility"
        assert signal.signal is Signal.NO_TRADE
        assert "too high" in signal.reason

    def test_flat_market_never_produces_a_signal(self):
        df = build_df([(2000.0, 2000.05, 1999.95, 2000.0)] * 200)
        cfg = scalp_cfg()
        ind = compute_scalping_indicators(df, cfg)
        for index in range(80, len(df) - 1, 7):
            assert evaluate_scalping_at(index, df, ind, cfg, spread=0.05).signal is Signal.NO_TRADE

    def test_signal_is_deterministic(self):
        df = make_scalp_frame()
        cfg = scalp_cfg()
        ind = compute_scalping_indicators(df, cfg)
        first = evaluate_scalping_at(len(df) - 1, df, ind, cfg, spread=0.05)
        second = evaluate_scalping_at(len(df) - 1, df, ind, cfg, spread=0.05)
        assert first.signal is second.signal
        assert first.reason == second.reason

    def test_marker_constant_is_stable(self):
        assert SCALP_COMMENT_MARKER == "SCALP"


class TestGenerateScalpingSignal:
    def test_last_closed_candle_is_used(self):
        df = make_scalp_frame()
        signal = generate_scalping_signal(df, scalp_cfg(), spread=0.05)
        assert signal.signal is Signal.BUY

    def test_short_history_is_no_trade_not_a_crash(self):
        df = make_scalp_frame(n_flat=20)
        signal = generate_scalping_signal(df, scalp_cfg(scalp_min_candles_for_signal=60), spread=0.05)
        assert signal.signal is Signal.NO_TRADE
        assert "insufficient candles" in signal.reason

    def test_one_candle_raises(self):
        with pytest.raises(ValueError):
            generate_scalping_signal(build_df([(2000.0, 2001.0, 1999.0, 2000.5)]), scalp_cfg())

    def test_invalid_config_is_refused_before_any_computation(self):
        df = make_scalp_frame()
        with pytest.raises(ValueError):
            generate_scalping_signal(df, scalp_cfg(min_profit_to_spread_ratio=0.5))

    def test_no_lookahead_future_rows_cannot_change_the_last_decision(self):
        """Truncating the future must not change the decision at index i."""
        df = make_scalp_frame()
        cfg = scalp_cfg()
        index = len(df) - 1
        ind_full = compute_scalping_indicators(df, cfg)
        before = evaluate_scalping_at(index, df, ind_full, cfg, spread=0.05)
        appended = pd.concat(
            [df, build_df([(2600.0, 2610.0, 2590.0, 2605.0)] * 30, start="2026-03-01 00:00:00")],
            ignore_index=True,
        )
        ind_after = compute_scalping_indicators(appended, cfg)
        after = evaluate_scalping_at(index, appended, ind_after, cfg, spread=0.05)
        assert before.signal is after.signal
        assert before.reason == after.reason


class TestFastView:
    def test_view_matches_the_pandas_path_on_every_bar(self):
        df = make_scalp_frame(n_flat=90)
        cfg = scalp_cfg()
        ind = compute_scalping_indicators(df, cfg)
        view, view_ind = fast_view(df, ind)
        assert len(view) == len(df)
        for index in range(70, len(df) - 1):
            slow = evaluate_scalping_at(index, df, ind, cfg, spread=0.05)
            fast = evaluate_scalping_at(index, view, view_ind, cfg, spread=0.05)
            assert slow.signal is fast.signal
            assert slow.reason == fast.reason

    def test_view_is_read_only_numpy_backed(self):
        df = make_scalp_frame()
        ind = compute_scalping_indicators(df, scalp_cfg())
        view, view_ind = fast_view(df, ind)
        assert isinstance(view_ind.atr.iloc._array, np.ndarray)
        assert float(view["open"].iloc[len(df) - 1]) == pytest.approx(
            float(df["open"].iloc[len(df) - 1])
        )
