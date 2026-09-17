"""Unit tests for EMA / RSI / ATR (deterministic, no look-ahead)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gold_trader.strategy.indicators import atr, ema, latest_valid, rsi


class TestEMA:
    def test_matches_manual_recursion(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 5.0]
        series = pd.Series(values)
        period = 3
        alpha = 2.0 / (period + 1)
        expected = [values[0]]
        for value in values[1:]:
            expected.append(alpha * value + (1.0 - alpha) * expected[-1])
        result = ema(series, period)
        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        for i in range(2, len(values)):
            assert result.iloc[i] == pytest.approx(expected[i], rel=1e-12)

    def test_constant_series(self):
        result = ema(pd.Series([5.0] * 10), 5)
        for i in range(4, 10):
            assert result.iloc[i] == pytest.approx(5.0)

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            ema(pd.Series([1.0, 2.0]), 0)


class TestRSI:
    def test_all_gains_gives_100(self):
        series = pd.Series([float(x) for x in range(1, 40)])
        assert rsi(series, 14).iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_all_losses_gives_0(self):
        series = pd.Series([float(x) for x in range(40, 1, -1)])
        assert rsi(series, 14).iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_flat_series_is_neutral(self):
        result = rsi(pd.Series([100.0] * 30), 14)
        assert result.iloc[-1] == pytest.approx(50.0)

    def test_bounded_between_0_and_100(self):
        rng = np.random.default_rng(42)
        series = pd.Series(100.0 + rng.normal(0.0, 1.0, 300).cumsum())
        result = rsi(series, 14).dropna()
        assert (result >= 0.0).all()
        assert (result <= 100.0).all()

    def test_initial_values_are_nan(self):
        result = rsi(pd.Series([float(x) for x in range(1, 30)]), 14)
        assert result.iloc[:14].isna().all()

    def test_no_lookahead(self):
        rng = np.random.default_rng(7)
        values = 2000.0 + rng.normal(0.0, 1.0, 120).cumsum()
        full = rsi(pd.Series(values), 14)
        partial = rsi(pd.Series(values[:80]), 14)
        pd.testing.assert_series_equal(
            full.iloc[:80].reset_index(drop=True),
            partial.reset_index(drop=True),
            check_names=False,
        )


class TestATR:
    def test_constant_range(self):
        n = 40
        close = pd.Series([100.0] * n)
        high = close + 0.5
        low = close - 0.5
        result = atr(high, low, close, 14)
        for i in range(13, n):
            assert result.iloc[i] == pytest.approx(1.0)

    def test_nan_before_min_periods(self):
        n = 40
        close = pd.Series([100.0] * n)
        result = atr(close + 0.5, close - 0.5, close, 14)
        assert result.iloc[:13].isna().all()

    def test_no_lookahead(self):
        rng = np.random.default_rng(11)
        values = 2000.0 + rng.normal(0.0, 1.0, 120).cumsum()
        close = pd.Series(values)
        high = close + 0.5
        low = close - 0.5
        full = atr(high, low, close, 14)
        partial = atr(high[:80], low[:80], close[:80], 14)
        pd.testing.assert_series_equal(
            full.iloc[:80].reset_index(drop=True),
            partial.reset_index(drop=True),
            check_names=False,
        )


def test_latest_valid():
    series = pd.Series([np.nan, np.nan, 1.5, np.nan, 2.5])
    assert latest_valid(series) == 2.5
    assert latest_valid(pd.Series([np.nan, np.nan])) is None
