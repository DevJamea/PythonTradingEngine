"""Unit tests for candlestick pattern detection."""
from __future__ import annotations

import pytest

from gold_trader.strategy.candles import Candle, detect_patterns_pair, latest_patterns
from tests._helpers import build_df


def c(open_, high, low, close) -> Candle:
    return Candle(open=open_, high=high, low=low, close=close)


class TestBullishBearish:
    def test_bullish(self):
        patterns = detect_patterns_pair(c(100, 102, 99, 101), c(101, 105, 100.5, 104))
        assert patterns["bullish_candle"].detected
        assert 0.0 < patterns["bullish_candle"].strength <= 1.0
        assert not patterns["bearish_candle"].detected

    def test_bearish(self):
        patterns = detect_patterns_pair(c(100, 102, 99, 101), c(104, 104.5, 100.5, 101))
        assert patterns["bearish_candle"].detected
        assert 0.0 < patterns["bearish_candle"].strength <= 1.0
        assert not patterns["bullish_candle"].detected

    def test_zero_range_is_neutral(self):
        patterns = detect_patterns_pair(c(100, 100, 100, 100), c(100, 100, 100, 100))
        assert not patterns["bullish_candle"].detected
        assert not patterns["bearish_candle"].detected
        assert not patterns["doji"].detected
        assert not patterns["hammer"].detected
        assert not patterns["shooting_star"].detected


class TestDoji:
    def test_doji_detected(self):
        prev = c(100, 102, 99, 101)
        cur = c(101, 101.3, 100.7, 101.05)  # body 0.05 <= 0.1 * 0.6
        pattern = detect_patterns_pair(prev, cur)["doji"]
        assert pattern.detected
        assert pattern.strength == pytest.approx(1.0 - 0.05 / 0.6)

    def test_not_doji_with_real_body(self):
        pattern = detect_patterns_pair(
            c(100, 102, 99, 101), c(101, 101.9, 100.7, 101.8)
        )["doji"]
        assert not pattern.detected
        assert pattern.strength == 0.0


class TestEngulfing:
    def test_bullish_engulfing(self):
        prev = c(102, 102.5, 99.5, 100)  # bearish, body 2
        cur = c(99.8, 103.2, 99.4, 102.3)  # bullish, body 2.5
        pattern = detect_patterns_pair(prev, cur)["bullish_engulfing"]
        assert pattern.detected
        assert pattern.strength == pytest.approx(2.5 / 4.5)
        assert 0.5 < pattern.strength < 1.0

    def test_equal_body_is_not_engulfing(self):
        prev = c(102, 102.5, 100, 100)  # bearish, body 2
        cur = c(100, 102, 99.9, 102)  # bullish, body 2 (not strictly larger)
        pattern = detect_patterns_pair(prev, cur)["bullish_engulfing"]
        assert not pattern.detected

    def test_bearish_engulfing(self):
        prev = c(100, 102.5, 99.8, 102)  # bullish, body 2
        cur = c(102.2, 102.6, 99.6, 99.8)  # bearish, body 2.4
        pattern = detect_patterns_pair(prev, cur)["bearish_engulfing"]
        assert pattern.detected
        assert pattern.strength == pytest.approx(2.4 / 4.4)

    def test_prev_must_be_bearish_for_bullish_engulfing(self):
        prev = c(100, 102.5, 99.8, 102)  # bullish
        cur = c(99.8, 103.2, 99.4, 102.3)
        pattern = detect_patterns_pair(prev, cur)["bullish_engulfing"]
        assert not pattern.detected


class TestHammerShootingStar:
    def test_hammer(self):
        prev = c(100, 102, 99, 101)
        cur = c(101.0, 101.2, 98.5, 101.1)  # lower wick 2.5 >= 2*body 0.1
        pattern = detect_patterns_pair(prev, cur)["hammer"]
        assert pattern.detected
        assert pattern.strength == pytest.approx(2.5 / 2.7)

    def test_not_hammer_long_upper_wick(self):
        cur = c(101.0, 101.5, 100.5, 101.1)
        pattern = detect_patterns_pair(c(100, 102, 99, 101), cur)["hammer"]
        assert not pattern.detected

    def test_shooting_star(self):
        prev = c(100, 102, 99, 101)
        cur = c(100.9, 103.5, 100.8, 101.0)  # upper wick 2.5 >= 2*body 0.1
        pattern = detect_patterns_pair(prev, cur)["shooting_star"]
        assert pattern.detected
        assert pattern.strength == pytest.approx(2.5 / 2.7)

    def test_not_shooting_star_long_lower_wick(self):
        cur = c(101.0, 101.5, 100.5, 101.1)
        pattern = detect_patterns_pair(c(100, 102, 99, 101), cur)["shooting_star"]
        assert not pattern.detected


def test_latest_patterns_uses_last_two_candles():
    rows = [
        (100, 102, 99, 101),
        (102, 102.5, 99.5, 100),  # bearish
        (99.8, 103.2, 99.4, 102.3),  # bullish engulfing
    ]
    patterns = latest_patterns(build_df(rows))
    assert patterns["bullish_engulfing"].detected


def test_latest_patterns_requires_two_candles():
    with pytest.raises(ValueError):
        latest_patterns(build_df([(100, 102, 99, 101)]))
