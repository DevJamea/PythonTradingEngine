"""Unit tests for the trend engine rules."""
from __future__ import annotations

from gold_trader.models import Trend
from gold_trader.strategy.trend import classify_trend_values


def test_uptrend():
    result = classify_trend_values(2010, 2005, 2000, 1990, require_close=True)
    assert result.trend is Trend.UPTREND


def test_uptrend_requires_close_above_fast_ema():
    result = classify_trend_values(2003, 2005, 2000, 1990, require_close=True)
    assert result.trend is Trend.SIDEWAYS


def test_uptrend_without_close_requirement():
    result = classify_trend_values(2003, 2005, 2000, 1990, require_close=False)
    assert result.trend is Trend.UPTREND


def test_downtrend():
    result = classify_trend_values(1990, 1995, 2000, 2010, require_close=True)
    assert result.trend is Trend.DOWNTREND


def test_downtrend_requires_close_below_fast_ema():
    result = classify_trend_values(1997, 1995, 2000, 2010, require_close=True)
    assert result.trend is Trend.SIDEWAYS


def test_sideways_when_emas_not_aligned():
    result = classify_trend_values(2005, 2005, 1998, 2002, require_close=True)
    assert result.trend is Trend.SIDEWAYS


def test_sideways_on_missing_values():
    result = classify_trend_values(2005, 2005, None, 2002, require_close=True)
    assert result.trend is Trend.SIDEWAYS
    assert "NaN" in result.reason
