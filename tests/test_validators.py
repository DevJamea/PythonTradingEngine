"""Unit tests for validators."""
from __future__ import annotations

import pytest

from gold_trader.utils.validators import ensure_finite_positive, round_price, validate_sl_tp

POINT = 0.01
STOPS_LEVEL = 20  # min distance 0.20


def test_valid_buy_sl_tp():
    problems = validate_sl_tp(2000.0, 1995.0, 2010.0, True, POINT, STOPS_LEVEL)
    assert problems == []


def test_valid_sell_sl_tp():
    problems = validate_sl_tp(2000.0, 2005.0, 1990.0, False, POINT, STOPS_LEVEL)
    assert problems == []


def test_buy_sl_above_entry():
    problems = validate_sl_tp(2000.0, 2005.0, 2010.0, True, POINT, STOPS_LEVEL)
    assert any("below entry" in p for p in problems)


def test_sell_tp_above_entry():
    problems = validate_sl_tp(2000.0, 2005.0, 2010.0, False, POINT, STOPS_LEVEL)
    assert any("below entry" in p for p in problems)


def test_stop_too_close():
    problems = validate_sl_tp(2000.0, 1999.9, 2010.0, True, POINT, STOPS_LEVEL)
    assert any("stop loss too close" in p for p in problems)


def test_tp_too_close():
    problems = validate_sl_tp(2000.0, 1995.0, 2000.1, True, POINT, STOPS_LEVEL)
    assert any("take profit too close" in p for p in problems)


def test_non_positive_prices():
    problems = validate_sl_tp(2000.0, 0.0, 2010.0, True, POINT, STOPS_LEVEL)
    assert any("invalid stop loss" in p for p in problems)


def test_non_numeric_input():
    problems = validate_sl_tp("a", 1995.0, 2010.0, True, POINT, STOPS_LEVEL)
    assert problems == ["entry/SL/TP must be numbers"]


def test_ensure_finite_positive():
    assert ensure_finite_positive("x", 5) == 5.0
    with pytest.raises(ValueError):
        ensure_finite_positive("x", -1)
    with pytest.raises(ValueError):
        ensure_finite_positive("x", float("nan"))
    with pytest.raises(ValueError):
        ensure_finite_positive("x", "abc")


def test_round_price():
    assert round_price(2385.404999, 2) == 2385.4
    assert round_price(2385.405, 2) == 2385.41
