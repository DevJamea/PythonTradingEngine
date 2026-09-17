"""Unit tests for validators."""
from __future__ import annotations

import pytest

from gold_trader.models import PositionInfo, SymbolSpec
from gold_trader.mt5.symbols import is_gold_symbol
from gold_trader.utils.validators import (
    ensure_finite_positive,
    round_price,
    validate_sl_modification,
    validate_sl_tp,
)

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


# ---------------------------------------------------------------------------
# BUG 2 & BUG 5 regression tests
# ---------------------------------------------------------------------------

def _make_spec(stops_level=20, freeze_level=10, digits=2, point=0.01):
    return SymbolSpec(
        name="XAUUSD",
        point=point,
        digits=digits,
        volume_min=0.01,
        volume_max=10.0,
        volume_step=0.01,
        stops_level=stops_level,
        freeze_level=freeze_level,
        visible=True,
        trade_mode=4,
        contract_size=100.0,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        filling_mode=3,
    )


def _make_pos(is_buy=True, entry=2000.0, sl=1995.0, tp=0.0):
    return PositionInfo(
        ticket=1,
        symbol="XAUUSD",
        is_buy=is_buy,
        volume=0.1,
        price_open=entry,
        sl=sl,
        tp=tp,
        profit=0.0,
        magic=77,
        comment="GB|sl=1995.00|vol=0.10",
        open_time=None,
    )


def test_valid_be_modification():
    spec = _make_spec()
    pos = _make_pos(is_buy=True, entry=2000.0, sl=1995.0)
    # Market at 2010.00 / 2010.20, new BE SL at 2000.10
    problems = validate_sl_modification(pos, 2000.10, bid=2010.00, ask=2010.20, spec=spec)
    assert problems == []


def test_sl_too_close_to_bid_for_buy():
    spec = _make_spec(stops_level=20)  # min distance = 0.20
    pos = _make_pos(is_buy=True, entry=2000.0, sl=1995.0)
    # Bid is 2000.00, new SL is 1999.90 (dist 0.10 < 0.20)
    problems = validate_sl_modification(pos, 1999.90, bid=2000.00, ask=2000.20, spec=spec)
    assert any("too close to Bid" in p for p in problems)


def test_sl_too_close_to_ask_for_sell():
    spec = _make_spec(stops_level=20)  # min distance = 0.20
    pos = _make_pos(is_buy=False, entry=2000.0, sl=2005.0)
    # Ask is 2000.00, new SL is 2000.10 (dist 0.10 < 0.20)
    problems = validate_sl_modification(pos, 2000.10, bid=1999.80, ask=2000.00, spec=spec)
    assert any("too close to Ask" in p for p in problems)


def test_freeze_level_violation_proposed_sl():
    spec = _make_spec(stops_level=10, freeze_level=50)  # freeze dist = 0.50
    pos = _make_pos(is_buy=True, entry=2000.0, sl=1995.0)
    # Bid is 2000.00, new SL is 1999.60 (dist 0.40 >= stops 0.10, but < freeze 0.50)
    problems = validate_sl_modification(pos, 1999.60, bid=2000.00, ask=2000.20, spec=spec)
    assert any("within freeze level of Bid" in p for p in problems)


def test_freeze_level_violation_frozen_order():
    spec = _make_spec(stops_level=10, freeze_level=50)  # freeze dist = 0.50
    # Existing SL is 1999.80, Bid is 2000.00 (distance 0.20 < freeze 0.50 -> order is frozen)
    pos = _make_pos(is_buy=True, entry=2000.0, sl=1999.80)
    problems = validate_sl_modification(pos, 2000.10, bid=2000.00, ask=2000.20, spec=spec)
    assert any("current stop loss is within freeze level" in p for p in problems)


def test_stops_level_violation():
    spec = _make_spec(stops_level=30)  # min distance = 0.30
    pos = _make_pos(is_buy=False, entry=2000.0, sl=2005.0)
    # Ask is 2000.00, new SL is 2000.15 (dist 0.15 < 0.30)
    problems = validate_sl_modification(pos, 2000.15, bid=1999.80, ask=2000.00, spec=spec)
    assert any("too close to Ask" in p for p in problems)


def test_sl_rounding_to_symbol_digits():
    spec = _make_spec(digits=2)
    pos = _make_pos(is_buy=True, entry=2000.0, sl=1995.0)
    # Unrounded 4 decimal places
    unrounded = 2000.1234
    problems = validate_sl_modification(pos, unrounded, bid=2010.00, ask=2010.20, spec=spec)
    assert any("exceeds symbol digits constraint" in p for p in problems)

    # Correctly rounded
    rounded = round_price(unrounded, spec.digits)
    assert rounded == 2000.12
    problems_ok = validate_sl_modification(pos, rounded, bid=2010.00, ask=2010.20, spec=spec)
    assert problems_ok == []


def test_direction_violation_buy():
    spec = _make_spec()
    pos = _make_pos(is_buy=True, entry=2000.0, sl=1995.0)
    # Bid is 2010.00, proposed SL is 2015.00 >= Bid
    problems = validate_sl_modification(pos, 2015.00, bid=2010.00, ask=2010.20, spec=spec)
    assert any("must be below Bid" in p for p in problems)


def test_direction_violation_sell():
    spec = _make_spec()
    pos = _make_pos(is_buy=False, entry=2000.0, sl=2005.0)
    # Ask is 2000.00, proposed SL is 1995.00 <= Ask
    problems = validate_sl_modification(pos, 1995.00, bid=1999.80, ask=2000.00, spec=spec)
    assert any("must be above Ask" in p for p in problems)


def test_gold_symbols_accepted():
    accepted = ["XAUUSD", "XAUUSDm", "XAUUSD.a", "GOLD", "GOLDm", "XAUUSD#", "XAUUSD_i", "XAUEUR", "GOLDUSD"]
    for sym in accepted:
        assert is_gold_symbol(sym) is True, f"Expected {sym} to be accepted as gold"


def test_gold_symbols_rejected():
    rejected = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "GOLDEN", "MARIGOLD", "GOLDFIELDS", "US30", ""]
    for sym in rejected:
        assert is_gold_symbol(sym) is False, f"Expected {sym} to be rejected as non-gold"

