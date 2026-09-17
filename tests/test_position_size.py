"""Unit tests for position sizing and volume rounding."""
from __future__ import annotations

import pytest

from gold_trader.models import SymbolSpec
from gold_trader.risk.position_size import (
    calculate_position_size,
    profit_for_volume,
    round_volume_to_step,
)


def make_spec(**overrides) -> SymbolSpec:
    """Gold-like spec: tick 0.01 = 1.0 currency => $100 per $1 move per lot."""
    base = dict(
        name="XAUUSD",
        point=0.01,
        digits=2,
        volume_min=0.01,
        volume_max=10.0,
        volume_step=0.01,
        stops_level=20,
        freeze_level=10,
        visible=True,
        trade_mode=1,
        contract_size=100.0,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        filling_mode=3,
    )
    base.update(overrides)
    return SymbolSpec(**base)


SPEC = make_spec()


def test_basic_sizing():
    # risk $50, stop distance $5 -> loss per lot $500 -> 0.1 lot
    result = calculate_position_size(10_000, 0.005, 2000.0, 1995.0, SPEC)
    assert result.volume == pytest.approx(0.1)
    assert result.risk_amount == pytest.approx(50.0)
    assert result.reason == "ok"


def test_rounds_down_to_step():
    # distance $1.953 -> per lot $195.3 -> raw 0.256 -> 0.25
    result = calculate_position_size(10_000, 0.005, 2000.0, 1998.047, SPEC)
    assert result.volume == pytest.approx(0.25)
    assert result.raw_volume > 0.25


def test_below_broker_minimum_is_no_trade():
    # distance $500 -> per lot $50,000 -> raw 0.001 < 0.01 minimum
    result = calculate_position_size(10_000, 0.005, 2000.0, 1500.0, SPEC)
    assert result.volume is None
    assert "minimum" in result.reason


def test_above_broker_max_is_clamped():
    # raw volume 100 > max 10 -> clamp (actual risk below target, safe side)
    result = calculate_position_size(1_000_000, 0.01, 2000.0, 1999.0, SPEC)
    assert result.volume == pytest.approx(10.0)
    assert "clamped" in result.reason


def test_stop_equal_to_entry_is_no_trade():
    result = calculate_position_size(10_000, 0.005, 2000.0, 2000.0, SPEC)
    assert result.volume is None
    assert "equals" in result.reason


def test_invalid_inputs_are_no_trade():
    assert calculate_position_size(-1, 0.005, 2000.0, 1995.0, SPEC).volume is None
    assert calculate_position_size(10_000, 0.0, 2000.0, 1995.0, SPEC).volume is None
    assert calculate_position_size(10_000, 0.005, -2000.0, 1995.0, SPEC).volume is None


def test_fallback_to_contract_size_without_tick_value():
    spec = make_spec(trade_tick_size=0.0, trade_tick_value=0.0)
    # per lot = 5 * 100 = 500 -> same 0.1 lot
    result = calculate_position_size(10_000, 0.005, 2000.0, 1995.0, spec)
    assert result.volume == pytest.approx(0.1)


def test_no_loss_computation_possible_is_no_trade():
    spec = make_spec(trade_tick_size=0.0, trade_tick_value=0.0, contract_size=0.0)
    result = calculate_position_size(10_000, 0.005, 2000.0, 1995.0, spec)
    assert result.volume is None
    assert "loss per lot" in result.reason


def test_zero_volume_step_is_no_trade():
    spec = make_spec(volume_step=0.0)
    result = calculate_position_size(10_000, 0.005, 2000.0, 1995.0, spec)
    assert result.volume is None


def test_round_volume_to_step_down():
    assert round_volume_to_step(0.129, 0.01) == pytest.approx(0.12)
    assert round_volume_to_step(0.1, 0.01) == pytest.approx(0.1)
    assert round_volume_to_step(0.30000000000000004, 0.01) == pytest.approx(0.3)


def test_round_volume_invalid_step():
    with pytest.raises(ValueError):
        round_volume_to_step(0.1, 0.0)


def test_profit_for_volume():
    # $2.5 move, 0.2 lot -> 250 per lot * 0.2 = $50
    assert profit_for_volume(2.5, 0.2, SPEC) == pytest.approx(50.0)


def test_profit_for_volume_rejects_negative_distance():
    with pytest.raises(ValueError):
        profit_for_volume(-1.0, 0.1, SPEC)
