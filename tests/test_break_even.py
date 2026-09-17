"""Unit tests for break-even management (idempotent, comment-state based)."""
from __future__ import annotations

import pytest

from gold_trader.models import PositionInfo
from gold_trader.trade_management.break_even import (
    breakeven_target,
    format_position_comment,
    manage_break_even,
    parse_position_comment,
)
from tests._helpers import make_cfg

MAGIC = 77


def make_spec():
    from gold_trader.models import SymbolSpec

    return SymbolSpec(
        name="XAUUSD", point=0.01, digits=2, volume_min=0.01, volume_max=10.0,
        volume_step=0.01, stops_level=20, freeze_level=10, visible=True,
        trade_mode=1, contract_size=100.0, trade_tick_size=0.01,
        trade_tick_value=1.0, filling_mode=3,
    )


def make_position(is_buy=True, entry=2000.0, initial_sl=1995.0, volume=0.5, sl=1995.0, comment=None):
    if comment is None:
        comment = format_position_comment("GB", initial_sl, volume, 2)
    return PositionInfo(
        ticket=1, symbol="XAUUSD", is_buy=is_buy, volume=volume,
        price_open=entry, sl=sl, tp=0.0, profit=0.0, magic=MAGIC, comment=comment,
    )


CFG = make_cfg(break_even_enabled=True, break_even_r=1.0, break_even_buffer=0.1)
SPEC = make_spec()


def test_comment_roundtrip():
    comment = format_position_comment("GB", 1995.25, 0.3, 2)
    assert parse_position_comment(comment) == {
        "initial_sl": 1995.25,
        "initial_volume": 0.3,
    }
    assert parse_position_comment("manual trade") is None
    assert parse_position_comment("") is None


def test_breakeven_target():
    assert breakeven_target(make_position(is_buy=True), 0.1) == pytest.approx(2000.1)
    assert breakeven_target(make_position(is_buy=False, initial_sl=2005.0), 0.1) == pytest.approx(1999.9)


def test_buy_moves_to_break_even_at_1R():
    # R = 5.0, price at 2005.5 >= entry + 1R = 2005.0
    position = make_position(is_buy=True)
    actions = manage_break_even([position], bid=2005.5, ask=2005.6, cfg=CFG, spec=SPEC)
    assert len(actions) == 1
    assert actions[0].kind == "move_sl"
    assert actions[0].new_sl == pytest.approx(2000.1)


def test_buy_no_move_before_1R():
    position = make_position(is_buy=True)
    actions = manage_break_even([position], bid=2004.0, ask=2004.1, cfg=CFG, spec=SPEC)
    assert actions == []


def test_buy_idempotent_when_already_moved():
    position = make_position(is_buy=True, sl=2000.1)
    actions = manage_break_even([position], bid=2005.5, ask=2005.6, cfg=CFG, spec=SPEC)
    assert actions == []


def test_sell_moves_to_break_even_at_1R():
    position = make_position(is_buy=False, initial_sl=2005.0, sl=2005.0)
    actions = manage_break_even([position], bid=1994.4, ask=1994.5, cfg=CFG, spec=SPEC)
    assert len(actions) == 1
    assert actions[0].new_sl == pytest.approx(1999.9)


def test_sell_no_move_before_1R():
    position = make_position(is_buy=False, initial_sl=2005.0)
    actions = manage_break_even([position], bid=1995.6, ask=1995.7, cfg=CFG, spec=SPEC)
    assert actions == []


def test_manual_position_is_never_touched():
    position = make_position(is_buy=True, comment="some manual comment")
    actions = manage_break_even([position], bid=2005.5, ask=2005.6, cfg=CFG, spec=SPEC)
    assert actions == []


def test_disabled_by_config():
    position = make_position(is_buy=True)
    cfg = make_cfg(break_even_enabled=False)
    actions = manage_break_even([position], bid=2005.5, ask=2005.6, cfg=cfg, spec=SPEC)
    assert actions == []


def test_custom_r_multiple():
    position = make_position(is_buy=True)  # R = 5.0
    cfg = make_cfg(break_even_r=2.0, break_even_buffer=0.1)
    # at +1.5R nothing happens, at +2R it does
    assert manage_break_even([position], bid=2007.5, ask=2007.6, cfg=cfg, spec=SPEC) == []
    assert len(manage_break_even([position], bid=2010.0, ask=2010.1, cfg=cfg, spec=SPEC)) == 1


def test_no_current_sl_gets_one():
    position = make_position(is_buy=True, sl=0.0)
    actions = manage_break_even([position], bid=2005.5, ask=2005.6, cfg=CFG, spec=SPEC)
    assert len(actions) == 1
    assert actions[0].new_sl == pytest.approx(2000.1)
