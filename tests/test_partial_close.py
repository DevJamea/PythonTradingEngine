"""Unit tests for partial close logic (stateless, broker-constraint aware)."""
from __future__ import annotations

import pytest

from gold_trader.models import PositionInfo
from gold_trader.trade_management.break_even import format_position_comment
from gold_trader.trade_management.partial_close import (
    manage_partial_close,
    planned_remaining_volume,
)
from tests._helpers import make_cfg

MAGIC = 77

LEVELS = ((1.0, 0.50), (2.0, 0.30), (3.0, 1.00))


def make_spec(**overrides):
    from gold_trader.models import SymbolSpec

    base = dict(
        name="XAUUSD", point=0.01, digits=2, volume_min=0.01, volume_max=10.0,
        volume_step=0.01, stops_level=20, freeze_level=10, visible=True,
        trade_mode=1, contract_size=100.0, trade_tick_size=0.01,
        trade_tick_value=1.0, filling_mode=3,
    )
    base.update(overrides)
    return SymbolSpec(**base)


def make_position(is_buy=True, entry=2000.0, initial_sl=1995.0, volume=0.5):
    return PositionInfo(
        ticket=1, symbol="XAUUSD", is_buy=is_buy, volume=volume,
        price_open=entry, sl=0.0, tp=0.0, profit=0.0, magic=MAGIC,
        comment=format_position_comment("GB", initial_sl, volume, 2),
    )


CFG = make_cfg(partial_close_enabled=True, partial_close_levels=LEVELS)
SPEC = make_spec()


def test_planned_remaining_at_each_level():
    # R = 5, buy entry 2000, initial volume 0.5
    assert planned_remaining_volume(2000.0, True, 0.5, 2004.0, LEVELS, 5.0) == pytest.approx(0.5)
    assert planned_remaining_volume(2000.0, True, 0.5, 2005.5, LEVELS, 5.0) == pytest.approx(0.25)
    assert planned_remaining_volume(2000.0, True, 0.5, 2011.0, LEVELS, 5.0) == pytest.approx(0.1)
    assert planned_remaining_volume(2000.0, True, 0.5, 2016.0, LEVELS, 5.0) == pytest.approx(0.0)


def test_planned_remaining_sell_side():
    assert planned_remaining_volume(2000.0, False, 0.5, 1994.5, LEVELS, 5.0) == pytest.approx(0.25)
    assert planned_remaining_volume(2000.0, False, 0.5, 2000.5, LEVELS, 5.0) == pytest.approx(0.5)


def test_first_level_closes_50_percent():
    position = make_position()
    actions = manage_partial_close([position], bid=2005.5, ask=2005.6, cfg=CFG, spec=SPEC)
    assert len(actions) == 1
    assert actions[0].kind == "partial_close"
    assert actions[0].close_volume == pytest.approx(0.25)


def test_catches_up_across_multiple_levels():
    position = make_position()
    actions = manage_partial_close([position], bid=2011.0, ask=2011.1, cfg=CFG, spec=SPEC)
    assert len(actions) == 1
    assert actions[0].close_volume == pytest.approx(0.4)  # 0.5 * (0.5 + 0.3)


def test_final_level_closes_remaining():
    position = make_position()
    actions = manage_partial_close([position], bid=2016.0, ask=2016.1, cfg=CFG, spec=SPEC)
    assert len(actions) == 1
    assert actions[0].kind == "full_close"
    assert actions[0].close_volume == pytest.approx(0.5)


def test_no_action_before_first_level():
    position = make_position()
    actions = manage_partial_close([position], bid=2004.0, ask=2004.1, cfg=CFG, spec=SPEC)
    assert actions == []


def test_partial_below_minimum_is_skipped_not_sent():
    """0.35 of 0.02 = 0.007 below the 0.01 minimum -> no action (and no
    final level configured, so nothing is force-closed)."""
    position = make_position(volume=0.02)
    cfg = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.35),))
    actions = manage_partial_close([position], bid=2005.5, ask=2005.6, cfg=cfg, spec=SPEC)
    assert actions == []


def test_final_level_closes_dust_remainder():
    """With a final (1.0) level, a dust remainder below the minimum is
    closed as a full close instead of being stranded."""
    position = make_position(volume=0.02)
    actions = manage_partial_close([position], bid=2016.0, ask=2016.1, cfg=CFG, spec=SPEC)
    assert len(actions) == 1
    assert actions[0].kind == "full_close"
    assert actions[0].close_volume == pytest.approx(0.02)


def test_sell_side_partial_close():
    position = make_position(is_buy=False, initial_sl=2005.0)
    actions = manage_partial_close([position], bid=1994.4, ask=1994.5, cfg=CFG, spec=SPEC)
    assert len(actions) == 1
    assert actions[0].close_volume == pytest.approx(0.25)


def test_disabled_by_config():
    position = make_position()
    cfg = make_cfg(partial_close_enabled=False)
    actions = manage_partial_close([position], bid=2016.0, ask=2016.1, cfg=cfg, spec=SPEC)
    assert actions == []


def test_manual_position_never_touched():
    position = PositionInfo(
        ticket=9, symbol="XAUUSD", is_buy=True, volume=0.5, price_open=2000.0,
        sl=1995.0, tp=0.0, profit=0.0, magic=1, comment="manual",
    )
    actions = manage_partial_close([position], bid=2016.0, ask=2016.1, cfg=CFG, spec=SPEC)
    assert actions == []
