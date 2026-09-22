"""Micro-hardening regression tests for Final Audit V2 findings.

Covers:
- FIX 1: MarketState permission defaults must be fail-safe (False)
- FIX 2: close_position() must fail explicitly when Bid/Ask unavailable
- FIX 3: partial close remaining volume must stay broker-valid
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import gold_trader.mt5.connection as mt5_conn
from gold_trader.models import MarketState, PositionInfo, SymbolSpec
from gold_trader.mt5.execution_gate import (
    ExecutionPermission,
    execution_permission,
    refresh_verified_account_safety,
)
from gold_trader.mt5.market_data import TickData
from gold_trader.mt5.positions import close_position
from gold_trader.trade_management.break_even import format_position_comment
from gold_trader.trade_management.partial_close import manage_partial_close
from tests._helpers import make_cfg, publish_terminal_account


def _make_spec(volume_min=0.05, volume_step=0.01, volume_max=10.0):
    return SymbolSpec(
        name="XAUUSD",
        point=0.01,
        digits=2,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        stops_level=20,
        freeze_level=10,
        visible=True,
        trade_mode=4,
        contract_size=100.0,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        filling_mode=3,
    )


def _make_tick(bid=2000.50, ask=2000.80):
    return TickData(
        bid=bid,
        ask=ask,
        last=2000.60,
        time=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_mt5():
    mock = MagicMock()
    mock.TRADE_ACTION_DEAL = 1
    mock.TRADE_ACTION_PENDING = 5
    mock.TRADE_ACTION_SLTP = 6
    mock.TRADE_ACTION_REMOVE = 8
    mock.ORDER_TYPE_BUY = 0
    mock.ORDER_TYPE_SELL = 1
    mock.ORDER_TIME_GTC = 0
    mock.ORDER_FILLING_FOK = 0
    mock.ORDER_FILLING_IOC = 1
    mock.ORDER_FILLING_RETURN = 2
    mock.TRADE_RETCODE_DONE = 10009
    mock.TRADE_RETCODE_INVALID_PRICE = 10015
    mock.TRADE_RETCODE_INVALID_STOPS = 10016
    mock.SYMBOL_TRADE_MODE_FULL = 4

    class MockCheck:
        retcode = 0
        comment = "Check OK"

    class MockSend:
        retcode = 10009
        comment = "Done"
        order = 12345

    mock.order_check.return_value = MockCheck()
    mock.order_send.return_value = MockSend()
    mock.last_error.return_value = (0, "Success")

    # Opt in only for this execution-layer fixture. The stub proves Demo
    # through account_info; a forged account_is_demo=True is not accepted.
    allow = ExecutionPermission(trading_enabled=True, dry_run=False)
    with (
        patch.object(mt5_conn, "MT5_AVAILABLE", True),
        patch.object(mt5_conn, "_mt5", mock),
    ):
        publish_terminal_account(0)
        assert refresh_verified_account_safety() is True
        with execution_permission(allow):
            yield mock


# ---------------------------------------------------------------------------
# FIX 1 — permission defaults fail-safe
# ---------------------------------------------------------------------------

def test_market_state_defaults_fail_safe():
    state = MarketState(
        connected=True,
        symbol_valid=True,
        market_open=True,
        server_trading_allowed=True,
        spread=0.25,
        open_position_count=0,
        pending_order_count=0,
        daily_pnl=0.0,
        account_balance=10000.0,
        account_equity=10000.0,
        now=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
    )
    assert state.terminal_trade_allowed is False
    assert state.account_trade_allowed is False
    assert state.expert_trade_allowed is False
    assert state.daily_pnl_known is False


def test_market_state_missing_permissions_blocks_trade():
    from gold_trader.risk.risk_manager import check_trade
    from gold_trader.models import OrderType, TradePlan

    state = MarketState(
        connected=True,
        symbol_valid=True,
        market_open=True,
        server_trading_allowed=True,
        spread=0.25,
        open_position_count=0,
        pending_order_count=0,
        daily_pnl=0.0,
        account_balance=10000.0,
        account_equity=10000.0,
        now=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
    )
    plan = TradePlan(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        entry=2000.0,
        sl=1995.0,
        tp=2010.0,
        volume=0.1,
        risk_amount=50.0,
    )
    cfg = make_cfg(trading_enabled=True, dry_run=False)
    spec = _make_spec(volume_min=0.01)
    decision = check_trade(plan, state, cfg, spec)
    assert not decision.allowed
    assert any("trading disabled" in f for f in decision.gate_failures)


def test_market_state_all_true_allows():
    from gold_trader.risk.risk_manager import check_trade
    from gold_trader.models import OrderType, TradePlan

    state = MarketState(
        connected=True,
        symbol_valid=True,
        market_open=True,
        server_trading_allowed=True,
        terminal_trade_allowed=True,
        account_trade_allowed=True,
        expert_trade_allowed=True,
        spread=0.25,
        open_position_count=0,
        pending_order_count=0,
        daily_pnl=0.0,
        account_balance=10000.0,
        account_equity=10000.0,
        now=datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
        daily_pnl_known=True,
    )
    plan = TradePlan(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        entry=2000.0,
        sl=1995.0,
        tp=2010.0,
        volume=0.1,
        risk_amount=50.0,
    )
    cfg = make_cfg(trading_enabled=True, dry_run=False)
    spec = _make_spec(volume_min=0.01)
    decision = check_trade(plan, state, cfg, spec)
    assert decision.allowed


# ---------------------------------------------------------------------------
# FIX 2 — close_position must fail when Bid/Ask unavailable
# ---------------------------------------------------------------------------

def test_close_position_missing_bid_fails_without_order_check(mock_mt5):
    spec = _make_spec()
    pos = PositionInfo(
        ticket=1,
        symbol="XAUUSD",
        is_buy=True,  # BUY close requires Bid
        volume=0.1,
        price_open=1990.0,
        sl=1980.0,
        tp=2020.0,
        profit=0.0,
        magic=77,
        comment="GB|sl=1980.00|vol=0.10",
        open_time=datetime.now(timezone.utc),
    )
    # Tick with Bid=0 (invalid)
    bad_tick = _make_tick(bid=0.0, ask=2000.5)

    result = close_position(pos, spec, tick=bad_tick)

    assert result.success is False
    assert "Bid/Ask unavailable" in result.comment
    assert not mock_mt5.order_check.called
    assert not mock_mt5.order_send.called


def test_close_position_missing_ask_fails_without_order_check(mock_mt5):
    spec = _make_spec()
    pos = PositionInfo(
        ticket=2,
        symbol="XAUUSD",
        is_buy=False,  # SELL close requires Ask
        volume=0.1,
        price_open=2010.0,
        sl=2020.0,
        tp=1990.0,
        profit=0.0,
        magic=77,
        comment="GB|sl=2020.00|vol=0.10",
        open_time=datetime.now(timezone.utc),
    )
    bad_tick = _make_tick(bid=2000.0, ask=0.0)

    result = close_position(pos, spec, tick=bad_tick)

    assert result.success is False
    assert "Bid/Ask unavailable" in result.comment
    assert not mock_mt5.order_check.called
    assert not mock_mt5.order_send.called


def test_close_position_no_tick_and_get_tick_fails(mock_mt5):
    """When tick=None and get_tick raises, close must fail locally."""
    spec = _make_spec()
    pos = PositionInfo(
        ticket=3,
        symbol="XAUUSD",
        is_buy=True,
        volume=0.1,
        price_open=1990.0,
        sl=1980.0,
        tp=2020.0,
        profit=0.0,
        magic=77,
        comment="GB|sl=1980.00|vol=0.10",
        open_time=datetime.now(timezone.utc),
    )
    # Mock get_tick to raise
    with patch("gold_trader.mt5.market_data.get_tick", side_effect=Exception("no tick")):
        result = close_position(pos, spec, tick=None)

    assert result.success is False
    assert "Bid/Ask unavailable" in result.comment
    assert not mock_mt5.order_check.called
    assert not mock_mt5.order_send.called


def test_close_position_valid_tick_still_uses_correct_price(mock_mt5):
    spec = _make_spec()
    pos_buy = PositionInfo(
        ticket=10,
        symbol="XAUUSD",
        is_buy=True,
        volume=0.5,
        price_open=1990.0,
        sl=1980.0,
        tp=2020.0,
        profit=0.0,
        magic=77,
        comment="GB|sl=1980.00|vol=0.50",
        open_time=datetime.now(timezone.utc),
    )
    tick = _make_tick(bid=2000.25, ask=2000.55)
    result = close_position(pos_buy, spec, tick=tick)
    assert result.success is True
    req = mock_mt5.order_send.call_args[0][0]
    assert req["type"] == mock_mt5.ORDER_TYPE_SELL
    assert req["price"] == 2000.25

    mock_mt5.order_check.reset_mock()
    mock_mt5.order_send.reset_mock()

    pos_sell = PositionInfo(
        ticket=11,
        symbol="XAUUSD",
        is_buy=False,
        volume=0.25,
        price_open=2010.0,
        sl=2020.0,
        tp=1990.0,
        profit=0.0,
        magic=77,
        comment="GB|sl=2020.00|vol=0.25",
        open_time=datetime.now(timezone.utc),
    )
    result2 = close_position(pos_sell, spec, tick=tick)
    assert result2.success is True
    req2 = mock_mt5.order_send.call_args[0][0]
    assert req2["type"] == mock_mt5.ORDER_TYPE_BUY
    assert req2["price"] == 2000.55


# ---------------------------------------------------------------------------
# FIX 3 — partial close remaining volume regression
# ---------------------------------------------------------------------------

def test_partial_close_case_a_current_0_06_min_0_05_close_0_04():
    """Case A: current 0.06, min 0.05, close 0.04 must NOT leave 0.02 invalid."""
    spec = _make_spec(volume_min=0.05, volume_step=0.01)
    # Levels: 66.67% close => target 0.02, close 0.04
    cfg = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.6667),))
    pos = PositionInfo(
        ticket=1,
        symbol="XAUUSD",
        is_buy=True,
        volume=0.06,
        price_open=2000.0,
        sl=0.0,
        tp=0.0,
        profit=0.0,
        magic=77,
        comment=format_position_comment("GB", 1995.0, 0.06, 2, volume_step=0.01),
        open_time=datetime.now(timezone.utc),
    )
    actions = manage_partial_close([pos], bid=2005.5, ask=2005.6, cfg=cfg, spec=spec)
    # close 0.04 < min 0.05, so should be skipped (no final level) OR full_close if final level
    # With no final level, must be []
    assert actions == [] or all(a.kind == "full_close" for a in actions)
    # Ensure no action leaves 0.02
    for a in actions:
        remaining = pos.volume - a.close_volume
        if a.kind == "partial_close":
            assert remaining >= spec.volume_min - 1e-9 or remaining <= 1e-9


def test_partial_close_case_a_with_final_level_full_closes():
    spec = _make_spec(volume_min=0.05, volume_step=0.01)
    cfg = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.6667), (3.0, 1.0)))
    pos = PositionInfo(
        ticket=2,
        symbol="XAUUSD",
        is_buy=True,
        volume=0.06,
        price_open=2000.0,
        sl=0.0,
        tp=0.0,
        profit=0.0,
        magic=77,
        comment=format_position_comment("GB", 1995.0, 0.06, 2, volume_step=0.01),
        open_time=datetime.now(timezone.utc),
    )
    actions = manage_partial_close([pos], bid=2005.5, ask=2005.6, cfg=cfg, spec=spec)
    # With final level, dust remainder should be full closed, not leave 0.02
    assert len(actions) == 1
    assert actions[0].kind == "full_close"
    assert actions[0].close_volume == pytest.approx(0.06)


def test_partial_close_case_b_valid_remainder():
    """Case B: 0.10 volume, min 0.01, close 0.05 remaining 0.05 valid."""
    spec = _make_spec(volume_min=0.01, volume_step=0.01)
    cfg = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.5),))
    pos = PositionInfo(
        ticket=3,
        symbol="XAUUSD",
        is_buy=True,
        volume=0.10,
        price_open=2000.0,
        sl=0.0,
        tp=0.0,
        profit=0.0,
        magic=77,
        comment=format_position_comment("GB", 1995.0, 0.10, 2, volume_step=0.01),
        open_time=datetime.now(timezone.utc),
    )
    actions = manage_partial_close([pos], bid=2005.5, ask=2005.6, cfg=cfg, spec=spec)
    assert len(actions) == 1
    assert actions[0].kind == "partial_close"
    assert actions[0].close_volume == pytest.approx(0.05)
    remaining = pos.volume - actions[0].close_volume
    assert remaining >= spec.volume_min - 1e-9


def test_partial_close_case_c_step_0_001_remaining_valid():
    """Case C: step 0.001, ensure remaining invariant holds."""
    spec = _make_spec(volume_min=0.001, volume_step=0.001)
    cfg = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.5),))
    pos = PositionInfo(
        ticket=4,
        symbol="XAUUSD",
        is_buy=True,
        volume=0.123,
        price_open=2000.0,
        sl=0.0,
        tp=0.0,
        profit=0.0,
        magic=77,
        comment=format_position_comment("GB", 1995.0, 0.123, 2, volume_step=0.001),
        open_time=datetime.now(timezone.utc),
    )
    actions = manage_partial_close([pos], bid=2005.5, ask=2005.6, cfg=cfg, spec=spec)
    assert len(actions) == 1
    # 50% of 0.123 = 0.0615 -> floored 0.061
    assert actions[0].close_volume == pytest.approx(0.061)
    remaining = pos.volume - actions[0].close_volume
    assert remaining >= spec.volume_min - 1e-9
    # remaining should be 0.062
    assert remaining == pytest.approx(0.062)


def test_partial_close_remaining_below_min_triggers_full_close_or_skip():
    """Remaining 0.04 < min 0.05 must not be left as partial."""
    spec = _make_spec(volume_min=0.05, volume_step=0.01)
    # 60% close of 0.10 = 0.06, remaining 0.04 < min
    cfg = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.6),))
    pos = PositionInfo(
        ticket=5,
        symbol="XAUUSD",
        is_buy=True,
        volume=0.10,
        price_open=2000.0,
        sl=0.0,
        tp=0.0,
        profit=0.0,
        magic=77,
        comment=format_position_comment("GB", 1995.0, 0.10, 2, volume_step=0.01),
        open_time=datetime.now(timezone.utc),
    )
    actions = manage_partial_close([pos], bid=2005.5, ask=2005.6, cfg=cfg, spec=spec)
    # Without final level, should skip (not leave invalid remainder)
    assert actions == []

    # With final level, should full close
    cfg_final = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.6), (3.0, 1.0)))
    actions_final = manage_partial_close([pos], bid=2005.5, ask=2005.6, cfg=cfg_final, spec=spec)
    assert len(actions_final) == 1
    assert actions_final[0].kind == "full_close"
    assert actions_final[0].close_volume == pytest.approx(0.10)
