"""Dedicated mock execution-layer tests for MetaTrader 5 integration.

Tests the MT5 execution and trade management layer without requiring
a real MT5 terminal (all MT5 API interactions are mocked).

Covers:
- Market data (valid Bid/Ask, last=0, missing tick, invalid prices)
- Account permissions (terminal/account/expert blocked and allowed)
- Market orders (valid, invalid, order_check reject, order_send reject, success)
- Close positions (BUY uses Bid, SELL uses Ask, rejection handling, deviation)
- SL/TP modification (valid, stops violation, freeze violation, weaker SL rejected, stronger accepted)
- Break-even + trailing (BE only, trailing only, conflict resolution, precedence)
- Partial close (volume steps 0.01 and 0.001, min volume, precision preservation)
- Pending orders (BUY_LIMIT, BUY_STOP, SELL_LIMIT, SELL_STOP, distance validation)
- Pending order type constants (official ENUM_ORDER_TYPE values, mock/fallback parity)
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import gold_trader.mt5.connection as mt5_conn
from gold_trader.models import (
    ManagementAction,
    OrderResult,
    OrderType,
    PendingOrderInfo,
    PositionInfo,
    SymbolSpec,
    TradePlan,
)
from gold_trader.mt5.market_data import TickData, get_tick, is_tick_usable
from gold_trader.mt5.orders import (
    _ORDER_TYPE_FALLBACK,
    _order_type_from_int,
    _order_type_value,
    place_buy_limit,
    place_buy_stop,
    place_market_buy,
    place_market_sell,
    place_sell_limit,
    place_sell_stop,
    send_plan,
    send_request,
)
from gold_trader.mt5.positions import close_position, modify_position_sltp
from gold_trader.trade_management.break_even import (
    format_position_comment,
    manage_break_even,
    parse_position_comment,
)
from gold_trader.trade_management.partial_close import manage_partial_close
from gold_trader.mt5.symbols import find_gold_symbol, is_gold_symbol
from gold_trader.trade_management.reconciliation import resolve_management_actions
from gold_trader.trade_management.trailing_stop import manage_trailing_stop
from tests._helpers import make_cfg


class MockMT5Tick:
    def __init__(self, bid=2000.50, ask=2000.80, last=2000.60, time=1700000000):
        self.bid = bid
        self.ask = ask
        self.last = last
        self.time = time


class MockTradeCheckResult:
    def __init__(self, retcode=0, comment="Done"):
        self.retcode = retcode
        self.comment = comment


class MockTradeSendResult:
    def __init__(self, retcode=10009, comment="Done", order=98765):
        self.retcode = retcode
        self.comment = comment
        self.order = order


def _make_spec(stops_level=20, freeze_level=10, digits=2, volume_step=0.01, volume_min=0.01):
    return SymbolSpec(
        name="XAUUSD",
        point=0.01,
        digits=digits,
        volume_min=volume_min,
        volume_max=10.0,
        volume_step=volume_step,
        stops_level=stops_level,
        freeze_level=freeze_level,
        visible=True,
        trade_mode=4,
        contract_size=100.0,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        filling_mode=3,
    )


def _make_tick(bid=2000.50, ask=2000.80, last=2000.60):
    return TickData(
        bid=bid,
        ask=ask,
        last=last,
        time=datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_mt5():
    """Configure mocked MT5 environment."""
    mock = MagicMock()
    mock.TRADE_ACTION_DEAL = 1
    mock.TRADE_ACTION_PENDING = 5
    mock.TRADE_ACTION_SLTP = 6
    mock.TRADE_ACTION_REMOVE = 8
    mock.ORDER_TYPE_BUY = 0
    mock.ORDER_TYPE_SELL = 1
    mock.ORDER_TYPE_BUY_LIMIT = 2
    # Official ENUM_ORDER_TYPE ordering: LIMIT/STOP alternate by side.
    mock.ORDER_TYPE_SELL_LIMIT = 3
    mock.ORDER_TYPE_BUY_STOP = 4
    mock.ORDER_TYPE_SELL_STOP = 5
    mock.ORDER_TIME_GTC = 0
    mock.ORDER_FILLING_FOK = 0
    mock.ORDER_FILLING_IOC = 1
    mock.ORDER_FILLING_RETURN = 2
    mock.TRADE_RETCODE_DONE = 10009
    mock.TRADE_RETCODE_DONE_PARTIAL = 10008
    mock.TRADE_RETCODE_PLACED = 10010
    mock.TRADE_RETCODE_INVALID_STOPS = 10016
    mock.TRADE_RETCODE_INVALID_PRICE = 10015
    mock.TRADE_RETCODE_INVALID_VOLUME = 10014
    mock.TRADE_RETCODE_NO_MONEY = 10019
    mock.SYMBOL_TRADE_MODE_FULL = 4

    # Default order_check and order_send responses
    mock.order_check.return_value = MockTradeCheckResult(0, "Check OK")
    mock.order_send.return_value = MockTradeSendResult(10009, "Order placed", 12345)
    mock.last_error.return_value = (0, "Success")

    with patch.object(mt5_conn, "MT5_AVAILABLE", True), patch.object(mt5_conn, "_mt5", mock):
        yield mock


# ===========================================================================
# 1. MARKET DATA TESTS (BUG 1)
# ===========================================================================

def test_market_data_valid_bid_ask(mock_mt5):
    mock_mt5.symbol_info_tick.return_value = MockMT5Tick(bid=2000.5, ask=2000.8, last=2000.6)
    tick = get_tick("XAUUSD")
    assert tick.bid == 2000.5
    assert tick.ask == 2000.8
    assert tick.last == 2000.6
    assert tick.is_usable is True


def test_market_data_last_is_zero(mock_mt5):
    # Forex/CFD: last is 0, bid/ask valid -> usable
    mock_mt5.symbol_info_tick.return_value = MockMT5Tick(bid=2000.5, ask=2000.8, last=0.0)
    tick = get_tick("XAUUSD")
    assert tick.bid == 2000.5
    assert tick.ask == 2000.8
    assert tick.last == 0.0
    assert tick.is_usable is True
    assert is_tick_usable(tick) is True


def test_market_data_missing_tick(mock_mt5):
    mock_mt5.symbol_info_tick.return_value = None
    with pytest.raises(mt5_conn.MT5DataError):
        get_tick("XAUUSD")


def test_market_data_invalid_prices(mock_mt5):
    mock_mt5.symbol_info_tick.return_value = MockMT5Tick(bid=0.0, ask=0.0, last=0.0)
    with pytest.raises(mt5_conn.MT5DataError):
        get_tick("XAUUSD")

    # Inverted spread
    inverted = MockMT5Tick(bid=2001.0, ask=2000.0)
    assert is_tick_usable(inverted) is False


# ===========================================================================
# 2. ACCOUNT PERMISSIONS TESTS (BUG 6)
# ===========================================================================

def test_account_permissions_blocked_and_allowed():
    from gold_trader.risk.risk_manager import check_trade
    from tests.test_risk import make_plan, make_state
    cfg = make_cfg(trading_enabled=True, dry_run=False)

    # Terminal blocked
    s1 = make_state(terminal_trade_allowed=False, account_trade_allowed=True, expert_trade_allowed=True)
    res1 = check_trade(make_plan(), s1, cfg, _make_spec())
    assert not res1.allowed
    assert any("terminal trading disabled" in f for f in res1.gate_failures)

    # Account blocked
    s2 = make_state(terminal_trade_allowed=True, account_trade_allowed=False, expert_trade_allowed=True)
    res2 = check_trade(make_plan(), s2, cfg, _make_spec())
    assert not res2.allowed
    assert any("account trading disabled" in f for f in res2.gate_failures)

    # Expert blocked
    s3 = make_state(terminal_trade_allowed=True, account_trade_allowed=True, expert_trade_allowed=False)
    res3 = check_trade(make_plan(), s3, cfg, _make_spec())
    assert not res3.allowed
    assert any("expert trading disabled" in f for f in res3.gate_failures)

    # All allowed
    s4 = make_state(terminal_trade_allowed=True, account_trade_allowed=True, expert_trade_allowed=True)
    res4 = check_trade(make_plan(), s4, cfg, _make_spec())
    assert res4.allowed


# ===========================================================================
# 3. MARKET ORDERS & PREFLIGHT order_check (BUG 8)
# ===========================================================================

def test_market_buy_successful(mock_mt5):
    spec = _make_spec()
    tick = _make_tick(bid=2000.0, ask=2000.3)
    result = place_market_buy(spec, tick, volume=0.1, sl=1995.0, tp=2010.0, magic=77, deviation=10, comment="buy test")

    assert mock_mt5.order_check.called
    assert mock_mt5.order_send.called
    assert result.success is True
    assert result.order == 12345


def test_market_sell_successful(mock_mt5):
    spec = _make_spec()
    tick = _make_tick(bid=2000.0, ask=2000.3)
    result = place_market_sell(spec, tick, volume=0.1, sl=2005.0, tp=1990.0, magic=77, deviation=10, comment="sell test")

    assert mock_mt5.order_check.called
    assert mock_mt5.order_send.called
    assert result.success is True


def test_order_check_rejection_prevents_order_send(mock_mt5):
    mock_mt5.order_check.return_value = MockTradeCheckResult(10019, "Not enough money")
    spec = _make_spec()
    tick = _make_tick()

    result = place_market_buy(spec, tick, volume=10.0, sl=1995.0, tp=2010.0, magic=77, deviation=10, comment="big order")

    assert mock_mt5.order_check.called
    assert not mock_mt5.order_send.called
    assert result.success is False
    assert result.retcode == 10019
    assert "Not enough money" in result.comment


def test_order_check_exception_handling(mock_mt5):
    mock_mt5.order_check.side_effect = RuntimeError("MT5 internal crash")
    spec = _make_spec()
    tick = _make_tick()

    result = place_market_buy(spec, tick, volume=0.1, sl=1995.0, tp=2010.0, magic=77, deviation=10, comment="err")

    assert not mock_mt5.order_send.called
    assert result.success is False
    assert "exception" in result.comment


def test_order_send_rejection_handled(mock_mt5):
    mock_mt5.order_check.return_value = MockTradeCheckResult(0, "Check OK")
    mock_mt5.order_send.return_value = MockTradeSendResult(10015, "Invalid price")
    spec = _make_spec()
    tick = _make_tick()

    result = place_market_buy(spec, tick, volume=0.1, sl=1995.0, tp=2010.0, magic=77, deviation=10, comment="err")

    assert mock_mt5.order_check.called
    assert mock_mt5.order_send.called
    assert result.success is False
    assert result.retcode == 10015


# ===========================================================================
# 4. CLOSE POSITION EXECUTION REQUEST (BUG 7)
# ===========================================================================

def test_close_buy_position_uses_bid_and_deviation(mock_mt5):
    spec = _make_spec()
    pos = PositionInfo(
        ticket=101, symbol="XAUUSD", is_buy=True, volume=0.5, price_open=1990.0,
        sl=1980.0, tp=2020.0, profit=50.0, magic=77, comment="GB|sl=1980.00|vol=0.50",
        open_time=datetime.now(timezone.utc),
    )
    tick = _make_tick(bid=2000.25, ask=2000.55)

    close_position(pos, spec, volume=0.5, magic=77, deviation=15, comment="close test", tick=tick)

    assert mock_mt5.order_check.called
    request = mock_mt5.order_send.call_args[0][0]
    # BUY closed via SELL type deal at Bid
    assert request["type"] == mock_mt5.ORDER_TYPE_SELL
    assert request["price"] == 2000.25
    assert request["deviation"] == 15
    assert request["volume"] == 0.5


def test_close_sell_position_uses_ask_and_deviation(mock_mt5):
    spec = _make_spec()
    pos = PositionInfo(
        ticket=102, symbol="XAUUSD", is_buy=False, volume=0.25, price_open=2010.0,
        sl=2020.0, tp=1990.0, profit=25.0, magic=77, comment="GB|sl=2020.00|vol=0.25",
        open_time=datetime.now(timezone.utc),
    )
    tick = _make_tick(bid=2000.25, ask=2000.55)

    close_position(pos, spec, volume=0.25, magic=77, deviation=25, comment="close sell", tick=tick)

    request = mock_mt5.order_send.call_args[0][0]
    # SELL closed via BUY type deal at Ask
    assert request["type"] == mock_mt5.ORDER_TYPE_BUY
    assert request["price"] == 2000.55
    assert request["deviation"] == 25
    assert request["volume"] == 0.25


def test_close_position_rejection(mock_mt5):
    mock_mt5.order_check.return_value = MockTradeCheckResult(10013, "Invalid request")
    spec = _make_spec()
    pos = PositionInfo(
        ticket=103, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=1995.0, tp=2010.0, profit=0.0, magic=77, comment="GB|sl=1995.00|vol=0.10",
        open_time=datetime.now(timezone.utc),
    )
    tick = _make_tick()

    result = close_position(pos, spec, tick=tick)
    assert not mock_mt5.order_send.called
    assert result.success is False
    assert result.retcode == 10013


# ===========================================================================
# 5. SL/TP MODIFICATION & SAFETY INVARIANTS (BUG 2 & BUG 3)
# ===========================================================================

def test_sltp_modify_valid(mock_mt5):
    spec = _make_spec(stops_level=20, freeze_level=10)
    pos = PositionInfo(
        ticket=201, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=1995.0, tp=2020.0, profit=50.0, magic=77, comment="GB|sl=1995.00|vol=0.10",
        open_time=datetime.now(timezone.utc),
    )
    tick = _make_tick(bid=2010.0, ask=2010.3)

    result = modify_position_sltp(pos, sl=2000.10, tp=2020.0, spec=spec, tick=tick)
    assert result.success is True
    assert mock_mt5.order_send.called
    req = mock_mt5.order_send.call_args[0][0]
    assert req["sl"] == 2000.10


def test_sltp_modify_stops_violation_rejected_before_send(mock_mt5):
    spec = _make_spec(stops_level=20)  # min stop distance = 0.20
    pos = PositionInfo(
        ticket=202, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=1995.0, tp=2020.0, profit=50.0, magic=77, comment="GB|sl=1995.00|vol=0.10",
        open_time=datetime.now(timezone.utc),
    )
    tick = _make_tick(bid=2000.0, ask=2000.3)  # Bid=2000.0, SL=1999.90 -> dist 0.10 < 0.20

    result = modify_position_sltp(pos, sl=1999.90, tp=2020.0, spec=spec, tick=tick)
    assert not mock_mt5.order_send.called
    assert result.success is False
    assert result.retcode == 10016
    assert "too close to Bid" in result.comment


def test_sltp_modify_freeze_violation_rejected_before_send(mock_mt5):
    spec = _make_spec(stops_level=10, freeze_level=50)  # freeze dist = 0.50
    pos = PositionInfo(
        ticket=203, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=1995.0, tp=2020.0, profit=50.0, magic=77, comment="GB|sl=1995.00|vol=0.10",
        open_time=datetime.now(timezone.utc),
    )
    tick = _make_tick(bid=2000.0, ask=2000.3)  # Bid=2000.0, SL=1999.60 -> dist 0.40 < 0.50

    result = modify_position_sltp(pos, sl=1999.60, tp=2020.0, spec=spec, tick=tick)
    assert not mock_mt5.order_send.called
    assert result.success is False
    assert result.retcode == 10016
    assert "within freeze level" in result.comment


def test_sltp_modify_weaker_sl_rejected(mock_mt5):
    spec = _make_spec()
    pos = PositionInfo(
        ticket=204, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=2000.10, tp=2020.0, profit=50.0, magic=77, comment="GB|sl=1995.00|vol=0.10",
        open_time=datetime.now(timezone.utc),
    )
    tick = _make_tick(bid=2015.0, ask=2015.3)

    # Trying to move SL backwards from 2000.10 to 1999.50
    result = modify_position_sltp(pos, sl=1999.50, tp=2020.0, spec=spec, tick=tick)
    assert not mock_mt5.order_send.called
    assert result.success is False
    assert result.retcode == 10016
    assert "reduces protection" in result.comment


def test_sltp_modify_stronger_sl_accepted(mock_mt5):
    spec = _make_spec()
    pos = PositionInfo(
        ticket=205, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=1995.0, tp=2020.0, profit=50.0, magic=77, comment="GB|sl=1995.00|vol=0.10",
        open_time=datetime.now(timezone.utc),
    )
    tick = _make_tick(bid=2015.0, ask=2015.3)

    result = modify_position_sltp(pos, sl=2002.50, tp=2020.0, spec=spec, tick=tick)
    assert mock_mt5.order_send.called
    assert result.success is True
    assert result.request["sl"] == 2002.50


# ===========================================================================
# 6. BREAK-EVEN + TRAILING RESOLUTION (BUG 3)
# ===========================================================================

def test_be_only():
    spec = _make_spec()
    cfg = make_cfg(break_even_enabled=True, break_even_r=1.0, break_even_buffer=0.1)
    pos = PositionInfo(
        ticket=301, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=1995.0, tp=0.0, profit=0.0, magic=77,
        comment=format_position_comment("GB", 1995.0, 0.1, 2),
        open_time=datetime.now(timezone.utc),
    )
    actions = manage_break_even([pos], bid=2005.5, ask=2005.8, cfg=cfg, spec=spec)
    assert len(actions) == 1
    assert actions[0].new_sl == pytest.approx(2000.10)


def test_trailing_only():
    spec = _make_spec()
    cfg = make_cfg(trailing_stop_enabled=True, trailing_atr_multiplier=2.0)
    pos = PositionInfo(
        ticket=302, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=1995.0, tp=0.0, profit=0.0, magic=77,
        comment=format_position_comment("GB", 1995.0, 0.1, 2),
        open_time=datetime.now(timezone.utc),
    )
    # price 2010.0, ATR 3.0, trailing dist 6.0 -> SL 2004.0
    actions = manage_trailing_stop([pos], bid=2010.0, ask=2010.3, atr_value=3.0, cfg=cfg, spec=spec)
    assert len(actions) == 1
    assert actions[0].new_sl == pytest.approx(2004.0)


def test_be_and_trailing_conflict_resolution():
    """BE and Trailing propose different SLs in same cycle."""
    pos = PositionInfo(
        ticket=303, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=1995.0, tp=0.0, profit=0.0, magic=77,
        comment=format_position_comment("GB", 1995.0, 0.1, 2),
        open_time=datetime.now(timezone.utc),
    )
    be_act = ManagementAction(kind="move_sl", ticket=303, new_sl=2000.10, description="BE")
    trail_act = ManagementAction(kind="move_sl", ticket=303, new_sl=1999.50, description="Trailing")

    resolved = resolve_management_actions([be_act, trail_act], [pos])
    assert len(resolved) == 1
    # Stronger SL wins (BE 2000.10 > 1999.50)
    assert resolved[0].new_sl == pytest.approx(2000.10)


def test_stronger_trailing_replaces_be():
    pos = PositionInfo(
        ticket=304, symbol="XAUUSD", is_buy=True, volume=0.1, price_open=2000.0,
        sl=1995.0, tp=0.0, profit=0.0, magic=77,
        comment=format_position_comment("GB", 1995.0, 0.1, 2),
        open_time=datetime.now(timezone.utc),
    )
    be_act = ManagementAction(kind="move_sl", ticket=304, new_sl=2000.10, description="BE")
    trail_act = ManagementAction(kind="move_sl", ticket=304, new_sl=2003.50, description="Trailing")

    resolved = resolve_management_actions([be_act, trail_act], [pos])
    assert len(resolved) == 1
    # Stronger SL wins (Trailing 2003.50 > 2000.10)
    assert resolved[0].new_sl == pytest.approx(2003.50)


def test_weaker_trailing_cannot_replace_be():
    pos = PositionInfo(
        ticket=305, symbol="XAUUSD", is_buy=False, volume=0.1, price_open=2000.0,
        sl=2005.0, tp=0.0, profit=0.0, magic=77,
        comment=format_position_comment("GB", 2005.0, 0.1, 2),
        open_time=datetime.now(timezone.utc),
    )
    # SELL: lower SL is stronger
    be_act = ManagementAction(kind="move_sl", ticket=305, new_sl=1999.90, description="BE")
    trail_act = ManagementAction(kind="move_sl", ticket=305, new_sl=2001.00, description="Trailing")

    resolved = resolve_management_actions([be_act, trail_act], [pos])
    assert len(resolved) == 1
    assert resolved[0].new_sl == pytest.approx(1999.90)


# ===========================================================================
# 7. PARTIAL CLOSE & PRECISION (BUG 4)
# ===========================================================================

def test_partial_close_step_0_01():
    spec = _make_spec(volume_step=0.01, volume_min=0.01)
    cfg = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.5),))
    pos = PositionInfo(
        ticket=401, symbol="XAUUSD", is_buy=True, volume=0.25, price_open=2000.0,
        sl=1995.0, tp=0.0, profit=0.0, magic=77,
        comment=format_position_comment("GB", 1995.0, 0.25, 2, volume_step=0.01),
        open_time=datetime.now(timezone.utc),
    )
    actions = manage_partial_close([pos], bid=2005.5, ask=2005.8, cfg=cfg, spec=spec)
    assert len(actions) == 1
    # 50% of 0.25 = 0.125 -> floored to step 0.01 = 0.12
    assert actions[0].close_volume == pytest.approx(0.12)


def test_partial_close_step_0_001():
    spec = _make_spec(volume_step=0.001, volume_min=0.001)
    cfg = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.5),))
    pos = PositionInfo(
        ticket=402, symbol="XAUUSD", is_buy=True, volume=0.123, price_open=2000.0,
        sl=1995.0, tp=0.0, profit=0.0, magic=77,
        comment=format_position_comment("GB", 1995.0, 0.123, 2, volume_step=0.001),
        open_time=datetime.now(timezone.utc),
    )
    actions = manage_partial_close([pos], bid=2005.5, ask=2005.8, cfg=cfg, spec=spec)
    assert len(actions) == 1
    # 50% of 0.123 = 0.0615 -> floored to step 0.001 = 0.061
    assert actions[0].close_volume == pytest.approx(0.061)


def test_partial_close_below_minimum_volume_skipped():
    spec = _make_spec(volume_step=0.01, volume_min=0.05)
    cfg = make_cfg(partial_close_enabled=True, partial_close_levels=((1.0, 0.2),))
    pos = PositionInfo(
        ticket=403, symbol="XAUUSD", is_buy=True, volume=0.10, price_open=2000.0,
        sl=1995.0, tp=0.0, profit=0.0, magic=77,
        comment=format_position_comment("GB", 1995.0, 0.10, 2),
        open_time=datetime.now(timezone.utc),
    )
    # 20% of 0.10 = 0.02 < 0.05 min -> skipped, no invalid order sent
    actions = manage_partial_close([pos], bid=2005.5, ask=2005.8, cfg=cfg, spec=spec)
    assert actions == []


@pytest.mark.parametrize("vol", [0.01, 0.10, 0.123, 0.001, 1.237])
def test_volume_comment_precision_preservation(vol):
    comment = format_position_comment("GB", 1995.00, vol, 2, volume_step=0.001)
    parsed = parse_position_comment(comment)
    assert parsed is not None
    assert parsed["initial_volume"] == pytest.approx(vol)


# ===========================================================================
# 8. PENDING ORDERS (BUG 9)
# ===========================================================================

def test_pending_buy_limit(mock_mt5):
    spec = _make_spec(stops_level=20)  # min distance 0.20
    tick = _make_tick(bid=2000.0, ask=2000.3)
    # Price must be at least 0.20 below bid 2000.0 -> e.g. 1990.0
    result = place_buy_limit(spec, tick, volume=0.1, price=1990.0, sl=1980.0, tp=2010.0, magic=77, deviation=10, comment="limit")
    assert result.success is True
    req = mock_mt5.order_send.call_args[0][0]
    assert req["action"] == mock_mt5.TRADE_ACTION_PENDING
    assert req["type"] == mock_mt5.ORDER_TYPE_BUY_LIMIT
    assert req["price"] == 1990.0


def test_pending_buy_stop(mock_mt5):
    spec = _make_spec(stops_level=20)
    tick = _make_tick(bid=2000.0, ask=2000.3)
    # Price must be at least 0.20 above ask 2000.3 -> e.g. 2010.0
    result = place_buy_stop(spec, tick, volume=0.1, price=2010.0, sl=2000.0, tp=2030.0, magic=77, deviation=10, comment="stop")
    assert result.success is True
    req = mock_mt5.order_send.call_args[0][0]
    assert req["action"] == mock_mt5.TRADE_ACTION_PENDING
    assert req["type"] == mock_mt5.ORDER_TYPE_BUY_STOP
    assert req["price"] == 2010.0


def test_pending_sell_limit(mock_mt5):
    spec = _make_spec(stops_level=20)
    tick = _make_tick(bid=2000.0, ask=2000.3)
    # Price must be at least 0.20 above ask 2000.3 -> e.g. 2010.0
    result = place_sell_limit(spec, tick, volume=0.1, price=2010.0, sl=2020.0, tp=1990.0, magic=77, deviation=10, comment="sell limit")
    assert result.success is True
    req = mock_mt5.order_send.call_args[0][0]
    assert req["action"] == mock_mt5.TRADE_ACTION_PENDING
    assert req["type"] == mock_mt5.ORDER_TYPE_SELL_LIMIT
    assert req["price"] == 2010.0


def test_pending_sell_stop(mock_mt5):
    spec = _make_spec(stops_level=20)
    tick = _make_tick(bid=2000.0, ask=2000.3)
    # Price must be at least 0.20 below bid 2000.0 -> e.g. 1990.0
    result = place_sell_stop(spec, tick, volume=0.1, price=1990.0, sl=2000.0, tp=1970.0, magic=77, deviation=10, comment="sell stop")
    assert result.success is True
    req = mock_mt5.order_send.call_args[0][0]
    assert req["action"] == mock_mt5.TRADE_ACTION_PENDING
    assert req["type"] == mock_mt5.ORDER_TYPE_SELL_STOP
    assert req["price"] == 1990.0


# ---------------------------------------------------------------------------
# 8b. PENDING ORDER TYPE CONSTANTS (regression: BUY_STOP/SELL_LIMIT swap)
# ---------------------------------------------------------------------------

#: Official ENUM_ORDER_TYPE values shipped by the MetaTrader5 Python package
#: (MetaTrader5/__init__.py, "order types, ENUM_ORDER_TYPE"). Hard-coded on
#: purpose: these tests must fail if the production fallback table OR the
#: test mock drifts away from the real terminal constants.
OFFICIAL_ORDER_TYPES = {
    "BUY": 0,
    "SELL": 1,
    "BUY_LIMIT": 2,
    "SELL_LIMIT": 3,
    "BUY_STOP": 4,
    "SELL_STOP": 5,
}


def test_order_type_fallback_matches_official_mt5_constants():
    """The non-Windows fallback table must equal the real MT5 constants."""
    assert _ORDER_TYPE_FALLBACK == OFFICIAL_ORDER_TYPES


def test_mock_mt5_order_types_match_official_constants(mock_mt5):
    """The mock must faithfully represent the real MetaTrader5 constants."""
    for name, value in OFFICIAL_ORDER_TYPES.items():
        assert getattr(mock_mt5, f"ORDER_TYPE_{name}") == value, name


def test_mock_and_production_order_types_agree(mock_mt5):
    """Mock and production must never disagree about an order type value."""
    for order_type in OrderType:
        assert _order_type_value(order_type) == getattr(
            mock_mt5, f"ORDER_TYPE_{order_type.value}"
        ), order_type.value


def test_order_type_int_round_trip():
    """int -> OrderType is the exact inverse (BUY_STOP is not SELL_LIMIT)."""
    for order_type in OrderType:
        assert _order_type_from_int(_order_type_value(order_type)) is order_type


def test_pending_requests_carry_official_order_type_values(mock_mt5):
    """Each place_* helper must emit the official numeric order type."""
    spec = _make_spec(stops_level=20)  # min distance 0.20
    tick = _make_tick(bid=2000.0, ask=2000.3)
    cases = [
        (place_buy_limit, 1990.0, OFFICIAL_ORDER_TYPES["BUY_LIMIT"]),
        (place_buy_stop, 2010.0, OFFICIAL_ORDER_TYPES["BUY_STOP"]),
        (place_sell_limit, 2010.0, OFFICIAL_ORDER_TYPES["SELL_LIMIT"]),
        (place_sell_stop, 1990.0, OFFICIAL_ORDER_TYPES["SELL_STOP"]),
    ]
    for place_fn, price, expected_type in cases:
        mock_mt5.order_send.reset_mock()
        result = place_fn(
            spec, tick, volume=0.1, price=price, sl=1980.0, tp=2030.0,
            magic=77, deviation=10, comment="regression",
        )
        assert result.success is True, place_fn.__name__
        req = mock_mt5.order_send.call_args[0][0]
        assert req["action"] == 5, place_fn.__name__  # TRADE_ACTION_PENDING
        assert req["type"] == expected_type, place_fn.__name__


def test_pending_invalid_distances_rejected_locally(mock_mt5):
    spec = _make_spec(stops_level=20)  # min distance 0.20
    tick = _make_tick(bid=2000.0, ask=2000.3)

    # BUY_LIMIT not below bid - 0.20
    r1 = place_buy_limit(spec, tick, volume=0.1, price=1999.90, sl=1980.0, tp=2010.0, magic=77, deviation=10, comment="bad")
    assert not mock_mt5.order_send.called
    assert r1.success is False

    # BUY_STOP not above ask + 0.20
    r2 = place_buy_stop(spec, tick, volume=0.1, price=2000.40, sl=1980.0, tp=2010.0, magic=77, deviation=10, comment="bad")
    assert not mock_mt5.order_send.called
    assert r2.success is False

    # SELL_LIMIT not above ask + 0.20
    r3 = place_sell_limit(spec, tick, volume=0.1, price=2000.40, sl=2020.0, tp=1980.0, magic=77, deviation=10, comment="bad")
    assert not mock_mt5.order_send.called
    assert r3.success is False

    # SELL_STOP not below bid - 0.20
    r4 = place_sell_stop(spec, tick, volume=0.1, price=1999.90, sl=2020.0, tp=1980.0, magic=77, deviation=10, comment="bad")
    assert not mock_mt5.order_send.called
    assert r4.success is False


# ===========================================================================
# 9. GOLD SYMBOL DISCOVERY & VALIDATION (BUG 5)
# ===========================================================================

class MockSymbolInfo:
    def __init__(self, name="XAUUSD", visible=True, trade_mode=4):
        self.name = name
        self.visible = visible
        self.trade_mode = trade_mode
        self.point = 0.01
        self.digits = 2
        self.volume_min = 0.01
        self.volume_max = 10.0
        self.volume_step = 0.01
        self.stops_level = 20
        self.freeze_level = 10
        self.trade_contract_size = 100.0
        self.trade_tick_size = 0.01
        self.trade_tick_value = 1.0
        self.filling_mode = 3


def test_find_gold_symbol_explicit_non_gold_rejected(mock_mt5):
    """Explicit SYMBOL=EURUSD must be rejected before trading."""
    mock_mt5.symbols_get.return_value = [MockSymbolInfo("EURUSD")]
    with pytest.raises(ValueError, match="not a Gold/XAU instrument"):
        find_gold_symbol(preferred="EURUSD")


def test_find_gold_symbol_explicit_gold_accepted(mock_mt5):
    """Explicit SYMBOL=XAUUSDm must be accepted."""
    sym = MockSymbolInfo("XAUUSDm")
    mock_mt5.symbols_get.return_value = [sym]
    mock_mt5.symbol_info.return_value = sym
    spec = find_gold_symbol(preferred="XAUUSDm")
    assert spec.name == "XAUUSDm"


def test_find_gold_symbol_auto_discovery_ignores_non_gold(mock_mt5):
    """Auto discovery must ignore EURUSD and select GOLD."""
    s1 = MockSymbolInfo("EURUSD")
    s2 = MockSymbolInfo("GOLD")
    mock_mt5.symbols_get.return_value = [s1, s2]
    mock_mt5.symbol_info.side_effect = lambda n: s2 if n == "GOLD" else s1
    spec = find_gold_symbol()
    assert spec.name == "GOLD"

