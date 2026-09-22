"""Demo safety is part of the central execution gate, not only start().

A REAL or unknown account must not reach order_check or order_send, even
when cycle() is called directly with live switches and due management
actions. A per-call permission cannot mark that account as Demo.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from gold_trader.main import TradingBot
from gold_trader.models import OrderType, PendingOrderInfo, PositionInfo, TradePlan
from gold_trader.mt5.execution_gate import (
    ExecutionPermission,
    execution_permission,
    get_execution_permission,
    install_execution_permission,
    refresh_verified_account_safety,
    verified_account_is_demo,
)
from gold_trader.mt5.market_data import TickData
from gold_trader.mt5.orders import (
    delete_order,
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
)
from gold_trader.trade_management.partial_close import manage_partial_close
from gold_trader.trade_management.pending_orders import plan_cleanup
from gold_trader.trade_management.trailing_stop import manage_trailing_stop
from tests._helpers import make_cfg, make_uptrend_with_engulfing, publish_terminal_account
from tests.test_execution_safety import (
    _exercise_break_even,
    _exercise_entry,
    _exercise_expired,
    _exercise_full,
    _exercise_partial,
    _exercise_pending,
    _exercise_trailing,
    _expired_pending,
    _position,
    _spec,
)

NOW = datetime.now(timezone.utc)
TICK = TickData(bid=2010.0, ask=2010.3, last=2010.1, time=NOW)
DEMO = 0
REAL = 2
PHRASE = "WOULD EXECUTE: blocked by demo safety"

# (trading_enabled, dry_run) combinations that must never send.
CLOSED_SWITCHES = [
    pytest.param(False, True, id="disabled_dry_run"),
    pytest.param(True, True, id="enabled_dry_run"),
    pytest.param(False, False, id="disabled_live"),
]


def _account(bot: TradingBot, trade_mode) -> None:
    """Point the bot at a stub account. None means the type cannot be read."""
    bot.conn.is_connected = lambda: True
    bot.conn.terminal_info = lambda: {"trade_allowed": True, "connected": True}

    def account_info():
        info = {
            "balance": 10_000.0,
            "equity": 10_000.0,
            "trade_allowed": True,
            "trade_expert": True,
        }
        if trade_mode is not None:
            info["trade_mode"] = trade_mode
        return info

    bot.conn.account_info = account_info
    if trade_mode is None:
        publish_terminal_account()
    else:
        publish_terminal_account(trade_mode)


def _bot(trading_enabled: bool, dry_run: bool, **cfg_overrides) -> TradingBot:
    cfg = make_cfg(
        trading_enabled=trading_enabled,
        dry_run=dry_run,
        max_open_positions=5,
        **cfg_overrides,
    )
    bot = TradingBot(cfg)
    bot.spec = _spec()
    bot.conn.is_connected = lambda: True
    return bot


def _expired() -> PendingOrderInfo:
    pending = _expired_pending()
    # cycle() compares expiry with utcnow(), not the frozen test clock.
    return PendingOrderInfo(
        ticket=pending.ticket,
        symbol=pending.symbol,
        order_type=pending.order_type,
        price=pending.price,
        volume=pending.volume,
        sl=pending.sl,
        tp=pending.tp,
        magic=pending.magic,
        comment=pending.comment,
        time_setup=NOW - timedelta(days=2),
        time_expire=NOW - timedelta(hours=1),
    )


def _run_cycle(bot: TradingBot, positions, pendings) -> None:
    from unittest.mock import patch

    df = make_uptrend_with_engulfing()
    with (
        patch("gold_trader.main.mt5_positions.get_positions", return_value=positions),
        patch("gold_trader.main.mt5_orders.get_pending_orders", return_value=pendings),
        patch("gold_trader.main.market_data.get_tick", return_value=TICK),
        patch("gold_trader.main.market_data.get_candles", return_value=df),
    ):
        bot.cycle()


def _assert_blocked(mt5_stub, caplog) -> None:
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0
    assert PHRASE in caplog.text


def _due_management_bot(trading_enabled: bool, dry_run: bool) -> TradingBot:
    return _bot(
        trading_enabled,
        dry_run,
        break_even_enabled=True,
        partial_close_enabled=True,
        partial_close_levels=((1.0, 0.5),),
        trailing_stop_enabled=True,
        trailing_atr_multiplier=1.0,
    )


@pytest.mark.parametrize("trading_enabled,dry_run", CLOSED_SWITCHES + [
    pytest.param(True, False, id="enabled_live"),
])
def test_real_cycle_never_sends(trading_enabled, dry_run, mt5_stub, caplog):
    """Direct cycle() on REAL, including the live-switch combination."""
    caplog.set_level(logging.INFO)
    bot = _due_management_bot(trading_enabled, dry_run)
    _account(bot, REAL)
    pos = _position()
    pending = _expired()
    assert manage_break_even([pos], TICK.bid, TICK.ask, bot.cfg, bot.spec)
    assert manage_trailing_stop([pos], TICK.bid, TICK.ask, 1.2, bot.cfg, bot.spec)
    assert manage_partial_close([pos], TICK.bid, TICK.ask, bot.cfg, bot.spec)
    assert plan_cleanup([pending], NOW)
    _run_cycle(bot, [pos], [pending])
    _assert_blocked(mt5_stub, caplog)
    assert get_execution_permission().account_is_demo is False
    assert get_execution_permission().allows_order_send is False


@pytest.mark.parametrize(
    "path,exerciser,cfg",
    [
        ("entry", _exercise_entry, {}),
        ("break_even", _exercise_break_even, {
            "break_even_enabled": True,
            "partial_close_enabled": False,
            "trailing_stop_enabled": False,
        }),
        ("trailing", _exercise_trailing, {
            "break_even_enabled": False,
            "partial_close_enabled": False,
            "trailing_stop_enabled": True,
            "trailing_atr_multiplier": 1.0,
        }),
        ("partial_close", _exercise_partial, {
            "break_even_enabled": False,
            "trailing_stop_enabled": False,
            "partial_close_enabled": True,
            "partial_close_levels": ((1.0, 0.5),),
        }),
        ("full_close", _exercise_full, {
            "break_even_enabled": False,
            "trailing_stop_enabled": False,
            "partial_close_enabled": True,
            "partial_close_levels": ((1.0, 1.0),),
        }),
        ("expired_pending", _exercise_expired, {}),
        ("pending_placement", _exercise_pending, {}),
    ],
)
def test_real_live_blocks_every_send_path(path, exerciser, cfg, mt5_stub, caplog):
    """Each order_send path, after the bot has installed a REAL account."""
    caplog.set_level(logging.INFO)
    bot = _bot(True, False, **cfg)
    _account(bot, REAL)
    bot._install_execution_gate()
    exerciser(bot, mt5_stub)
    _assert_blocked(mt5_stub, caplog)


def test_real_live_cycle_blocks_each_management_and_entry_path(mt5_stub, caplog):
    """The same paths, entered through cycle() itself rather than helpers."""
    caplog.set_level(logging.INFO)
    pos = _position()
    pending = _expired()

    entry_bot = _bot(True, False, break_even_enabled=False, partial_close_enabled=False, trailing_stop_enabled=False)
    _account(entry_bot, REAL)
    _run_cycle(entry_bot, [], [])

    be_bot = _bot(True, False, break_even_enabled=True, partial_close_enabled=False, trailing_stop_enabled=False)
    _account(be_bot, REAL)
    assert manage_break_even([pos], TICK.bid, TICK.ask, be_bot.cfg, be_bot.spec)
    _run_cycle(be_bot, [pos], [])

    trail_bot = _bot(
        True, False,
        break_even_enabled=False,
        partial_close_enabled=False,
        trailing_stop_enabled=True,
        trailing_atr_multiplier=1.0,
    )
    _account(trail_bot, REAL)
    assert manage_trailing_stop([pos], TICK.bid, TICK.ask, 1.2, trail_bot.cfg, trail_bot.spec)
    _run_cycle(trail_bot, [pos], [])

    partial_bot = _bot(
        True, False,
        break_even_enabled=False,
        trailing_stop_enabled=False,
        partial_close_enabled=True,
        partial_close_levels=((1.0, 0.5),),
    )
    _account(partial_bot, REAL)
    assert manage_partial_close([pos], TICK.bid, TICK.ask, partial_bot.cfg, partial_bot.spec)[0].kind == "partial_close"
    _run_cycle(partial_bot, [pos], [])

    full_bot = _bot(
        True, False,
        break_even_enabled=False,
        trailing_stop_enabled=False,
        partial_close_enabled=True,
        partial_close_levels=((1.0, 1.0),),
    )
    _account(full_bot, REAL)
    assert manage_partial_close([pos], TICK.bid, TICK.ask, full_bot.cfg, full_bot.spec)[0].kind == "full_close"
    _run_cycle(full_bot, [pos], [])

    delete_bot = _bot(True, False, break_even_enabled=False, partial_close_enabled=False, trailing_stop_enabled=False)
    _account(delete_bot, REAL)
    assert plan_cleanup([pending], NOW)
    _run_cycle(delete_bot, [], [pending])

    _assert_blocked(mt5_stub, caplog)


def test_real_permission_override_cannot_reopen_demo_safety(mt5_stub, caplog):
    caplog.set_level(logging.INFO)
    bot = _bot(True, False)
    _account(bot, REAL)
    bot._install_execution_gate()
    lie = ExecutionPermission(trading_enabled=True, dry_run=False, account_is_demo=True)
    result = send_request(
        {"action": 1, "symbol": "XAUUSD", "volume": 0.1, "type": 0},
        permission=lie,
    )
    assert result.blocked_by_safety
    assert PHRASE in result.comment
    assert get_execution_permission().account_is_demo is False
    # A replaced context must not survive into cycle(): the bot reinstalls
    # the account fact before any management or entry send.
    pos = _position()
    pending = _expired()
    with execution_permission(lie):
        _run_cycle(bot, [pos], [pending])
    _assert_blocked(mt5_stub, caplog)


def test_real_direct_api_after_cycle_stays_blocked(mt5_stub, caplog):
    """Market, pending, SL/TP, close and delete after a REAL cycle()."""
    caplog.set_level(logging.INFO)
    bot = _due_management_bot(True, False)
    _account(bot, REAL)
    _run_cycle(bot, [_position()], [_expired()])
    spec = _spec()
    pos = _position()
    blocked = [
        place_market_buy(spec, TICK, 0.1, 1995.0, 2020.0, 77, 10, "buy"),
        place_market_sell(spec, TICK, 0.1, 2025.0, 1990.0, 77, 10, "sell"),
        place_buy_limit(spec, TICK, 0.1, 1990.0, 1980.0, 2030.0, 77, 10, "bl"),
        place_buy_stop(spec, TICK, 0.1, 2020.0, 1980.0, 2030.0, 77, 10, "bs"),
        place_sell_limit(spec, TICK, 0.1, 2020.0, 1980.0, 2030.0, 77, 10, "sl"),
        place_sell_stop(spec, TICK, 0.1, 1990.0, 1980.0, 2030.0, 77, 10, "ss"),
        modify_position_sltp(pos, 2000.10, 2030.0, spec, tick=TICK),
        close_position(pos, spec, volume=0.05, magic=77, tick=TICK),
        delete_order(900),
    ]
    assert all(item.blocked_by_safety for item in blocked)
    assert all(PHRASE in item.comment for item in blocked)
    _assert_blocked(mt5_stub, caplog)


@pytest.mark.parametrize("trading_enabled,dry_run", CLOSED_SWITCHES)
def test_demo_closed_switches_do_not_send(trading_enabled, dry_run, mt5_stub, caplog):
    caplog.set_level(logging.INFO)
    bot = _due_management_bot(trading_enabled, dry_run)
    _account(bot, DEMO)
    _run_cycle(bot, [_position()], [_expired()])
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0
    assert get_execution_permission().account_is_demo is True
    assert get_execution_permission().allows_order_send is False


def test_demo_live_cycle_reaches_stub_when_risk_gates_pass(mt5_stub):
    bot = _bot(
        True, False,
        break_even_enabled=False,
        partial_close_enabled=False,
        trailing_stop_enabled=False,
    )
    _account(bot, DEMO)
    _run_cycle(bot, [], [])
    assert mt5_stub.order_check.call_count > 0
    assert mt5_stub.order_send.call_count > 0
    request = mt5_stub.order_send.call_args[0][0]
    assert request["action"] == 1
    assert request["symbol"] == "XAUUSD"
    assert request["volume"] > 0


def test_unknown_account_live_cycle_does_not_send(mt5_stub, caplog):
    caplog.set_level(logging.INFO)
    bot = _due_management_bot(True, False)
    _account(bot, None)
    _run_cycle(bot, [_position()], [_expired()])
    _assert_blocked(mt5_stub, caplog)
    assert get_execution_permission().account_is_demo is False

    # Unreadable account info is also unknown, so the gate stays closed.
    mt5_stub.order_check.reset_mock()
    mt5_stub.order_send.reset_mock()
    broken = _bot(True, False)

    def explode():
        raise RuntimeError("account_info unavailable")

    broken.conn.account_info = explode
    publish_terminal_account(explode=True)
    broken._install_execution_gate()
    assert get_execution_permission().account_is_demo is False
    result = send_request(
        {"action": 1, "symbol": "XAUUSD", "volume": 0.1, "type": 0},
    )
    assert result.blocked_by_safety
    assert PHRASE in result.comment
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0


def test_env_cannot_mark_the_account_as_demo(monkeypatch, mt5_stub, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("ACCOUNT_IS_DEMO", "true")
    monkeypatch.setenv("ACCOUNT_TRADE_MODE", "0")
    result = place_market_buy(
        _spec(), TICK, volume=0.1, sl=1995.0, tp=2020.0, magic=77, deviation=10, comment="env"
    )
    assert result.blocked_by_safety
    assert PHRASE in result.comment
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0


def _lie() -> ExecutionPermission:
    return ExecutionPermission(trading_enabled=True, dry_run=False, account_is_demo=True)


def _market_request() -> dict:
    return {"action": 1, "symbol": "XAUUSD", "volume": 0.1, "type": 0}


def _window_attempts():
    """Paths that must be refused before order_check and order_send."""
    spec = _spec()
    pos = _position()
    plan = TradePlan(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        entry=TICK.ask,
        sl=1995.0,
        tp=2020.0,
        volume=0.1,
        risk_amount=50.0,
        comment="window",
    )
    return [
        place_market_buy(spec, TICK, 0.1, 1995.0, 2020.0, 77, 10, "buy"),
        place_market_sell(spec, TICK, 0.1, 2025.0, 1990.0, 77, 10, "sell"),
        place_buy_limit(spec, TICK, 0.1, 1990.0, 1980.0, 2030.0, 77, 10, "bl"),
        place_buy_stop(spec, TICK, 0.1, 2020.0, 1980.0, 2030.0, 77, 10, "bs"),
        place_sell_limit(spec, TICK, 0.1, 2020.0, 1980.0, 2030.0, 77, 10, "sl"),
        place_sell_stop(spec, TICK, 0.1, 1990.0, 1980.0, 2030.0, 77, 10, "ss"),
        modify_position_sltp(pos, 2000.10, 2030.0, spec, tick=TICK),
        close_position(pos, spec, volume=0.05, magic=77, tick=TICK),
        close_position(pos, spec, volume=None, magic=77, tick=TICK),
        delete_order(900),
        send_plan(plan, spec, TICK, 77, 10),
        send_request(_market_request(), permission=_lie()),
    ]


def _assert_attempts_blocked(attempts, mt5_stub, before_check: int, before_send: int) -> None:
    assert all(item.blocked_by_safety for item in attempts)
    assert all(PHRASE in item.comment for item in attempts)
    assert mt5_stub.order_check.call_count == before_check
    assert mt5_stub.order_send.call_count == before_send


def _patch_reconnect(monkeypatch) -> None:
    monkeypatch.setattr(
        "gold_trader.main.mt5_symbols.find_gold_symbol",
        lambda *args, **kwargs: _spec(),
    )


def test_reconnect_drops_stale_demo_until_current_account_is_verified(mt5_stub, monkeypatch, caplog):
    """After reconnect, the old Demo proof must not authorize any send.

    The window before the next cycle() — and the sleep inside reconnect,
    before verification — stays closed on REAL, unknown, and an unverified
    Demo stub. A later successful verification is what re-opens Demo.
    """
    caplog.set_level(logging.INFO)
    _patch_reconnect(monkeypatch)
    probes = []

    def _sleep(*_args, **_kwargs):
        probes.append(verified_account_is_demo())
        result = delete_order(900)
        assert result.blocked_by_safety
        assert PHRASE in result.comment

    monkeypatch.setattr("gold_trader.main.time.sleep", _sleep)

    bot = _bot(True, False)
    _account(bot, DEMO)
    bot._install_execution_gate()
    assert verified_account_is_demo() is True
    assert get_execution_permission().allows_order_send is True
    opened = place_market_buy(_spec(), TICK, 0.1, 1995.0, 2020.0, 77, 10, "demo")
    assert opened.success is True
    assert mt5_stub.order_check.call_count == 1
    assert mt5_stub.order_send.call_count == 1

    mt5_stub.order_check.reset_mock()
    mt5_stub.order_send.reset_mock()
    _account(bot, REAL)
    bot._reconnect()
    assert probes and probes[-1] is False
    assert verified_account_is_demo() is False
    assert get_execution_permission().account_is_demo is False
    assert get_execution_permission().trading_enabled is True
    assert get_execution_permission().dry_run is False
    assert get_execution_permission().allows_order_send is False
    # No cycle() yet. Every execution path in the gap must stay at 0.
    _assert_attempts_blocked(_window_attempts(), mt5_stub, 0, 0)

    _account(bot, None)
    bot._reconnect()
    assert verified_account_is_demo() is False
    unknown = place_market_buy(_spec(), TICK, 0.1, 1995.0, 2020.0, 77, 10, "unknown")
    assert unknown.blocked_by_safety
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0

    # Changing the stub to Demo is not verification. A send must not loosen.
    _account(bot, DEMO)
    assert verified_account_is_demo() is False
    premature = place_market_buy(_spec(), TICK, 0.1, 1995.0, 2020.0, 77, 10, "early")
    assert premature.blocked_by_safety
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0

    bot._reconnect()
    assert verified_account_is_demo() is True
    assert get_execution_permission().allows_order_send is True
    # The sleep probe ran before that verification and stayed closed.
    assert probes[-1] is False
    reopened = place_market_buy(_spec(), TICK, 0.1, 1995.0, 2020.0, 77, 10, "again")
    assert reopened.success is True
    assert mt5_stub.order_check.call_count == 1
    assert mt5_stub.order_send.call_count == 1


def test_failed_reconnect_stays_closed_until_later_verification(mt5_stub, monkeypatch):
    _patch_reconnect(monkeypatch)
    monkeypatch.setattr("gold_trader.main.time.sleep", lambda *_a, **_k: None)
    bot = _bot(True, False)
    _account(bot, DEMO)
    bot._install_execution_gate()
    assert verified_account_is_demo() is True

    mt5_stub.initialize.return_value = False
    bot._reconnect()
    assert verified_account_is_demo() is False
    assert not get_execution_permission().allows_order_send

    # The terminal answers Demo again, but the failed reconnect did not verify.
    mt5_stub.initialize.return_value = True
    _account(bot, DEMO)
    denied = delete_order(900)
    assert denied.blocked_by_safety
    assert PHRASE in denied.comment
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0

    bot._install_execution_gate()
    assert verified_account_is_demo() is True
    opened = place_market_buy(_spec(), TICK, 0.1, 1995.0, 2020.0, 77, 10, "later")
    assert opened.success is True
    assert mt5_stub.order_send.call_count == 1


def test_forged_account_is_demo_cannot_open_real_or_unknown(mt5_stub, caplog):
    """A declared account_is_demo=True is not broker proof."""
    caplog.set_level(logging.INFO)
    lie = _lie()
    assert lie.account_is_demo is True
    assert lie.allows_order_send is False

    publish_terminal_account(REAL)
    install_execution_permission(lie)
    assert get_execution_permission().trading_enabled is True
    assert get_execution_permission().dry_run is False
    assert get_execution_permission().account_is_demo is False
    assert verified_account_is_demo() is False
    assert get_execution_permission().allows_order_send is False

    installed = send_request(_market_request())
    assert installed.blocked_by_safety
    assert PHRASE in installed.comment

    with execution_permission(lie):
        assert get_execution_permission().account_is_demo is False
        assert get_execution_permission().trading_enabled is True
        replaced = send_request(_market_request())
        assert replaced.blocked_by_safety
        planned = send_plan(
            TradePlan(
                symbol="XAUUSD",
                order_type=OrderType.BUY,
                entry=TICK.ask,
                sl=1995.0,
                tp=2020.0,
                volume=0.1,
                risk_amount=50.0,
                comment="forged",
            ),
            _spec(),
            TICK,
            77,
            10,
        )
        assert planned.blocked_by_safety

    install_execution_permission(ExecutionPermission(trading_enabled=True, dry_run=False))
    overridden = send_request(_market_request(), permission=lie)
    assert overridden.blocked_by_safety
    removed = delete_order(900)
    assert removed.blocked_by_safety
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0

    # A previously verified Demo latch must not survive a REAL read.
    publish_terminal_account(DEMO)
    assert refresh_verified_account_safety() is True
    install_execution_permission(ExecutionPermission(trading_enabled=True, dry_run=False))
    assert get_execution_permission().allows_order_send is True
    publish_terminal_account(REAL)
    assert verified_account_is_demo() is True
    stale = send_request(_market_request(), permission=lie)
    assert stale.blocked_by_safety
    assert PHRASE in stale.comment
    assert verified_account_is_demo() is False
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0

    with execution_permission(lie):
        still = send_request(_market_request())
    assert still.blocked_by_safety
    install_execution_permission(lie)
    still_installed = send_plan(
        TradePlan(
            symbol="XAUUSD",
            order_type=OrderType.BUY_LIMIT,
            entry=1990.0,
            sl=1980.0,
            tp=2030.0,
            volume=0.1,
            risk_amount=50.0,
            comment="forged-pending",
        ),
        _spec(),
        TICK,
        77,
        10,
    )
    assert still_installed.blocked_by_safety
    assert mt5_stub.order_check.call_count == 0
    assert mt5_stub.order_send.call_count == 0
