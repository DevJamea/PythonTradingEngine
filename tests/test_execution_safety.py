"""Regression: DRY_RUN / TRADING_ENABLED must block every order_send path.

Before the central gate, ``risk_manager`` blocked new entries but
``_apply_actions`` still reached ``mt5.order_send`` for break-even, trailing,
partial/full close and expired-pending deletion.

These tests drive the production paths (and the direct execution API) against
an MT5 stub. A closed gate must leave ``order_send`` at 0 and log WOULD.
The live combination may reach the stub only when the risk gates also pass.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from gold_trader.config import Config
from gold_trader.main import TradingBot
from gold_trader.models import (
    MarketState,
    OrderType,
    PendingOrderInfo,
    PositionInfo,
    SymbolSpec,
    TradePlan,
)
from gold_trader.mt5.execution_gate import (
    ExecutionPermission,
    execution_permission,
    get_execution_permission,
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
from gold_trader.trade_management.break_even import (
    format_position_comment,
    manage_break_even,
)
from gold_trader.trade_management.partial_close import manage_partial_close
from gold_trader.trade_management.pending_orders import plan_cleanup
from gold_trader.trade_management.trailing_stop import manage_trailing_stop
from tests._helpers import (
    make_cfg,
    make_downtrend_with_engulfing,
    make_uptrend_with_engulfing,
    publish_terminal_account,
)

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
TICK = TickData(bid=2010.0, ask=2010.3, last=2010.1, time=NOW)

BLOCKED = [
    pytest.param(False, True, id="disabled_dry_run"),
    pytest.param(True, True, id="enabled_dry_run"),
    pytest.param(False, False, id="disabled_live"),
]

PATHS = [
    "entry",
    "break_even",
    "trailing",
    "partial_close",
    "full_close",
    "expired_pending",
    "pending_placement",
]


def _spec() -> SymbolSpec:
    return SymbolSpec(
        name="XAUUSD",
        point=0.01,
        digits=2,
        volume_min=0.01,
        volume_max=10.0,
        volume_step=0.01,
        stops_level=20,
        freeze_level=0,
        visible=True,
        trade_mode=4,
        contract_size=100.0,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        filling_mode=3,
    )


def _position() -> PositionInfo:
    return PositionInfo(
        ticket=501,
        symbol="XAUUSD",
        is_buy=True,
        volume=0.10,
        price_open=2000.0,
        sl=1995.0,
        tp=2030.0,
        profit=10.0,
        magic=123456789,
        comment=format_position_comment("GB", 1995.0, 0.10, 2),
        open_time=datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc),
    )


def _expired_pending() -> PendingOrderInfo:
    return PendingOrderInfo(
        ticket=900,
        symbol="XAUUSD",
        order_type=OrderType.BUY_LIMIT,
        price=1980.0,
        volume=0.1,
        sl=1970.0,
        tp=2000.0,
        magic=123456789,
        comment="",
        time_setup=datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc),
        time_expire=datetime(2026, 9, 17, 11, 0, tzinfo=timezone.utc),
    )


def _passing_state(**overrides) -> MarketState:
    base = dict(
        connected=True,
        symbol_valid=True,
        market_open=True,
        server_trading_allowed=True,
        terminal_trade_allowed=True,
        account_trade_allowed=True,
        expert_trade_allowed=True,
        spread=0.30,
        open_position_count=0,
        pending_order_count=0,
        daily_pnl=0.0,
        daily_pnl_known=True,
        account_balance=10_000.0,
        account_equity=10_000.0,
        now=NOW,
    )
    base.update(overrides)
    return MarketState(**base)


def _bot(trading_enabled: bool, dry_run: bool, **cfg_overrides) -> TradingBot:
    cfg = make_cfg(trading_enabled=trading_enabled, dry_run=dry_run, **cfg_overrides)
    bot = TradingBot(cfg)
    bot.spec = _spec()
    return bot


def _confirm_demo(bot: TradingBot, trade_mode: int = 0) -> None:
    """Stub a confirmed Demo account so a live-switch test may reach the stub.

    The proof is the broker ``account_info`` object, not a caller-set
    ``account_is_demo=True``. These tests do not describe a REAL account.
    """
    bot.conn.is_connected = lambda: True
    bot.conn.account_info = lambda: {
        "balance": 10_000.0,
        "equity": 10_000.0,
        "trade_allowed": True,
        "trade_expert": True,
        "trade_mode": trade_mode,
    }
    bot.conn.terminal_info = lambda: {"trade_allowed": True, "connected": True}
    publish_terminal_account(trade_mode)


def _assert_no_send(mt5_stub, caplog) -> None:
    assert mt5_stub.order_send.call_count == 0
    assert mt5_stub.order_check.call_count == 0
    assert "WOULD EXECUTE" in caplog.text


def _exercise_entry(bot: TradingBot, mt5_stub) -> None:
    df = make_uptrend_with_engulfing()
    bot._evaluate_entry(df, _passing_state(), TICK, [], [], 1.5)
    assert mt5_stub.order_send.call_count == 0

    spec = _spec()
    for place, sl, tp in (
        (place_market_buy, 1995.0, 2020.0),
        (place_market_sell, 2025.0, 1990.0),
    ):
        result = place(
            spec, TICK, volume=0.1, sl=sl, tp=tp, magic=77, deviation=10, comment="entry"
        )
        assert result.blocked_by_safety
        assert result.success is False
        assert result.comment.startswith("WOULD EXECUTE")

    plan = TradePlan(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        entry=TICK.ask,
        sl=1995.0,
        tp=2020.0,
        volume=0.1,
        risk_amount=50.0,
        comment="plan",
    )
    planned = send_plan(plan, spec, TICK, 77, 10)
    assert planned.blocked_by_safety


def _exercise_break_even(bot: TradingBot, mt5_stub) -> None:
    pos = _position()
    actions = manage_break_even([pos], TICK.bid, TICK.ask, bot.cfg, bot.spec)
    assert actions, "break-even must be due — otherwise the test does not prove the gate"
    bot._apply_actions(actions, [pos], [], tick=TICK)


def _exercise_trailing(bot: TradingBot, mt5_stub) -> None:
    pos = _position()
    actions = manage_trailing_stop(
        [pos], TICK.bid, TICK.ask, 2.0, bot.cfg, bot.spec
    )
    assert actions, "trailing must be due — otherwise the test does not prove the gate"
    bot._apply_actions(actions, [pos], [], tick=TICK)


def _exercise_partial(bot: TradingBot, mt5_stub) -> None:
    pos = _position()
    actions = manage_partial_close([pos], TICK.bid, TICK.ask, bot.cfg, bot.spec)
    assert actions and actions[0].kind == "partial_close"
    bot._apply_actions(actions, [pos], [], tick=TICK)


def _exercise_full(bot: TradingBot, mt5_stub) -> None:
    pos = _position()
    actions = manage_partial_close([pos], TICK.bid, TICK.ask, bot.cfg, bot.spec)
    assert actions and actions[0].kind == "full_close"
    bot._apply_actions(actions, [pos], [], tick=TICK)


def _exercise_expired(bot: TradingBot, mt5_stub) -> None:
    pending = _expired_pending()
    actions = plan_cleanup([pending], NOW)
    assert actions and actions[0].kind == "delete_order"
    bot._apply_actions(actions, [], [pending], tick=TICK)
    direct = delete_order(pending.ticket)
    assert direct.blocked_by_safety


def _exercise_pending(bot: TradingBot, mt5_stub) -> None:
    spec = _spec()
    calls = (
        (place_buy_limit, 1990.0),
        (place_buy_stop, 2020.0),
        (place_sell_limit, 2020.0),
        (place_sell_stop, 1990.0),
    )
    for place, price in calls:
        result = place(
            spec, TICK, volume=0.1, price=price, sl=1980.0, tp=2030.0,
            magic=77, deviation=10, comment="pending",
        )
        assert result.blocked_by_safety, place.__name__
    plan = TradePlan(
        symbol="XAUUSD",
        order_type=OrderType.BUY_LIMIT,
        entry=1990.0,
        sl=1980.0,
        tp=2030.0,
        volume=0.1,
        risk_amount=50.0,
        comment="pending-plan",
    )
    planned = send_plan(plan, spec, TICK, 77, 10)
    assert planned.blocked_by_safety


_EXERCISERS = {
    "entry": _exercise_entry,
    "break_even": _exercise_break_even,
    "trailing": _exercise_trailing,
    "partial_close": _exercise_partial,
    "full_close": _exercise_full,
    "expired_pending": _exercise_expired,
    "pending_placement": _exercise_pending,
}

_PATH_CFG = {
    "entry": {},
    "break_even": {
        "break_even_enabled": True,
        "partial_close_enabled": False,
        "trailing_stop_enabled": False,
    },
    "trailing": {
        "break_even_enabled": False,
        "partial_close_enabled": False,
        "trailing_stop_enabled": True,
        "trailing_atr_multiplier": 1.0,
    },
    "partial_close": {
        "break_even_enabled": False,
        "trailing_stop_enabled": False,
        "partial_close_enabled": True,
        "partial_close_levels": ((1.0, 0.5),),
    },
    "full_close": {
        "break_even_enabled": False,
        "trailing_stop_enabled": False,
        "partial_close_enabled": True,
        "partial_close_levels": ((1.0, 1.0),),
    },
    "expired_pending": {},
    "pending_placement": {},
}


@pytest.mark.parametrize("trading_enabled,dry_run", BLOCKED)
@pytest.mark.parametrize("path", PATHS)
def test_closed_gate_never_calls_order_send(trading_enabled, dry_run, path, mt5_stub, caplog):
    caplog.set_level(logging.INFO)
    bot = _bot(trading_enabled, dry_run, **_PATH_CFG[path])
    assert get_execution_permission() == ExecutionPermission(
        trading_enabled=trading_enabled, dry_run=dry_run
    )
    assert not get_execution_permission().allows_order_send
    _EXERCISERS[path](bot, mt5_stub)
    _assert_no_send(mt5_stub, caplog)


def test_default_permission_denies_without_any_install():
    permission = ExecutionPermission()
    assert permission.trading_enabled is False
    assert permission.dry_run is True
    assert permission.allows_order_send is False
    assert ExecutionPermission(trading_enabled=True, dry_run=True).allows_order_send is False
    assert ExecutionPermission(trading_enabled=False, dry_run=False).allows_order_send is False
    assert ExecutionPermission(trading_enabled=True, dry_run=False).allows_order_send is False
    assert ExecutionPermission(
        trading_enabled=True, dry_run=False, account_is_demo=False
    ).allows_order_send is False
    # The field is not a grant. Only a broker proof opens the latch.
    assert ExecutionPermission(
        trading_enabled=True, dry_run=False, account_is_demo=True
    ).allows_order_send is False


def test_env_flags_do_not_open_the_gate(monkeypatch, mt5_stub, caplog):
    """orders.py must not read TRADING_ENABLED / DRY_RUN from the environment."""
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    result = place_market_buy(
        _spec(), TICK, volume=0.1, sl=1995.0, tp=2020.0, magic=77, deviation=10, comment="env"
    )
    assert result.blocked_by_safety
    assert mt5_stub.order_send.call_count == 0
    assert "WOULD EXECUTE" in caplog.text


def test_bot_uses_injected_config_not_ambient_env(monkeypatch, mt5_stub, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    bot = _bot(False, True, break_even_enabled=True, partial_close_enabled=False, trailing_stop_enabled=False)
    _exercise_break_even(bot, mt5_stub)
    _assert_no_send(mt5_stub, caplog)


def test_explicit_permission_cannot_widen_a_closed_gate(mt5_stub, caplog):
    caplog.set_level(logging.INFO)
    bot = _bot(False, True)
    assert not get_execution_permission().allows_order_send
    result = send_request(
        {"action": 1, "symbol": "XAUUSD", "volume": 0.1},
        permission=ExecutionPermission(trading_enabled=True, dry_run=False),
    )
    assert result.blocked_by_safety
    assert mt5_stub.order_send.call_count == 0
    # The bot's own management path also reinstalls its config over a widened context.
    pos = _position()
    actions = manage_break_even(
        [pos], TICK.bid, TICK.ask,
        make_cfg(break_even_enabled=True, partial_close_enabled=False, trailing_stop_enabled=False),
        bot.spec,
    )
    assert actions
    with execution_permission(ExecutionPermission(trading_enabled=True, dry_run=False)):
        bot._apply_actions(actions, [pos], [], tick=TICK)
    assert mt5_stub.order_send.call_count == 0
    assert "WOULD EXECUTE" in caplog.text


def test_from_env_installs_the_permission_explicitly(monkeypatch, mt5_stub):
    """The application, not orders.py, is what reads the switches."""
    monkeypatch.setenv("TRADING_ENABLED", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    cfg = Config.from_env()
    assert cfg.trading_enabled is True
    assert cfg.dry_run is False
    bot = TradingBot(cfg)
    # Switches come from the application. Without a confirmed Demo account
    # the gate stays closed — orders.py still does not read the environment.
    installed = get_execution_permission()
    assert installed.trading_enabled is True
    assert installed.dry_run is False
    assert installed.account_is_demo is False
    assert installed.allows_order_send is False
    result = place_market_buy(
        _spec(), TICK, volume=0.1, sl=1995.0, tp=2020.0, magic=77, deviation=10, comment="wired"
    )
    assert result.blocked_by_safety
    assert mt5_stub.order_send.call_count == 0
    assert mt5_stub.order_check.call_count == 0
    _confirm_demo(bot)
    bot._install_execution_gate()
    assert get_execution_permission().allows_order_send is True
    sent = place_market_buy(
        _spec(), TICK, volume=0.1, sl=1995.0, tp=2020.0, magic=77, deviation=10, comment="wired"
    )
    assert sent.success is True
    assert mt5_stub.order_send.call_count == 1
    assert bot.cfg.trading_enabled and not bot.cfg.dry_run


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_live_entry_reaches_order_send_only_when_risk_gates_pass(side, mt5_stub):
    bot = _bot(True, False)
    _confirm_demo(bot)
    df = make_uptrend_with_engulfing() if side == "BUY" else make_downtrend_with_engulfing()
    bot._evaluate_entry(df, _passing_state(), TICK, [], [], 1.5)
    assert mt5_stub.order_send.call_count == 1
    assert mt5_stub.order_check.call_count == 1
    request = mt5_stub.order_send.call_args[0][0]
    assert request["symbol"] == "XAUUSD"
    assert request["volume"] > 0

    mt5_stub.order_send.reset_mock()
    mt5_stub.order_check.reset_mock()
    blocked = _passing_state(spread=5.0)
    bot._evaluate_entry(df, blocked, TICK, [], [], 1.5)
    assert mt5_stub.order_send.call_count == 0


def test_live_management_and_pending_reach_the_stub(mt5_stub):
    """The gate opens for the stub only when both switches allow live sends."""
    be_bot = _bot(
        True, False,
        break_even_enabled=True,
        partial_close_enabled=False,
        trailing_stop_enabled=False,
    )
    _confirm_demo(be_bot)
    pos = _position()
    be = manage_break_even([pos], TICK.bid, TICK.ask, be_bot.cfg, be_bot.spec)
    assert be
    be_bot._apply_actions(be, [pos], [], tick=TICK)
    assert mt5_stub.order_send.call_count == 1

    mt5_stub.order_send.reset_mock()
    trail_bot = _bot(
        True, False,
        break_even_enabled=False,
        partial_close_enabled=False,
        trailing_stop_enabled=True,
        trailing_atr_multiplier=1.0,
    )
    _confirm_demo(trail_bot)
    trail = manage_trailing_stop([pos], TICK.bid, TICK.ask, 2.0, trail_bot.cfg, trail_bot.spec)
    assert trail
    trail_bot._apply_actions(trail, [pos], [], tick=TICK)
    assert mt5_stub.order_send.call_count == 1

    mt5_stub.order_send.reset_mock()
    partial_bot = _bot(
        True, False,
        break_even_enabled=False,
        trailing_stop_enabled=False,
        partial_close_enabled=True,
        partial_close_levels=((1.0, 0.5),),
    )
    _confirm_demo(partial_bot)
    partial = manage_partial_close([pos], TICK.bid, TICK.ask, partial_bot.cfg, partial_bot.spec)
    assert partial and partial[0].kind == "partial_close"
    partial_bot._apply_actions(partial, [pos], [], tick=TICK)
    assert mt5_stub.order_send.call_count == 1

    mt5_stub.order_send.reset_mock()
    full_bot = _bot(
        True, False,
        break_even_enabled=False,
        trailing_stop_enabled=False,
        partial_close_enabled=True,
        partial_close_levels=((1.0, 1.0),),
    )
    _confirm_demo(full_bot)
    full = manage_partial_close([pos], TICK.bid, TICK.ask, full_bot.cfg, full_bot.spec)
    assert full and full[0].kind == "full_close"
    full_bot._apply_actions(full, [pos], [], tick=TICK)
    assert mt5_stub.order_send.call_count == 1

    mt5_stub.order_send.reset_mock()
    mt5_stub.order_check.reset_mock()
    pending = _expired_pending()
    cleanup = plan_cleanup([pending], NOW)
    assert cleanup
    be_bot._apply_actions(cleanup, [], [pending], tick=TICK)
    assert mt5_stub.order_send.call_count == 1
    # TRADE_ACTION_REMOVE skips order_check but must still be gated by permission.
    assert mt5_stub.order_check.call_count == 0

    mt5_stub.order_send.reset_mock()
    placed = place_buy_limit(
        _spec(), TICK, volume=0.1, price=1990.0, sl=1980.0, tp=2030.0,
        magic=77, deviation=10, comment="live-pending",
    )
    assert placed.success is True
    assert mt5_stub.order_send.call_count == 1


def test_cycle_safe_mode_management_does_not_send(mt5_stub, caplog):
    """Audit reproduction: owned position due for break-even + expired pending."""
    caplog.set_level(logging.INFO)
    bot = _bot(
        False, True,
        break_even_enabled=True,
        partial_close_enabled=False,
        trailing_stop_enabled=False,
    )
    bot.conn.is_connected = lambda: True
    bot.conn.account_info = lambda: {
        "balance": 10_000.0, "equity": 10_000.0,
        "trade_allowed": True, "trade_expert": True, "trade_mode": 0,
    }
    bot.conn.terminal_info = lambda: {"trade_allowed": True, "connected": True}
    pos = _position()
    pending = _expired_pending()
    assert manage_break_even([pos], TICK.bid, TICK.ask, bot.cfg, bot.spec)
    assert plan_cleanup([pending], NOW)
    df = make_uptrend_with_engulfing()

    with (
        patch("gold_trader.main.mt5_positions.get_positions", return_value=[pos]),
        patch("gold_trader.main.mt5_orders.get_pending_orders", return_value=[pending]),
        patch("gold_trader.main.market_data.get_tick", return_value=TICK),
        patch("gold_trader.main.market_data.get_candles", return_value=df),
    ):
        bot.cycle()

    assert mt5_stub.order_send.call_count == 0
    assert "WOULD EXECUTE" in caplog.text
    assert "break-even" in caplog.text


def test_live_cycle_management_reaches_stub_when_gates_allow(mt5_stub):
    bot = _bot(
        True, False,
        break_even_enabled=True,
        partial_close_enabled=False,
        trailing_stop_enabled=False,
    )
    _confirm_demo(bot)
    pos = _position()
    pending = _expired_pending()
    df = make_uptrend_with_engulfing()
    with (
        patch("gold_trader.main.mt5_positions.get_positions", return_value=[pos]),
        patch("gold_trader.main.mt5_orders.get_pending_orders", return_value=[pending]),
        patch("gold_trader.main.market_data.get_tick", return_value=TICK),
        patch("gold_trader.main.market_data.get_candles", return_value=df),
    ):
        bot.cycle()
    # Break-even + expired delete. Entry is blocked by the open position.
    assert mt5_stub.order_send.call_count == 2


def test_order_send_call_sites_are_only_inside_send_request():
    """No production bypass around the central gate."""
    root = Path(__file__).resolve().parents[1] / "gold_trader"
    call_sites = []
    for path in root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if ".order_send(" in line or stripped.startswith("order_send("):
                call_sites.append((path.name, lineno, stripped))
    assert call_sites, "the guarded order_send call must still exist"
    assert {name for name, _, _ in call_sites} == {"orders.py"}
    orders = (root / "mt5" / "orders.py").read_text(encoding="utf-8")
    body = orders.split("def send_request", 1)[1].split("\ndef ", 1)[0]
    code = body.split('"""', 2)[-1]
    assert "resolve_execution_permission" in code
    assert "WOULD EXECUTE" in code
    assert code.index("order_send(") > code.index("resolve_execution_permission")
    assert code.index("demote_verified_demo_if_stale") < code.index("order_check(")
    assert code.index("order_check(") < code.index("order_send(")
