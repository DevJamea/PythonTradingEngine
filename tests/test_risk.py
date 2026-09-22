"""Unit tests for the risk gate and duplicate protection."""
from __future__ import annotations

from datetime import datetime, timezone

from gold_trader.models import (
    OrderType,
    PendingOrderInfo,
    PositionInfo,
    SymbolSpec,
    TradePlan,
)
from gold_trader.risk.risk_manager import check_duplicate_entry, check_trade
from tests._helpers import make_cfg

MAGIC = 99


def make_spec() -> SymbolSpec:
    return SymbolSpec(
        name="XAUUSD",
        point=0.01,
        digits=2,
        volume_min=0.01,
        volume_max=10.0,
        volume_step=0.01,
        stops_level=20,  # min distance 0.20
        freeze_level=10,
        visible=True,
        trade_mode=1,
        contract_size=100.0,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        filling_mode=3,
    )


def make_state(**overrides):
    from gold_trader.models import MarketState

    base = dict(
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
        daily_pnl_known=True,
        account_balance=10_000.0,
        account_equity=10_000.0,
        now=datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return MarketState(**base)


def make_plan(**overrides) -> TradePlan:
    base = dict(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        entry=2000.0,
        sl=1995.0,
        tp=2010.0,
        volume=0.1,
        risk_amount=50.0,
    )
    base.update(overrides)
    return TradePlan(**base)


SPEC = make_spec()


def check(cfg, state, plan, positions=(), pendings=()):
    return check_trade(plan, state, cfg, SPEC, positions=positions, pendings=pendings)


def test_all_gates_pass():
    decision = check(make_cfg(trading_enabled=True, dry_run=False), make_state(), make_plan())
    assert decision.allowed
    assert decision.would_trade
    assert decision.reasons == []
    assert decision.gate_failures == []


def test_trading_disabled_blocks():
    cfg = make_cfg(trading_enabled=False, dry_run=False)
    decision = check(cfg, make_state(), make_plan())
    assert not decision.allowed
    assert decision.would_trade
    assert "TRADING_ENABLED=false" in decision.reasons


def test_dry_run_blocks_but_would_trade():
    cfg = make_cfg(trading_enabled=True, dry_run=True)
    decision = check(cfg, make_state(), make_plan())
    assert not decision.allowed
    assert decision.would_trade
    assert "DRY_RUN=true" in decision.reasons


def test_spread_too_wide():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(spread=0.9),
        make_plan(),
    )
    assert not decision.allowed
    assert not decision.would_trade
    assert any("spread" in f for f in decision.gate_failures)


def test_max_open_positions():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(open_position_count=1),
        make_plan(),
    )
    assert not decision.allowed
    assert any("open positions" in f for f in decision.gate_failures)


def test_max_pending_orders():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(pending_order_count=2),
        make_plan(),
    )
    assert not decision.allowed
    assert any("pending orders" in f for f in decision.gate_failures)


def test_daily_loss_limit():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(daily_pnl=-500.0),
        make_plan(),
    )
    assert not decision.allowed
    assert any("daily loss" in f for f in decision.gate_failures)


def test_outside_trading_hours():
    cfg = make_cfg(
        trading_enabled=True, dry_run=False,
        trading_start_time="01:00", trading_end_time="20:00",
    )
    blocked = check(
        cfg, make_state(now=datetime(2026, 9, 17, 23, 30, tzinfo=timezone.utc)), make_plan()
    )
    assert not blocked.allowed
    assert any("outside trading hours" in f for f in blocked.gate_failures)

    allowed = check(
        cfg, make_state(now=datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)), make_plan()
    )
    assert allowed.allowed


def test_trading_hours_crossing_midnight():
    cfg = make_cfg(
        trading_enabled=True, dry_run=False,
        trading_start_time="20:00", trading_end_time="02:00",
    )
    night = check(
        cfg, make_state(now=datetime(2026, 9, 17, 23, 0, tzinfo=timezone.utc)), make_plan()
    )
    assert night.allowed
    day = check(
        cfg, make_state(now=datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)), make_plan()
    )
    assert not day.allowed
    assert any("outside trading hours" in f for f in day.gate_failures)


def test_market_closed():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(market_open=False),
        make_plan(),
    )
    assert not decision.allowed


def test_no_connection():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(connected=False),
        make_plan(),
    )
    assert not decision.allowed


def test_sl_on_wrong_side():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(),
        make_plan(sl=2005.0),  # SL above entry for a BUY
    )
    assert not decision.allowed
    assert any("below entry" in f for f in decision.gate_failures)


def test_sl_too_close_to_entry():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(),
        make_plan(sl=1999.9),  # distance 0.1 < stops level 0.2
    )
    assert not decision.allowed
    assert any("too close" in f for f in decision.gate_failures)


def test_sizing_failure_blocks():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(),
        make_plan(volume=None, risk_amount=0.0),
    )
    assert not decision.allowed
    assert any("sizing" in f for f in decision.gate_failures)


def _position(is_buy=True, ticket=1):
    return PositionInfo(
        ticket=ticket,
        symbol="XAUUSD",
        is_buy=is_buy,
        volume=0.1,
        price_open=2000.0,
        sl=1995.0,
        tp=2010.0,
        profit=0.0,
        magic=MAGIC,
        comment="GB|sl=1995.00|vol=0.10",
    )


def _pending(order_type=OrderType.BUY_STOP, price=2000.0, ticket=2):
    from datetime import datetime as dt, timezone as tz

    return PendingOrderInfo(
        ticket=ticket,
        symbol="XAUUSD",
        order_type=order_type,
        price=price,
        volume=0.1,
        sl=0.0,
        tp=0.0,
        magic=MAGIC,
        comment="",
        time_setup=dt(2026, 9, 17, 11, 0, tzinfo=tz.utc),
    )


def test_duplicate_open_position_same_side():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False),
        make_state(open_position_count=1),
        make_plan(),
        positions=[_position(is_buy=True)],
    )
    assert not decision.allowed
    assert any("duplicate" in f for f in decision.gate_failures)


def test_opposite_position_is_not_a_duplicate():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False, max_open_positions=5),
        make_state(open_position_count=1),
        make_plan(),
        positions=[_position(is_buy=False)],
    )
    assert decision.allowed


def test_duplicate_pending_order_same_price():
    decision = check(
        make_cfg(trading_enabled=True, dry_run=False, max_pending_orders=5),
        make_state(pending_order_count=1),
        make_plan(order_type=OrderType.BUY_STOP, entry=2000.01),
        pendings=[_pending(order_type=OrderType.BUY_STOP, price=2000.0)],
    )
    assert not decision.allowed
    assert any("pending order already" in f for f in decision.gate_failures)


def test_check_duplicate_entry_directly():
    plan = make_plan()
    reason = check_duplicate_entry(plan, [_position(is_buy=True)], [], SPEC.point)
    assert reason is not None
    assert check_duplicate_entry(plan, [], [], SPEC.point) is None


# ---------------------------------------------------------------------------
# BUG 1 & BUG 6 regression tests
# ---------------------------------------------------------------------------

def test_permission_combinations():
    live_cfg = make_cfg(trading_enabled=True, dry_run=False)

    # 1. terminal allowed = false, account allowed = true, expert allowed = true -> BLOCK
    state_term_blocked = make_state(
        terminal_trade_allowed=False,
        account_trade_allowed=True,
        expert_trade_allowed=True,
    )
    res1 = check(live_cfg, state_term_blocked, make_plan())
    assert not res1.allowed
    assert any("terminal trading disabled" in f for f in res1.gate_failures)

    # 2. terminal allowed = true, account allowed = false, expert allowed = true -> BLOCK
    state_acc_blocked = make_state(
        terminal_trade_allowed=True,
        account_trade_allowed=False,
        expert_trade_allowed=True,
    )
    res2 = check(live_cfg, state_acc_blocked, make_plan())
    assert not res2.allowed
    assert any("account trading disabled" in f for f in res2.gate_failures)

    # 3. terminal allowed = true, account allowed = true, expert allowed = false -> BLOCK
    state_exp_blocked = make_state(
        terminal_trade_allowed=True,
        account_trade_allowed=True,
        expert_trade_allowed=False,
    )
    res3 = check(live_cfg, state_exp_blocked, make_plan())
    assert not res3.allowed
    assert any("expert trading disabled" in f for f in res3.gate_failures)

    # 4. all true -> allowed to continue
    state_all_allowed = make_state(
        terminal_trade_allowed=True,
        account_trade_allowed=True,
        expert_trade_allowed=True,
    )
    res4 = check(live_cfg, state_all_allowed, make_plan())
    assert res4.allowed
    assert len(res4.gate_failures) == 0


def test_market_usable_tick_conditions():
    from datetime import datetime, timezone
    from gold_trader.mt5.market_data import TickData, is_tick_usable

    # 1. bid > 0, ask > 0, last = 0 -> market considered usable
    tick_last_zero = TickData(
        bid=2000.50, ask=2000.80, last=0.0, time=datetime.now(timezone.utc)
    )
    assert is_tick_usable(tick_last_zero) is True
    assert tick_last_zero.is_usable is True

    # 2. bid = 0, ask = 0 -> market unavailable
    tick_zero = TickData(
        bid=0.0, ask=0.0, last=0.0, time=datetime.now(timezone.utc)
    )
    assert is_tick_usable(tick_zero) is False
    assert tick_zero.is_usable is False

    # 3. missing/invalid tick -> market unavailable
    assert is_tick_usable(None) is False
    assert is_tick_usable("invalid") is False

    # Inverted spread (ask < bid) -> unavailable
    tick_inverted = TickData(
        bid=2001.0, ask=2000.0, last=2000.5, time=datetime.now(timezone.utc)
    )
    assert is_tick_usable(tick_inverted) is False


def test_market_state_permission_defaults_are_fail_safe():
    """MarketState() with no permission args must be restrictive (fail-safe)."""
    from gold_trader.models import MarketState

    state = MarketState(
        connected=True,
        symbol_valid=True,
        market_open=True,
        server_trading_allowed=True,
        spread=0.25,
        open_position_count=0,
        pending_order_count=0,
        daily_pnl=0.0,
        account_balance=10_000.0,
        account_equity=10_000.0,
        now=datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc),
    )
    # Defaults must be False (fail-safe)
    assert state.terminal_trade_allowed is False
    assert state.account_trade_allowed is False
    assert state.expert_trade_allowed is False
    assert state.daily_pnl_known is False

    # Trading must be blocked when permissions default to False
    cfg = make_cfg(trading_enabled=True, dry_run=False)
    decision = check(cfg, state, make_plan())
    assert not decision.allowed
    # At least one permission failure should be reported
    assert any("trading disabled" in f for f in decision.gate_failures)


def test_market_state_all_permissions_true_allows_trading():
    """Explicit True permissions must allow trading when other gates pass."""
    from gold_trader.models import MarketState

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
        account_balance=10_000.0,
        account_equity=10_000.0,
        now=datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc),
        daily_pnl_known=True,
    )
    cfg = make_cfg(trading_enabled=True, dry_run=False)
    decision = check(cfg, state, make_plan())
    assert decision.allowed
    assert decision.would_trade
    assert decision.gate_failures == []

