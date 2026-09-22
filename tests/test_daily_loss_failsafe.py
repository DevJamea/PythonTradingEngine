"""Regression: unknown daily P/L must block entries regardless of floating P/L.

The previous fail-safe used ``daily_pnl = -max_daily_loss - 1 + floating``.
With ``max_daily_loss=500`` and ``floating=+600`` that became ``+99`` and the
numeric gate (``daily_pnl <= -500``) passed. The gate now uses an explicit
``daily_pnl_known`` flag and ignores the number when history is unavailable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from gold_trader.main import TradingBot
from gold_trader.models import OrderType, PositionInfo, SymbolSpec, TradePlan
from gold_trader.mt5.connection import MT5Error
from gold_trader.mt5.market_data import TickData
from gold_trader.risk.risk_manager import check_trade
from tests._helpers import make_cfg

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
TICK = TickData(bid=2000.0, ask=2000.3, last=2000.1, time=NOW)
MAGIC = 123456789


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


def _plan() -> TradePlan:
    return TradePlan(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        entry=2000.0,
        sl=1995.0,
        tp=2010.0,
        volume=0.1,
        risk_amount=50.0,
    )


def _position(profit: float) -> PositionInfo:
    return PositionInfo(
        ticket=1,
        symbol="XAUUSD",
        is_buy=True,
        volume=0.1,
        price_open=2000.0,
        sl=1990.0,
        tp=2020.0,
        profit=profit,
        magic=MAGIC,
        comment="",
        open_time=NOW,
    )


class _Deal:
    def __init__(self, profit: float, commission: float, swap: float, magic: int) -> None:
        self.profit = profit
        self.commission = commission
        self.swap = swap
        self.magic = magic


def _bot(cfg):
    bot = TradingBot(cfg)
    bot.spec = _spec()
    bot.conn.is_connected = lambda: True
    bot.conn.account_info = lambda: {
        "balance": 10_000.0,
        "equity": 10_000.0,
        "trade_allowed": True,
        "trade_expert": True,
        "trade_mode": 0,
    }
    bot.conn.terminal_info = lambda: {"trade_allowed": True, "connected": True}
    return bot


def _build(bot, mt5_stub, positions):
    with patch("gold_trader.main.market_data.get_tick", return_value=TICK):
        return bot._build_market_state(positions, [])


@pytest.mark.parametrize("floating", [0.0, 600.0, -600.0])
def test_unknown_history_blocks_regardless_of_floating(floating, mt5_stub):
    cfg = make_cfg(trading_enabled=True, dry_run=False, max_daily_loss=500.0)
    bot = _bot(cfg)
    mt5_stub.history_deals_get.return_value = None
    mt5_stub.last_error.return_value = (1, "history unavailable")

    state = _build(bot, mt5_stub, [_position(floating)])
    decision = check_trade(_plan(), state, cfg, _spec())

    # The old sentinel ``-501 + floating`` passed the numeric gate at +600.
    old_sentinel = -abs(cfg.max_daily_loss) - 1.0 + floating
    assert state.daily_pnl_known is False
    assert state.daily_pnl == 0.0
    assert state.daily_pnl != pytest.approx(old_sentinel)
    assert not decision.allowed
    assert not decision.would_trade
    assert any("history unavailable" in reason for reason in decision.gate_failures)
    assert not any("daily loss limit" in reason for reason in decision.gate_failures)


def test_history_exception_is_also_unknown(mt5_stub):
    cfg = make_cfg(trading_enabled=True, dry_run=False, max_daily_loss=500.0)
    bot = _bot(cfg)
    mt5_stub.history_deals_get.side_effect = MT5Error("history_deals_get exploded")
    state = _build(bot, mt5_stub, [_position(600.0)])
    decision = check_trade(_plan(), state, cfg, _spec())
    assert state.daily_pnl_known is False
    assert not decision.allowed
    assert any("history unavailable" in reason for reason in decision.gate_failures)


def test_positive_floating_cannot_clear_unknown_history_gate():
    """Even a hand-built +99 (the old bug's number) is blocked when unknown."""
    cfg = make_cfg(trading_enabled=True, dry_run=False, max_daily_loss=500.0)
    from tests.test_risk import make_state

    state = make_state(daily_pnl=99.0, daily_pnl_known=False)
    decision = check_trade(_plan(), state, cfg, _spec())
    assert not decision.allowed
    assert any("history unavailable" in reason for reason in decision.gate_failures)


def test_known_history_still_adds_floating_and_uses_numeric_gate(mt5_stub):
    cfg = make_cfg(
        trading_enabled=True, dry_run=False, max_daily_loss=500.0, max_open_positions=5,
    )
    bot = _bot(cfg)
    # Closed -100 + commission -1.5, plus a foreign-magic deal that must be ignored.
    mt5_stub.history_deals_get.return_value = [
        _Deal(profit=-100.0, commission=-1.0, swap=-0.5, magic=MAGIC),
        _Deal(profit=-999.0, commission=0.0, swap=0.0, magic=1),
    ]
    state = _build(bot, mt5_stub, [_position(50.0)])
    assert state.daily_pnl_known is True
    assert state.daily_pnl == pytest.approx(-51.5)
    decision = check_trade(_plan(), state, cfg, _spec())
    assert decision.allowed
    assert decision.would_trade
    assert decision.gate_failures == []


def test_known_history_still_blocks_when_closed_plus_floating_hits_limit(mt5_stub):
    cfg = make_cfg(trading_enabled=True, dry_run=False, max_daily_loss=500.0)
    bot = _bot(cfg)
    mt5_stub.history_deals_get.return_value = [
        _Deal(profit=-400.0, commission=0.0, swap=0.0, magic=MAGIC),
    ]
    state = _build(bot, mt5_stub, [_position(-100.0)])
    assert state.daily_pnl_known is True
    assert state.daily_pnl == pytest.approx(-500.0)
    decision = check_trade(_plan(), state, cfg, _spec())
    assert not decision.allowed
    assert any("daily loss limit" in reason for reason in decision.gate_failures)
    assert not any("history unavailable" in reason for reason in decision.gate_failures)


def test_known_zero_history_and_flat_float_still_allows(mt5_stub):
    cfg = make_cfg(trading_enabled=True, dry_run=False, max_daily_loss=500.0)
    bot = _bot(cfg)
    mt5_stub.history_deals_get.return_value = []
    state = _build(bot, mt5_stub, [])
    assert state.daily_pnl_known is True
    assert state.daily_pnl == pytest.approx(0.0)
    decision = check_trade(_plan(), state, cfg, _spec())
    assert decision.allowed


def test_known_positive_pnl_is_not_blocked_by_daily_loss():
    cfg = make_cfg(trading_enabled=True, dry_run=False, max_daily_loss=500.0)
    from tests.test_risk import make_state

    state = make_state(daily_pnl=99.0, daily_pnl_known=True)
    decision = check_trade(_plan(), state, cfg, _spec())
    assert decision.allowed
    assert not any("daily" in reason for reason in decision.gate_failures)
