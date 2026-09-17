"""Shared domain types (no MetaTrader 5 dependency).

These dataclasses keep the strategy, risk, backtest and management layers
independent from the MT5 integration, which makes them unit-testable on
any platform and keeps the architecture open for future agents/modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class Signal(str, Enum):
    """Decision produced by the signal engine."""

    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class Trend(str, Enum):
    """Trend classification produced by the trend engine."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"


class OrderType(str, Enum):
    """Supported MT5 order types."""

    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_LIMIT = "SELL_LIMIT"
    SELL_STOP = "SELL_STOP"

    @property
    def is_buy_side(self) -> bool:
        """True for all buy-side orders (market, limit, stop)."""
        return self in (OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP)


class AccountMode(str, Enum):
    """MT5 account position accounting mode (NETTING vs HEDGING)."""

    NETTING = "NETTING"
    HEDGING = "HEDGING"


# ---------------------------------------------------------------------------
# strategy results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeSignal:
    """Result of the signal engine: direction + machine-readable detail."""

    signal: Signal
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# market / symbol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolSpec:
    """Trading-relevant subset of an MT5 symbol description."""

    name: str
    point: float
    digits: int
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int
    freeze_level: int
    visible: bool
    trade_mode: int
    contract_size: float
    trade_tick_size: float
    trade_tick_value: float
    filling_mode: int

    def min_stop_distance(self) -> float:
        """Broker minimum distance between price and SL/TP."""
        return self.stops_level * self.point


def default_gold_spec() -> SymbolSpec:
    """Neutral gold-like spec for backtests/tests without a broker.

    1 lot = 100 oz; tick size 0.01 with tick value 1.0, i.e. $100 profit
    per $1 price move per lot. This is a computation default used when no
    broker is connected -- NOT a statement about any specific broker.
    """
    return SymbolSpec(
        name="XAUUSD",
        point=0.01,
        digits=2,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level=20,
        freeze_level=10,
        visible=True,
        # SYMBOL_TRADE_MODE_FULL = 4 (0=DISABLED,1=LONGONLY,2=SHORTONLY,3=CLOSEONLY,4=FULL)
        trade_mode=4,
        contract_size=100.0,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
        filling_mode=3,
    )


# ---------------------------------------------------------------------------
# positions / orders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionInfo:
    """A single open position (bot positions only, magic-filtered)."""

    ticket: int
    symbol: str
    is_buy: bool
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    magic: int
    comment: str
    open_time: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class PendingOrderInfo:
    """A single pending order (bot orders only, magic-filtered)."""

    ticket: int
    symbol: str
    order_type: OrderType
    price: float
    volume: float
    sl: float
    tp: float
    magic: int
    comment: str
    time_setup: Optional[pd.Timestamp] = None
    time_expire: Optional[pd.Timestamp] = None


@dataclass(frozen=True)
class TradePlan:
    """A fully priced, sized and validated trade ready for execution."""

    symbol: str
    order_type: OrderType
    entry: float
    sl: float
    tp: float
    volume: Optional[float]
    risk_amount: float
    comment: str = ""


@dataclass
class OrderResult:
    """Outcome of an ``order_send`` call (or a local rejection)."""

    success: bool
    retcode: int
    comment: str
    order: int
    request: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line, log-safe summary (the request never contains secrets)."""
        r = self.request
        return (
            f"retcode={self.retcode} | comment={self.comment!r} | order={self.order} | "
            f"symbol={r.get('symbol')} | type={r.get('type')} | volume={r.get('volume')} | "
            f"price={r.get('price')} | sl={r.get('sl')} | tp={r.get('tp')}"
        )


# ---------------------------------------------------------------------------
# risk / management
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketState:
    """Snapshot of everything the risk gate needs to know (pure data)."""

    connected: bool
    symbol_valid: bool
    market_open: bool
    server_trading_allowed: bool
    spread: float
    open_position_count: int
    pending_order_count: int
    daily_pnl: float
    account_balance: float
    account_equity: float
    now: datetime


@dataclass
class RiskDecision:
    """Result of the risk gate.

    ``allowed``       -- a real order may be sent.
    ``would_trade``   -- all structural gates passed (dry-run output may be
                         printed even when ``allowed`` is False).
    ``gate_failures`` -- failed structural checks (spread, limits, ...).
    ``reasons``       -- everything that blocks real execution right now.
    """

    allowed: bool
    would_trade: bool
    reasons: List[str] = field(default_factory=list)
    gate_failures: List[str] = field(default_factory=list)


@dataclass
class ManagementAction:
    """An action to apply to an existing bot order/position.

    ``kind`` is one of: ``move_sl``, ``partial_close``, ``full_close``,
    ``delete_order``.
    """

    kind: str
    ticket: int
    description: str
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    close_volume: Optional[float] = None
