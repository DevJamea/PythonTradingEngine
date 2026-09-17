"""Independent risk gate.

Pure logic: (TradePlan, MarketState, Config, SymbolSpec, positions,
pending orders) -> RiskDecision. This layer knows nothing about strategy
and touches no MT5 API, which keeps it unit-testable and swappable.

It blocks real execution when ANY of the following holds:
TRADING_ENABLED false, DRY_RUN true, no connection, invalid symbol,
market closed, server trading disabled, spread too wide, max open
positions reached, max pending orders reached, daily loss limit hit,
outside trading hours, sizing failed, invalid SL/TP, duplicate entry.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..config import Config
from ..models import (
    MarketState,
    PendingOrderInfo,
    PositionInfo,
    RiskDecision,
    SymbolSpec,
    TradePlan,
)
from ..utils.time_utils import is_within_trading_hours
from ..utils.validators import validate_sl_tp


def check_duplicate_entry(
    plan: TradePlan,
    positions: Sequence[PositionInfo],
    pendings: Sequence[PendingOrderInfo],
    point: float,
) -> Optional[str]:
    """Duplicate protection: same direction open position, or an existing
    same-side order at (almost) the same price. Returns a reason or None.
    """
    for position in positions:
        if position.symbol == plan.symbol and position.is_buy == plan.order_type.is_buy_side:
            return f"open position already in the same direction (ticket {position.ticket})"
    tolerance = max(point, 1e-12)
    for order in pendings:
        if (
            order.symbol == plan.symbol
            and order.order_type.is_buy_side == plan.order_type.is_buy_side
            and abs(order.price - plan.entry) <= tolerance
        ):
            return (
                f"pending order already exists at {order.price:.5f} (ticket {order.ticket})"
            )
    return None


def check_trade(
    plan: TradePlan,
    state: MarketState,
    cfg: Config,
    spec: SymbolSpec,
    positions: Sequence[PositionInfo] = (),
    pendings: Sequence[PendingOrderInfo] = (),
) -> RiskDecision:
    """Run every structural risk gate and the live-execution switches."""
    failures: List[str] = []

    if not state.connected:
        failures.append("no MT5 connection")
    if not state.symbol_valid:
        failures.append("symbol invalid or not tradable")
    if not state.market_open:
        failures.append("market closed / no tick available")
    if not state.terminal_trade_allowed or not state.server_trading_allowed:
        failures.append("terminal trading disabled")
    if not state.account_trade_allowed:
        failures.append("account trading disabled")
    if not state.expert_trade_allowed:
        failures.append("expert trading disabled")
    if state.spread > cfg.max_spread:
        failures.append(
            f"spread {state.spread:.5f} exceeds max {cfg.max_spread}"
        )
    if state.open_position_count >= cfg.max_open_positions:
        failures.append(
            f"open positions {state.open_position_count} >= max {cfg.max_open_positions}"
        )
    if state.pending_order_count >= cfg.max_pending_orders:
        failures.append(
            f"pending orders {state.pending_order_count} >= max {cfg.max_pending_orders}"
        )
    if state.daily_pnl <= -abs(cfg.max_daily_loss):
        failures.append(
            f"daily loss limit reached ({state.daily_pnl:.2f} <= -{abs(cfg.max_daily_loss):.2f})"
        )
    if not is_within_trading_hours(
        state.now, cfg.trading_start_time, cfg.trading_end_time
    ):
        failures.append(
            f"outside trading hours ({cfg.trading_start_time}-{cfg.trading_end_time} UTC)"
        )

    if plan.volume is None or plan.volume <= 0:
        failures.append("position sizing failed")
    else:
        failures.extend(
            validate_sl_tp(
                plan.entry,
                plan.sl,
                plan.tp,
                plan.order_type.is_buy_side,
                spec.point,
                spec.stops_level,
            )
        )

    duplicate = check_duplicate_entry(plan, positions, pendings, spec.point)
    if duplicate:
        failures.append(f"duplicate protection: {duplicate}")

    would_trade = not failures

    reasons: List[str] = []
    if not would_trade:
        reasons.extend(failures)
    if not cfg.trading_enabled:
        reasons.append("TRADING_ENABLED=false")
    if cfg.dry_run:
        reasons.append("DRY_RUN=true")

    allowed = (
        would_trade
        and cfg.trading_enabled
        and not cfg.dry_run
        and state.connected
    )
    return RiskDecision(
        allowed=allowed,
        would_trade=would_trade,
        reasons=reasons,
        gate_failures=failures,
    )
