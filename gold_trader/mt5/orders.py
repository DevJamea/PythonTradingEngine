"""Order building and sending via MT5 ``order_send``.

Market orders (BUY/SELL) plus all four pending order types
(BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP).

The filling mode is read from the symbol properties -- never assumed:
FOK when the symbol supports it, else IOC, else RETURN. Before placing a
pending order the distance to the current price is checked against the
broker ``stops_level``.

Every live send goes through :func:`send_request`. That function refuses
``order_send`` (and the preflight ``order_check``) unless an
:class:`~gold_trader.mt5.execution_gate.ExecutionPermission` allowing live
execution has been installed. This module does not read ``.env``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import (
    OrderResult,
    OrderType,
    PendingOrderInfo,
    SymbolSpec,
    TradePlan,
)
from ._constants import const
from .connection import MT5Error, require_mt5
from .execution_gate import ExecutionPermission, resolve_execution_permission
from .market_data import TickData

logger = logging.getLogger("gold_trader.mt5.orders")

#: MT5 retcodes that mean the request was fully/partially accepted.
#: Official ENUM_TRADE_RETCODE values (MetaTrader5 Python package):
#: TRADE_RETCODE_PLACED=10008, TRADE_RETCODE_DONE=10009,
#: TRADE_RETCODE_DONE_PARTIAL=10010.
RETCODE_PLACED = 10008
RETCODE_DONE = 10009
RETCODE_DONE_PARTIAL = 10010
#: retcode used for local rejections (nothing was sent to the server).
LOCAL_REJECTION = -1

#: Documented fallback values for the official ORDER_TYPE_* constants
#: (ENUM_ORDER_TYPE, MetaTrader5 Python package): BUY=0, SELL=1,
#: BUY_LIMIT=2, SELL_LIMIT=3, BUY_STOP=4, SELL_STOP=5. Note that the
#: LIMIT/STOP ordering alternates by side -- SELL_LIMIT precedes BUY_STOP.
_ORDER_TYPE_FALLBACK: Dict[str, int] = {
    "BUY": 0,
    "SELL": 1,
    "BUY_LIMIT": 2,
    "SELL_LIMIT": 3,
    "BUY_STOP": 4,
    "SELL_STOP": 5,
}


def _order_type_value(order_type: OrderType) -> int:
    return const(f"ORDER_TYPE_{order_type.value}", _ORDER_TYPE_FALLBACK[order_type.value])


def _order_type_from_int(value: int) -> OrderType:
    for order_type in OrderType:
        if _order_type_value(order_type) == value:
            return order_type
    raise MT5Error(f"unknown pending order type {value}")


def choose_filling_mode(spec: SymbolSpec) -> int:
    """Pick a filling mode supported by THIS symbol (FOK > IOC > RETURN)."""
    if spec.filling_mode & 1:
        return const("ORDER_FILLING_FOK", 0)
    if spec.filling_mode & 2:
        return const("ORDER_FILLING_IOC", 1)
    return const("ORDER_FILLING_RETURN", 2)


def build_request(
    order_type: OrderType,
    symbol: str,
    volume: float,
    price: float,
    sl: float,
    tp: float,
    deviation: int,
    magic: int,
    comment: str,
    spec: SymbolSpec,
) -> Dict[str, Any]:
    """Build a complete, explicit order request dict (no hidden fields).

    Market orders use TRADE_ACTION_DEAL (1), pending orders use
    TRADE_ACTION_PENDING (5) per official MT5 Python constants.
    """
    is_pending = order_type in (
        OrderType.BUY_LIMIT,
        OrderType.BUY_STOP,
        OrderType.SELL_LIMIT,
        OrderType.SELL_STOP,
    )
    if is_pending:
        action = const("TRADE_ACTION_PENDING", 5)
    else:
        action = const("TRADE_ACTION_DEAL", 1)
    return {
        "action": action,
        "symbol": symbol,
        "volume": float(volume),
        "type": _order_type_value(order_type),
        "price": float(price),
        "sl": round(float(sl), spec.digits),
        "tp": round(float(tp), spec.digits),
        "deviation": int(deviation),
        "magic": int(magic),
        "comment": comment,
        "type_time": const("ORDER_TIME_GTC", 0),
        "type_filling": choose_filling_mode(spec),
    }


def _format_request(request: Dict[str, Any]) -> str:
    """Compact, secret-free description of a request for WOULD/result logs."""
    keys = (
        "action", "symbol", "type", "volume", "price", "sl", "tp",
        "order", "position", "magic",
    )
    return " ".join(f"{key}={request[key]}" for key in keys if key in request)


def send_request(
    request: Dict[str, Any],
    *,
    permission: Optional[ExecutionPermission] = None,
) -> OrderResult:
    """Send an order request, or record WOULD EXECUTE when live sends are closed.

    This is the only production path to ``mt5.order_send``. The installed
    execution permission (injected by the application from its config, never
    read from ``.env`` here) must allow live execution. An explicit
    ``permission`` may only narrow that gate.

    When the gate is closed, neither ``order_check`` nor ``order_send`` is
    called. The returned result has ``blocked_by_safety=True`` and a comment
    that starts with ``WOULD EXECUTE``.
    """
    block_reason = resolve_execution_permission(permission).block_reason()
    if block_reason is not None:
        logger.info(
            "WOULD EXECUTE | blocked before order_send (%s) | %s",
            block_reason,
            _format_request(request),
        )
        return OrderResult(
            success=False,
            retcode=LOCAL_REJECTION,
            comment=f"WOULD EXECUTE ({block_reason})",
            order=0,
            request=dict(request),
            blocked_by_safety=True,
        )

    mt5_api = require_mt5()

    action = request.get("action")
    remove_action = const("TRADE_ACTION_REMOVE", 8)

    # Preflight check via order_check for trade requests (DEAL, PENDING, SLTP)
    if action != remove_action and hasattr(mt5_api, "order_check") and callable(mt5_api.order_check):
        try:
            check_res = mt5_api.order_check(request)
        except Exception as exc:
            logger.error("order_check raised exception: %s", exc)
            return OrderResult(
                False,
                LOCAL_REJECTION,
                f"order_check exception: {exc}",
                0,
                dict(request),
            )

        if check_res is None:
            err = str(mt5_api.last_error()) if hasattr(mt5_api, "last_error") else "unknown"
            logger.error("order_check returned None: %s", err)
            return OrderResult(
                False,
                LOCAL_REJECTION,
                f"order_check returned None: {err}",
                0,
                dict(request),
            )

        retcode = int(getattr(check_res, "retcode", -1))
        # 0 (CHECK_OK), 10008 (RETCODE_PLACED), 10009 (RETCODE_DONE), 10010 (RETCODE_DONE_PARTIAL)
        if retcode not in (0, RETCODE_DONE, RETCODE_DONE_PARTIAL, RETCODE_PLACED):
            comment = str(getattr(check_res, "comment", "") or "rejected by order_check")
            logger.warning(
                "order_check rejected request: retcode=%s comment=%r",
                retcode,
                comment,
            )
            return OrderResult(
                False,
                retcode,
                f"order_check rejected: {comment}",
                0,
                dict(request),
            )

    result = mt5_api.order_send(request)
    if result is None:
        err = str(mt5_api.last_error()) if hasattr(mt5_api, "last_error") else "unknown"
        return OrderResult(
            False,
            LOCAL_REJECTION,
            f"order_send returned None: {err}",
            0,
            dict(request),
        )
    retcode = int(result.retcode)
    success = retcode in (RETCODE_DONE, RETCODE_DONE_PARTIAL, RETCODE_PLACED)
    return OrderResult(
        success=success,
        retcode=retcode,
        comment=str(getattr(result, "comment", "")),
        order=int(getattr(result, "order", 0) or 0),
        request=dict(request),
    )


def _validate_pending_distance(
    spec: SymbolSpec, tick: TickData, order_type: OrderType, price: float
) -> Optional[str]:
    """Broker minimum distance between a pending order and the current price."""
    min_distance = spec.min_stop_distance()
    if order_type is OrderType.BUY_LIMIT and not price < tick.bid - min_distance:
        return (
            f"BUY_LIMIT {price:.5f} must be at least {min_distance:.5f} below "
            f"bid {tick.bid:.5f}"
        )
    if order_type is OrderType.BUY_STOP and not price > tick.ask + min_distance:
        return (
            f"BUY_STOP {price:.5f} must be at least {min_distance:.5f} above "
            f"ask {tick.ask:.5f}"
        )
    if order_type is OrderType.SELL_LIMIT and not price > tick.ask + min_distance:
        return (
            f"SELL_LIMIT {price:.5f} must be at least {min_distance:.5f} above "
            f"ask {tick.ask:.5f}"
        )
    if order_type is OrderType.SELL_STOP and not price < tick.bid - min_distance:
        return (
            f"SELL_STOP {price:.5f} must be at least {min_distance:.5f} below "
            f"bid {tick.bid:.5f}"
        )
    return None


def _local_rejection(problem: str, symbol: str, price: float, volume: float) -> OrderResult:
    logger.error("pending order rejected locally: %s", problem)
    return OrderResult(
        False,
        LOCAL_REJECTION,
        problem,
        0,
        {"symbol": symbol, "price": price, "volume": volume},
    )


# ---------------------------------------------------------------------------
# plan dispatcher (used by the main loop)
# ---------------------------------------------------------------------------

def send_plan(
    plan: TradePlan,
    spec: SymbolSpec,
    tick: TickData,
    magic: int,
    deviation: int,
) -> OrderResult:
    """Dispatch a TradePlan (market or pending) to the terminal."""
    if plan.order_type in (OrderType.BUY, OrderType.SELL):
        price = tick.ask if plan.order_type is OrderType.BUY else tick.bid
    else:
        problem = _validate_pending_distance(spec, tick, plan.order_type, plan.entry)
        if problem:
            return _local_rejection(problem, plan.symbol, plan.entry, plan.volume or 0.0)
        price = plan.entry
    if plan.volume is None or plan.volume <= 0:
        return _local_rejection(
            "invalid volume in plan", plan.symbol, price, plan.volume or 0.0
        )
    request = build_request(
        plan.order_type,
        plan.symbol,
        plan.volume,
        price,
        plan.sl,
        plan.tp,
        deviation,
        magic,
        plan.comment,
        spec,
    )
    return send_request(request)


# ---------------------------------------------------------------------------
# explicit functions per order type (public API)
# ---------------------------------------------------------------------------

def place_market_buy(
    spec: SymbolSpec, tick: TickData, volume: float, sl: float, tp: float,
    magic: int, deviation: int, comment: str,
) -> OrderResult:
    """Market BUY at the current ask."""
    request = build_request(
        OrderType.BUY, spec.name, volume, tick.ask, sl, tp, deviation, magic, comment, spec
    )
    return send_request(request)


def place_market_sell(
    spec: SymbolSpec, tick: TickData, volume: float, sl: float, tp: float,
    magic: int, deviation: int, comment: str,
) -> OrderResult:
    """Market SELL at the current bid."""
    request = build_request(
        OrderType.SELL, spec.name, volume, tick.bid, sl, tp, deviation, magic, comment, spec
    )
    return send_request(request)


def place_buy_limit(
    spec: SymbolSpec, tick: TickData, volume: float, price: float, sl: float, tp: float,
    magic: int, deviation: int, comment: str,
) -> OrderResult:
    """BUY_LIMIT below the current bid (distance validated first)."""
    problem = _validate_pending_distance(spec, tick, OrderType.BUY_LIMIT, price)
    if problem:
        return _local_rejection(problem, spec.name, price, volume)
    request = build_request(
        OrderType.BUY_LIMIT, spec.name, volume, price, sl, tp, deviation, magic, comment, spec
    )
    return send_request(request)


def place_buy_stop(
    spec: SymbolSpec, tick: TickData, volume: float, price: float, sl: float, tp: float,
    magic: int, deviation: int, comment: str,
) -> OrderResult:
    """BUY_STOP above the current ask (distance validated first)."""
    problem = _validate_pending_distance(spec, tick, OrderType.BUY_STOP, price)
    if problem:
        return _local_rejection(problem, spec.name, price, volume)
    request = build_request(
        OrderType.BUY_STOP, spec.name, volume, price, sl, tp, deviation, magic, comment, spec
    )
    return send_request(request)


def place_sell_limit(
    spec: SymbolSpec, tick: TickData, volume: float, price: float, sl: float, tp: float,
    magic: int, deviation: int, comment: str,
) -> OrderResult:
    """SELL_LIMIT above the current ask (distance validated first)."""
    problem = _validate_pending_distance(spec, tick, OrderType.SELL_LIMIT, price)
    if problem:
        return _local_rejection(problem, spec.name, price, volume)
    request = build_request(
        OrderType.SELL_LIMIT, spec.name, volume, price, sl, tp, deviation, magic, comment, spec
    )
    return send_request(request)


def place_sell_stop(
    spec: SymbolSpec, tick: TickData, volume: float, price: float, sl: float, tp: float,
    magic: int, deviation: int, comment: str,
) -> OrderResult:
    """SELL_STOP below the current bid (distance validated first)."""
    problem = _validate_pending_distance(spec, tick, OrderType.SELL_STOP, price)
    if problem:
        return _local_rejection(problem, spec.name, price, volume)
    request = build_request(
        OrderType.SELL_STOP, spec.name, volume, price, sl, tp, deviation, magic, comment, spec
    )
    return send_request(request)


# ---------------------------------------------------------------------------
# pending order listing / deletion
# ---------------------------------------------------------------------------

def get_pending_orders(
    magic: int, symbol: Optional[str] = None
) -> List[PendingOrderInfo]:
    """All pending orders of THIS bot (magic-filtered), optionally by symbol."""
    mt5_api = require_mt5()
    raw = mt5_api.orders_get(symbol=symbol) if symbol else mt5_api.orders_get()
    if raw is None:
        from .connection import MT5DataError

        raise MT5DataError(f"orders_get() failed: {mt5_api.last_error()}")
    out: list[PendingOrderInfo] = []
    for order in raw:
        if order.magic != magic:
            continue
        time_expire = getattr(order, "time_expire", 0)
        out.append(
            PendingOrderInfo(
                ticket=int(order.ticket),
                symbol=order.symbol,
                order_type=_order_type_from_int(int(order.type)),
                price=float(order.price_open),
                volume=float(order.volume),
                sl=float(order.sl),
                tp=float(order.tp),
                magic=int(order.magic),
                comment=str(getattr(order, "comment", "") or ""),
                time_setup=pd.to_datetime(order.time_setup, unit="s", utc=True),
                time_expire=(
                    pd.to_datetime(time_expire, unit="s", utc=True)
                    if time_expire
                    else None
                ),
            )
        )
    return out


def delete_order(ticket: int) -> OrderResult:
    """Delete a pending order (by ticket)."""
    request = {
        # TRADE_ACTION_REMOVE = 8 (official), fallback 8
        "action": const("TRADE_ACTION_REMOVE", 8),
        "order": int(ticket),
    }
    return send_request(request)
