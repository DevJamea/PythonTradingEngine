"""Bot position access and management (magic-number filtered).

Only positions carrying the bot's magic number are returned/modified;
manual user trades are never touched. Partial closes and SL/TP
modifications work the same on NETTING and HEDGING accounts (both close
through a position-close deal / SLTP modification).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import pandas as pd

from ..models import OrderResult, OrderType, PositionInfo, SymbolSpec
from ._constants import const
from .connection import MT5DataError, require_mt5
from .orders import _order_type_value, choose_filling_mode, send_request


def get_positions(
    symbol: Optional[str] = None, magic: Optional[int] = None
) -> List[PositionInfo]:
    """Open positions, optionally filtered by symbol and/or magic number.

    Passing ``magic`` returns ONLY the bot's positions.
    """
    mt5_api = require_mt5()
    raw = mt5_api.positions_get(symbol=symbol) if symbol else mt5_api.positions_get()
    if raw is None:
        raise MT5DataError(f"positions_get() failed: {mt5_api.last_error()}")
    out: list[PositionInfo] = []
    for position in raw:
        if magic is not None and position.magic != magic:
            continue
        out.append(
            PositionInfo(
                ticket=int(position.ticket),
                symbol=position.symbol,
                is_buy=(int(position.type) == const("POSITION_TYPE_BUY", 0)),
                volume=float(position.volume),
                price_open=float(position.price_open),
                sl=float(position.sl),
                tp=float(position.tp),
                profit=float(position.profit),
                magic=int(position.magic),
                comment=str(getattr(position, "comment", "") or ""),
                open_time=pd.to_datetime(position.time, unit="s", utc=True),
            )
        )
    return out


def close_position(
    position: PositionInfo,
    spec: SymbolSpec,
    volume: Optional[float] = None,
    magic: int = 0,
    deviation: int = 20,
    comment: str = "",
) -> OrderResult:
    """Close all (or part of) a position via a market close deal.

    ``volume=None`` closes the full position. Valid for both NETTING and
    HEDGING account modes.
    """
    close_volume = position.volume if volume is None else volume
    if close_volume <= 0 or close_volume > position.volume:
        raise ValueError(
            f"close volume {close_volume} invalid for position volume {position.volume}"
        )
    close_type = OrderType.SELL if position.is_buy else OrderType.BUY
    request = {
        "action": const("ORDER_ACTION_DEAL", 1),
        "symbol": position.symbol,
        "volume": float(close_volume),
        "type": _order_type_value(close_type),
        "position": int(position.ticket),
        "magic": int(magic),
        "comment": comment,
        "type_time": const("ORDER_TIME_GTC", 0),
        "type_filling": choose_filling_mode(spec),
    }
    return send_request(request)


def modify_position_sltp(
    position: PositionInfo, sl: float, tp: float, spec: SymbolSpec
) -> OrderResult:
    """Move the SL/TP of an open position (e.g. break-even, trailing)."""
    request = {
        "action": const("ORDER_ACTION_SLTP", 4),
        "position": int(position.ticket),
        "symbol": position.symbol,
        "sl": round(float(sl), spec.digits),
        "tp": round(float(tp), spec.digits),
    }
    return send_request(request)


def total_profit(positions: Sequence[PositionInfo]) -> float:
    """Floating P/L of the given positions."""
    return sum(p.profit for p in positions)
