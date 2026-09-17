"""MetaTrader 5 integration layer (thin wrappers over the official API).

Design rules:
* the MetaTrader5 package is Windows-only; imports are guarded so the rest
  of the project (strategy, risk, backtest, tests) runs on any platform;
* only bot orders/positions (magic number) are ever read for decisions or
  modified -- manual user trades are never touched;
* every failure surfaces as a typed MT5*Error with a clear message and is
  logged together with retcode/comment/request details (no credentials).
"""
from .connection import (
    MT5_AVAILABLE,
    MT5Connection,
    MT5ConnectionError,
    MT5DataError,
    MT5Error,
    require_mt5,
)
from .market_data import TickData, drop_unclosed_candle, get_candles, get_tick
from .orders import (
    delete_order,
    get_pending_orders,
    place_buy_limit,
    place_buy_stop,
    place_market_buy,
    place_market_sell,
    place_sell_limit,
    place_sell_stop,
    send_plan,
    send_request,
)
from .positions import (
    close_position,
    get_positions,
    modify_position_sltp,
    total_profit,
)
from .symbols import GoldSymbolNotFoundError, find_gold_symbol

__all__ = [
    "MT5_AVAILABLE",
    "MT5Connection",
    "MT5ConnectionError",
    "MT5DataError",
    "MT5Error",
    "require_mt5",
    "TickData",
    "drop_unclosed_candle",
    "get_candles",
    "get_tick",
    "delete_order",
    "get_pending_orders",
    "place_buy_limit",
    "place_buy_stop",
    "place_market_buy",
    "place_market_sell",
    "place_sell_limit",
    "place_sell_stop",
    "send_plan",
    "send_request",
    "close_position",
    "get_positions",
    "modify_position_sltp",
    "total_profit",
    "GoldSymbolNotFoundError",
    "find_gold_symbol",
]
