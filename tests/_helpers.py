"""Shared deterministic helpers for the unit tests."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from gold_trader.config import Config

# Sentinel: publish an account whose trade_mode cannot be read.
_UNSET: Any = object()


def make_cfg(**overrides) -> Config:
    """A Config with defaults, overridden by keyword arguments."""
    return replace(Config(), **overrides)


def build_df(rows, start: str = "2026-01-05 00:00:00", freq: str = "15min") -> pd.DataFrame:
    """Build a closed-candle OHLCV DataFrame from (open, high, low, close) rows."""
    n = len(rows)
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=n, freq=freq, tz="UTC"),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "tick_volume": [100] * n,
            "spread": [10] * n,
            "real_volume": [0] * n,
        }
    )


def make_uptrend_with_engulfing(n_ramp: int = 220, n_pullback: int = 6) -> pd.DataFrame:
    """Deterministic BUY scenario.

    Steady uptrend, short pullback (6 bearish candles), then a bullish
    engulfing of the last bearish candle. Designed so that at the final
    bar: close > EMA20 > EMA50 > EMA200 and RSI(14) < 70.
    """
    rows = []
    price = 2000.0
    for _ in range(n_ramp):
        o = price
        c = o + 0.8
        rows.append((o, c + 0.2, o - 0.2, c))
        price = c
    for _ in range(n_pullback):
        o = price
        c = o - 0.9
        rows.append((o, o + 0.2, c - 0.2, c))
        price = c
    # last (bearish) candle geometry: open = price + 0.9, close = price
    engulf_open = price - 0.2
    engulf_close = price + 1.9
    rows.append((engulf_open, engulf_close + 0.2, engulf_open - 0.2, engulf_close))
    return build_df(rows)


def make_downtrend_with_engulfing(n_ramp: int = 220, n_retrace: int = 6) -> pd.DataFrame:
    """Deterministic SELL scenario (mirror of the BUY scenario)."""
    rows = []
    price = 2000.0
    for _ in range(n_ramp):
        o = price
        c = o - 0.8
        rows.append((o, o + 0.2, c - 0.2, c))
        price = c
    for _ in range(n_retrace):
        o = price
        c = o + 0.9
        rows.append((o, c + 0.2, o - 0.2, c))
        price = c
    # last (bullish) candle geometry: open = price - 0.9, close = price
    engulf_open = price + 0.2
    engulf_close = price - 1.9
    rows.append((engulf_open, engulf_open + 0.2, engulf_close - 0.2, engulf_close))
    return build_df(rows)


class TerminalAccount:
    """Minimal ``account_info()`` object. Omit ``trade_mode`` with ``_UNSET``."""

    def __init__(self, trade_mode: Any = _UNSET) -> None:
        self.login = 10001
        self.server = "Stub-Server"
        self.currency = "USD"
        self.balance = 10_000.0
        self.equity = 10_000.0
        self.trade_allowed = True
        self.trade_expert = True
        self.margin_mode = 2
        self._include_mode = trade_mode is not _UNSET
        if self._include_mode:
            self.trade_mode = trade_mode

    def _asdict(self) -> dict:
        data = {
            "login": self.login,
            "server": self.server,
            "currency": self.currency,
            "balance": self.balance,
            "equity": self.equity,
            "trade_allowed": self.trade_allowed,
            "trade_expert": self.trade_expert,
            "margin_mode": self.margin_mode,
        }
        if self._include_mode:
            data["trade_mode"] = self.trade_mode
        return data


def publish_terminal_account(
    trade_mode: Any = _UNSET,
    *,
    explode: bool = False,
) -> None:
    """Point the patched MT5 module at a broker account.

    Opt-in execution tests must prove Demo through this object. Writing
    ``account_is_demo=True`` on a permission is not a substitute.
    """
    from gold_trader.mt5.connection import require_mt5

    api = require_mt5()
    if not isinstance(getattr(api, "ACCOUNT_TRADE_MODE_DEMO", None), int):
        api.ACCOUNT_TRADE_MODE_DEMO = 0
    if not isinstance(getattr(api, "ACCOUNT_TRADE_MODE_REAL", None), int):
        api.ACCOUNT_TRADE_MODE_REAL = 2
    if explode:
        api.account_info.side_effect = RuntimeError("account_info unavailable")
        return
    api.account_info.side_effect = None
    api.account_info.return_value = TerminalAccount(trade_mode)


def make_scalp_frame(
    side: str = "buy",
    n_flat: int = 70,
    n_push: int = 4,
    step: float = 0.9,
    wiggle: float = 0.02,
    freq: str = "5min",
    start: str = "2026-01-05 00:00:00",
) -> pd.DataFrame:
    """Deterministic mean-reversion scalp scenario for the scalping engine.

    A quiet flat stretch (so Bollinger has a narrow, well-defined band), a
    sharp excursion of ``n_push`` candles that pushes the close outside the
    lower (BUY) or upper (SELL) band with an extreme fast RSI, then one
    reversal candle closing back inside the band. The last row is the signal
    candle; the design makes every gate pass with
    ``spread=0.05, scalp_tp_atr_multiple=1.0``.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    rows = []
    price = 2000.0
    for i in range(n_flat):
        o = price
        c = o + (wiggle if i % 2 == 0 else -wiggle)
        rows.append((o, max(o, c) + wiggle, min(o, c) - wiggle, c))
        price = c
    direction = -1.0 if side == "buy" else 1.0
    for _ in range(n_push):
        o = price
        c = o + direction * step
        rows.append(
            (o, max(o, c) + wiggle, min(o, c) - wiggle, c)
        )
        price = c
    o = price
    c = o - direction * step          # reversal candle back towards the mean
    rows.append((o, max(o, c) + wiggle, min(o, c) - wiggle, c))
    return build_df(rows, start=start, freq=freq)


def scalp_cfg(**overrides):
    """Config tuned for the deterministic scalp fixture (filters neutralised)."""
    base = dict(
        scalping_enabled=True,
        active_strategy="scalping",
        scalp_vol_min_percentile=0.0,
        scalp_vol_max_percentile=1.0,
        scalp_vol_lookback=2,
        scalp_min_candles_for_signal=60,
        scalp_tp_atr_multiple=1.0,
        scalp_sl_atr_multiple=1.0,
        scalp_spread_safety_margin=0.05,
        min_profit_to_spread_ratio=3.0,
    )
    base.update(overrides)
    return make_cfg(**base)
