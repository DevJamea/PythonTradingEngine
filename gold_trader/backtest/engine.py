"""Event-driven backtest over closed OHLCV candles.

No-lookahead rules:
* at closed candle ``i`` the strategy only sees rows ``0..i``;
* a signal on candle ``i`` enters at the OPEN of candle ``i+1``;
* SL/TP are checked with the entry candle's high/low and afterwards;
* when a single candle touches both SL and TP, the SL is assumed to be
  hit first (conservative assumption, documented here and in tests).

Indicators are precomputed once on the full frame. Because every
indicator in this project is causal (the value at ``i`` depends only on
rows ``0..i``), that is mathematically identical to recomputing on the
visible window -- a unit test asserts this equivalence explicitly.

Other simplifications (documented): one position at a time (matches the
default ``MAX_OPEN_POSITIONS = 1``), no daily-loss simulation, spread
cost applied as a fixed price distance per side
(``Config.backtest_spread_cost``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from ..config import Config
from ..models import Signal, SymbolSpec, default_gold_spec
from ..risk.position_size import calculate_position_size, profit_for_volume
from ..strategy.levels import build_sl_tp
from ..strategy.signals import compute_indicators, evaluate_at
from .metrics import BacktestMetrics, TradeResult, compute_metrics


@dataclass
class BacktestResult:
    """Full result of one backtest run."""

    metrics: BacktestMetrics
    trades: List[TradeResult]
    initial_balance: float
    final_equity: float


@dataclass
class _OpenTrade:
    entry_index: int
    is_buy: bool
    entry: float
    sl: float
    tp: float
    volume: float
    risk_amount: float


class BacktestEngine:
    """Runs the live strategy logic over historical candles."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(
        self,
        df: pd.DataFrame,
        initial_balance: Optional[float] = None,
        spec: Optional[SymbolSpec] = None,
    ) -> BacktestResult:
        """Simulate the strategy over ``df`` (closed candles, OHLCV frame)."""
        cfg = self.cfg
        if len(df) < cfg.min_candles_for_signal + 2:
            raise ValueError(
                f"need at least {cfg.min_candles_for_signal + 2} candles, "
                f"got {len(df)}"
            )
        symbol_spec = spec or default_gold_spec()
        balance = float(
            cfg.backtest_initial_balance if initial_balance is None else initial_balance
        )

        ind = compute_indicators(df, cfg)
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        times = df["time"]

        equity = balance
        trades: list[TradeResult] = []
        open_trade: Optional[_OpenTrade] = None
        warmup = cfg.min_candles_for_signal

        for i in range(warmup, len(df) - 1):
            if open_trade is not None:
                exit_price, reason = self._check_exit(open_trade, highs[i], lows[i])
                if exit_price is not None:
                    pnl = self._realized_pnl(open_trade, exit_price, symbol_spec)
                    equity += pnl
                    trades.append(
                        self._record(
                            open_trade, i, exit_price, reason, pnl, equity, times
                        )
                    )
                    open_trade = None
                continue

            signal = evaluate_at(i, df, ind, cfg)
            if signal.signal is Signal.NO_TRADE:
                continue
            is_buy = signal.signal is Signal.BUY
            entry = float(opens[i + 1])
            atr_value = float(ind.atr.iloc[i])
            if not math.isfinite(atr_value) or atr_value <= 0:
                continue
            try:
                sltp = build_sl_tp(signal.signal, entry, atr_value, df.iloc[: i + 1], cfg)
            except ValueError:
                continue
            sizing = calculate_position_size(
                equity, cfg.risk_per_trade, entry, sltp.sl, symbol_spec
            )
            if sizing.volume is None:
                continue
            open_trade = _OpenTrade(
                entry_index=i + 1,
                is_buy=is_buy,
                entry=entry,
                sl=sltp.sl,
                tp=sltp.tp,
                volume=sizing.volume,
                risk_amount=sizing.risk_amount,
            )

        if open_trade is not None:
            last = len(df) - 1
            exit_price = float(closes[last])
            pnl = self._realized_pnl(open_trade, exit_price, symbol_spec)
            equity += pnl
            trades.append(
                self._record(open_trade, last, exit_price, "end_of_data", pnl, equity, times)
            )

        return BacktestResult(
            metrics=compute_metrics(trades, balance),
            trades=trades,
            initial_balance=balance,
            final_equity=equity,
        )

    # -- internals ---------------------------------------------------------

    def _check_exit(
        self, trade: _OpenTrade, high: float, low: float
    ) -> Tuple[Optional[float], str]:
        """Return (exit_price, reason) when SL/TP is hit this candle."""
        if trade.is_buy:
            if low <= trade.sl:
                return trade.sl, "stop_loss"
            if high >= trade.tp:
                return trade.tp, "take_profit"
        else:
            if high >= trade.sl:
                return trade.sl, "stop_loss"
            if low <= trade.tp:
                return trade.tp, "take_profit"
        return None, ""

    def _realized_pnl(
        self, trade: _OpenTrade, exit_price: float, spec: SymbolSpec
    ) -> float:
        """Signed P/L using broker tick economics, minus round-trip spread cost."""
        distance = (
            exit_price - trade.entry
            if trade.is_buy
            else trade.entry - exit_price
        )
        gross = profit_for_volume(abs(distance), trade.volume, spec)
        signed = gross if distance >= 0 else -(gross or 0.0)
        cost = profit_for_volume(
            2.0 * self.cfg.backtest_spread_cost, trade.volume, spec
        )
        return signed - (cost or 0.0)

    def _record(
        self,
        trade: _OpenTrade,
        exit_index: int,
        exit_price: float,
        reason: str,
        pnl: float,
        equity: float,
        times: pd.Series,
    ) -> TradeResult:
        return TradeResult(
            entry_time=times.iloc[trade.entry_index],
            exit_time=times.iloc[exit_index],
            side="BUY" if trade.is_buy else "SELL",
            entry=trade.entry,
            exit_price=exit_price,
            sl=trade.sl,
            tp=trade.tp,
            volume=trade.volume,
            pnl=pnl,
            reason=reason,
            equity_after=equity,
        )
