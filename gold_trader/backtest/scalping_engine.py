"""Scalping backtest engine with real (per-bar, variable) execution costs.

Same no-lookahead discipline as :mod:`gold_trader.backtest.engine` -- signal on
closed bar ``i``, entry at the open of bar ``i+1``, SL/TP checked bar by bar,
and a bar that touches both is charged the stop loss first (conservative).

What is deliberately different, because a scalper lives or dies on it:

* **Fill geometry.** A BUY is filled at ``bid_open + spread`` (the ask) and a
  SELL at ``bid_open``; the SL/TP levels are measured from that fill. Exits of
  a long trigger on the bid path, exits of a short on the ask path
  (``bid + spread``) -- the same asymmetry a real MT5 server applies. The
  spread is therefore charged *once, structurally*, never subtracted twice.
* **No position management.** Full close at SL or TP only (plus an end-of-data
  close). Break-even / partial close / trailing are not applicable to a target
  measured in tens of pips and are excluded here by design.
* **Daily entry cap.** ``Config.scalp_max_trades_per_day`` is enforced on
  simulated entries per UTC day, exactly like the live gate will be.
* **Cost accounting is a first-class output.** Every trade records the spread
  distance it paid, and the result totals the money the spread took, so a
  report can put "spread paid" next to "net P/L".

Like the original engine this simulates ONE position at a time (matching the
default ``MAX_OPEN_POSITIONS = 1``) and does not simulate the daily-loss
halt -- documented simplifications, both identical to the trend engine so the
two strategies stay comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import Config
from ..models import Signal, SymbolSpec, default_gold_spec
from ..risk.position_size import calculate_position_size, profit_for_volume
from ..strategy.scalping import (
    ScalpingSLTP,
    build_scalping_sl_tp,
    compute_scalping_indicators,
    evaluate_scalping_at,
    fast_view,
    validate_scalping_config,
)
from .cost_model import CostModel
from .metrics import BacktestMetrics, TradeResult, compute_metrics


@dataclass(frozen=True)
class ScalpingTradeRecord:
    """One simulated scalp plus the cost detail behind it."""

    trade: TradeResult
    spread_price: float
    commission: float
    bars_held: int
    signal_index: int


@dataclass
class ScalpingBacktestResult:
    """Result of one scalping run (metrics + cost accounting + veto counts)."""

    metrics: BacktestMetrics
    trades: List[TradeResult]
    records: List[ScalpingTradeRecord]
    initial_balance: float
    final_equity: float
    total_spread_cost: float
    total_commission_cost: float
    veto_counts: Dict[str, int] = field(default_factory=dict)
    bars: int = 0
    cost_model: str = ""

    def as_dict(self) -> Dict[str, object]:
        """Flat summary for reports and comparisons."""
        m = self.metrics
        return {
            "bars": self.bars,
            "total_trades": m.total_trades,
            "wins": m.wins,
            "losses": m.losses,
            "win_rate": m.win_rate,
            "profit_factor": m.profit_factor,
            "net_profit": m.net_profit,
            "gross_profit": m.gross_profit,
            "gross_loss": m.gross_loss,
            "max_drawdown": m.max_drawdown,
            "max_drawdown_pct": m.max_drawdown_pct,
            "average_win": m.average_win,
            "average_loss": m.average_loss,
            "final_equity": self.final_equity,
            "total_spread_cost": self.total_spread_cost,
            "total_commission_cost": self.total_commission_cost,
            "veto_counts": dict(self.veto_counts),
            "cost_model": self.cost_model,
        }


@dataclass
class _LiveScalp:
    signal_index: int
    entry_index: int
    is_buy: bool
    entry: float
    sl: float
    tp: float
    volume: float
    risk_amount: float
    spread_price: float
    commission: float


class ScalpingBacktestEngine:
    """Event-driven simulation of :mod:`gold_trader.strategy.scalping`."""

    def __init__(self, cfg: Config, cost: Optional[CostModel] = None) -> None:
        """``cost=None`` derives the cost model from the frame at :meth:`run`
        (real per-bar ``spread`` column when present, otherwise the fixed
        conservative floor). Pass an explicit :class:`CostModel` to force a
        scenario, e.g. a stress run.
        """
        validate_scalping_config(cfg)
        self.cfg = cfg
        self.cost = cost

    # -- public ------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        initial_balance: Optional[float] = None,
        spec: Optional[SymbolSpec] = None,
    ) -> ScalpingBacktestResult:
        """Simulate the scalper over a closed-candle frame (``spread`` optional)."""
        cfg = self.cfg
        warmup = int(cfg.scalp_min_candles_for_signal)
        if len(df) < warmup + 2:
            raise ValueError(
                f"need at least {warmup + 2} candles, got {len(df)}"
            )
        symbol_spec = spec or default_gold_spec()
        balance = float(
            cfg.backtest_initial_balance if initial_balance is None else initial_balance
        )
        cost = self.cost if self.cost is not None else CostModel.from_frame(df, cfg)

        ind = compute_scalping_indicators(df, cfg)
        # numpy read-only view: identical decisions, no per-bar pandas indexing
        df_view, ind_view = fast_view(df, ind)
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        times = pd.to_datetime(df["time"], utc=True)

        equity = balance
        trades: List[TradeResult] = []
        records: List[ScalpingTradeRecord] = []
        vetoes: Dict[str, int] = {}
        entries_by_day: Dict[object, int] = {}
        live: Optional[_LiveScalp] = None
        total_spread_cost = 0.0
        total_commission_cost = 0.0

        def veto(code: str) -> None:
            vetoes[code] = vetoes.get(code, 0) + 1

        for i in range(warmup, len(df) - 1):
            day = times.iloc[i].date()
            if live is not None:
                outcome = self._check_exit(live, highs[i], lows[i], cost, i)
                if outcome is not None:
                    exit_price, reason = outcome
                    total_spread_cost, total_commission_cost = self._close(
                        live, i, exit_price, reason, closes, times, equity, trades,
                        records, symbol_spec, cost, total_spread_cost,
                        total_commission_cost,
                    )
                    equity = records[-1].trade.equity_after
                    live = None
                continue

            signal = evaluate_scalping_at(
                i,
                df_view,
                ind_view,
                cfg,
                spread=cost.spread_at(i),
                trades_today=entries_by_day.get(day, 0),
            )
            if signal.signal is Signal.NO_TRADE:
                veto(str(signal.details.get("reject_code", "no_signal")))
                continue

            entry_index = i + 1
            spread_price = cost.spread_at(entry_index)
            slippage = cost.slippage_price()
            is_buy = signal.signal is Signal.BUY
            bid_open = float(opens[entry_index])
            # BUY pays the ask; SELL receives the bid. Slippage always worsens.
            entry_fill = (
                bid_open + spread_price + slippage
                if is_buy
                else bid_open - slippage
            )
            plan = self._plan(is_buy, entry_fill, float(ind.atr.iloc[i]), spread_price, symbol_spec)
            if plan is None:
                # build_scalping_sl_tp refuses: TP below the cost requirement,
                # below the broker minimum stop distance, or invalid ATR.
                veto("plan_refused")
                continue
            sizing = calculate_position_size(
                equity, cfg.scalping_risk_per_trade, entry_fill, plan.sl, symbol_spec
            )
            if sizing.volume is None:
                veto("sizing_failed")
                continue
            commission = cost.commission_currency(entry_index, sizing.volume)
            live = _LiveScalp(
                signal_index=i,
                entry_index=entry_index,
                is_buy=is_buy,
                entry=entry_fill,
                sl=plan.sl,
                tp=plan.tp,
                volume=sizing.volume,
                risk_amount=sizing.risk_amount,
                spread_price=spread_price,
                commission=commission,
            )
            entries_by_day[day] = entries_by_day.get(day, 0) + 1

        if live is not None:
            last = len(df) - 1
            exit_price = (
                float(closes[last]) if live.is_buy else float(closes[last]) + cost.spread_at(last)
            )
            total_spread_cost, total_commission_cost = self._close(
                live, last, exit_price, "end_of_data", closes, times, equity, trades,
                records, symbol_spec, cost, total_spread_cost, total_commission_cost,
            )
            equity = records[-1].trade.equity_after

        return ScalpingBacktestResult(
            metrics=compute_metrics(trades, balance),
            trades=trades,
            records=records,
            initial_balance=balance,
            final_equity=equity,
            total_spread_cost=total_spread_cost,
            total_commission_cost=total_commission_cost,
            veto_counts=vetoes,
            bars=int(len(df)),
            cost_model=cost.describe(),
        )

    # -- internals ----------------------------------------------------------

    def _plan(
        self,
        is_buy: bool,
        entry_fill: float,
        atr_value: float,
        spread_price: float,
        spec: SymbolSpec,
    ) -> Optional[ScalpingSLTP]:
        """Build the SL/TP, refusing plans the broker would not even accept."""
        direction = Signal.BUY if is_buy else Signal.SELL
        return build_scalping_sl_tp(
            direction,
            entry_fill,
            atr_value,
            spread_price,
            self.cfg,
            min_stop_distance=spec.min_stop_distance(),
        )

    def _check_exit(
        self, live: _LiveScalp, high: float, low: float, cost: CostModel, index: int
    ) -> Optional[Tuple[float, str]]:
        """Return (exit_price, reason) if SL/TP triggers on this bar.

        Longs are settled at the bid, so their levels are compared with the
        bid path directly. Shorts are settled at the ask (``bid + spread``), so
        their stop is reached sooner and their target later -- that asymmetry is
        exactly what makes scalping short targets hard to hit.
        """
        if live.is_buy:
            if low <= live.sl:
                return live.sl, "stop_loss"
            if high >= live.tp:
                return live.tp, "take_profit"
            return None
        shift = cost.spread_at(index)
        if high >= live.sl - shift:
            return live.sl, "stop_loss"
        if low <= live.tp - shift:
            return live.tp, "take_profit"
        return None

    def _close(
        self,
        live: _LiveScalp,
        exit_index: int,
        exit_price: float,
        reason: str,
        closes,
        times: pd.Series,
        equity: float,
        trades: List[TradeResult],
        records: List[ScalpingTradeRecord],
        spec: SymbolSpec,
        cost: CostModel,
        total_spread_cost: float,
        total_commission_cost: float,
    ) -> Tuple[float, float]:
        """Realise one trade: signed price P/L minus commission, both costs logged."""
        distance = exit_price - live.entry if live.is_buy else live.entry - exit_price
        gross = profit_for_volume(abs(distance), live.volume, spec) or 0.0
        signed = gross if distance >= 0 else -gross
        pnl = signed - live.commission
        new_equity = equity + pnl
        trade = TradeResult(
            entry_time=times.iloc[live.entry_index],
            exit_time=times.iloc[exit_index],
            side="BUY" if live.is_buy else "SELL",
            entry=live.entry,
            exit_price=float(exit_price),
            sl=live.sl,
            tp=live.tp,
            volume=live.volume,
            pnl=pnl,
            reason=reason,
            equity_after=new_equity,
        )
        trades.append(trade)
        records.append(
            ScalpingTradeRecord(
                trade=trade,
                spread_price=live.spread_price,
                commission=live.commission,
                bars_held=int(exit_index - live.entry_index) + 1,
                signal_index=live.signal_index,
            )
        )
        # spread actually paid, expressed in money: 1 spread per round trip
        spread_money = profit_for_volume(live.spread_price, live.volume, spec) or 0.0
        return total_spread_cost + spread_money, total_commission_cost + live.commission
