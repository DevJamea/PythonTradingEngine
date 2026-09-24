"""Research backtest engine — event-driven over closed M15 bars.

No-lookahead rules (same discipline as production, plus Bid/Ask):
* a signal on closed bar ``i`` enters at the OPEN of ``entry_index`` (>= i+1);
* SL/TP are checked from the entry bar onward on the conservative Bid/Ask
  side; a bar touching both fills SL first (conservative, documented);
* trailing distance uses the entry-bar ATR (fixed for the trade);
* partial fills execute at the exact R-level on the exit side;
* signals before the segment's tradeable start (warmup) are discarded;
* signals arriving while a position is open, or whose entry bar already
  passed, are skipped and COUNTED (never silently dropped);
* sizing reuses the production ``calculate_position_size`` read-only.

Costs: entry/exit on the correct Bid/Ask side, optional adverse slippage in
points, optional delay-proxy spread fraction at entry. Commission/swap are
unavailable and recorded as 0.0 with explicit flags (never invented).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from gold_trader.models import SymbolSpec, default_gold_spec
from gold_trader.risk.position_size import (calculate_position_size,
                                            profit_for_volume)

from ..costs import BidAsk, TradeCost


@dataclass(frozen=True)
class ManagementSpec:
    use_fixed_tp: bool = True
    be_trigger_r: Optional[float] = None
    be_buffer: float = 0.0
    partial_r: Optional[float] = None
    partial_fraction: float = 0.0
    trail_mult: Optional[float] = None  # ATR multiple, None = no trailing
    trail_activation_r: float = 1.0


MgmtResolver = Callable[[object], ManagementSpec]


@dataclass
class Fill:
    bar: int
    price: float
    volume: float
    pnl_gross: float
    cost_money: float
    reason: str  # "partial" | "stop_loss" | "take_profit" | "end_of_data"


@dataclass
class ResearchTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry: float
    exit_price: float       # last exit price (VWAP in exit_vwap)
    exit_vwap: float
    sl_initial: float
    tp_initial: Optional[float]
    volume_initial: float
    risk_amount: float
    pnl_gross: float
    pnl_net: float
    r_gross: float
    r_net: float
    reason: str
    equity_after: float
    entry_spread: float
    exit_spread: float
    cost: TradeCost
    cost_money: float
    duration_bars: int
    fills: List[Fill] = field(default_factory=list)
    signal_index: int = -1
    entry_index: int = -1


@dataclass
class EngineDiagnostics:
    signals_received: int = 0
    signals_warmup_discarded: int = 0
    signals_skipped_in_position: int = 0
    signals_sizing_failed: int = 0
    trades_executed: int = 0


@dataclass
class EngineResult:
    trades: List[ResearchTrade]
    initial_balance: float
    final_equity: float
    total_cost_money: float
    diagnostics: EngineDiagnostics


def _no_mgmt() -> ManagementSpec:
    return ManagementSpec()


def run_backtest(
    df: pd.DataFrame,
    signals: list,
    bidask: BidAsk,
    spec: Optional[SymbolSpec] = None,
    initial_balance: float = 10_000.0,
    risk_per_trade: float = 0.005,
    mgmt: MgmtResolver | ManagementSpec = ManagementSpec(),
    trade_start_pos: int = 0,
    atr_for_trail: Optional[np.ndarray] = None,
    slip_entry_pts: Optional[np.ndarray] = None,
    slip_exit_pts: Optional[np.ndarray] = None,
    point: float = 0.01,
    delay_entry_spread_frac: float = 0.0,
) -> EngineResult:
    """Run the research backtest. ``df`` rows align with ``bidask`` arrays."""
    spec = spec or default_gold_spec()
    resolver: MgmtResolver = mgmt if callable(mgmt) else (lambda _s: mgmt)
    n = len(df)
    times = pd.to_datetime(df["time"], utc=True)
    pos = (df["_pos"].to_numpy(dtype=int) if "_pos" in df.columns
           else np.arange(n))

    ordered = sorted(signals, key=lambda s: (s.entry_index, s.index))
    diag = EngineDiagnostics(signals_received=len(ordered))
    trades: List[ResearchTrade] = []
    equity = float(initial_balance)
    total_cost = 0.0

    # index signals by entry bar for O(1) lookup during the bar loop
    by_entry: dict[int, list[tuple[int, object]]] = {}
    for seq, s in enumerate(ordered):
        by_entry.setdefault(s.entry_index, []).append((seq, s))

    open_pos: Optional[dict] = None

    def _slip(seq: int, arr: Optional[np.ndarray]) -> float:
        if arr is None:
            return 0.0
        return float(arr[seq]) * point if seq < len(arr) else 0.0

    for i in range(n):
        if open_pos is not None:
            closed = _manage_bar(open_pos, i, bidask, spec, times, pos,
                                 atr_for_trail, _slip(open_pos["seq"], slip_exit_pts))
            if closed is not None:
                equity += closed.pnl_net
                closed.equity_after = equity
                total_cost += closed.cost_money
                trades.append(closed)
                open_pos = None
            continue
        # no open position: check due signals at this bar
        for seq, s in by_entry.get(i, []):
            if pos[i] < trade_start_pos or pos[s.index] < trade_start_pos:
                diag.signals_warmup_discarded += 1
                continue
            opened = _open_position(
                s, seq, i, df, bidask, spec, equity, risk_per_trade,
                resolver(s), times, pos, atr_for_trail,
                _slip(seq, slip_entry_pts), delay_entry_spread_frac)
            if opened is None:
                diag.signals_sizing_failed += 1
                continue
            open_pos = opened
            # Same convention as the production engine: the entry bar's own
            # high/low can already stop the trade out (SL-first on ambiguity).
            closed = _manage_bar(open_pos, i, bidask, spec, times, pos,
                                 atr_for_trail, _slip(open_pos["seq"], slip_exit_pts))
            if closed is not None:
                equity += closed.pnl_net
                closed.equity_after = equity
                total_cost += closed.cost_money
                trades.append(closed)
                open_pos = None
            break  # one position at a time; other same-bar signals are skipped

    # Skipped-in-position accounting is derived by conservation:
    # received = warmup_discarded + sizing_failed + executed(+open) + skipped.
    if open_pos is not None:
        # force-close at the last exit-side close
        i = n - 1
        t = _force_close(open_pos, i, bidask, spec, times, pos,
                         _slip(open_pos["seq"], slip_exit_pts))
        equity += t.pnl_net
        t.equity_after = equity
        total_cost += t.cost_money
        trades.append(t)

    # fix skipped count now that end-of-data close is included
    diag.trades_executed = len(trades)
    diag.signals_skipped_in_position = max(
        0, diag.signals_received - diag.signals_warmup_discarded
        - diag.signals_sizing_failed - len(trades))
    return EngineResult(trades, float(initial_balance), equity, total_cost, diag)


def _open_position(s, seq: int, i: int, df, ba: BidAsk, spec: SymbolSpec,
                   equity: float, risk_pct: float, m: ManagementSpec,
                   times, pos, atr_trail, slip_pts: float,
                   delay_frac: float) -> Optional[dict]:
    is_buy = s.direction == "BUY"
    spread = float(ba.spread[i])
    if is_buy:
        entry = float(ba.ask_open[i]) + slip_pts + delay_frac * spread
        sl = entry - s.sl_distance
        tp = entry + s.tp_distance if m.use_fixed_tp else None
    else:
        entry = float(ba.bid_open[i]) - slip_pts - delay_frac * spread
        sl = entry + s.sl_distance
        tp = entry - s.tp_distance if m.use_fixed_tp else None
    if not (entry > 0 and sl > 0 and (tp is None or tp > 0)):
        return None
    sizing = calculate_position_size(equity, risk_pct, entry, sl, spec)
    if sizing.volume is None:
        return None
    atr_entry = 0.0
    if atr_trail is not None and m.trail_mult is not None:
        atr_entry = float(atr_trail[i]) if np.isfinite(atr_trail[i]) else 0.0
    r_dist = abs(entry - sl)
    return {
        "seq": seq, "signal": s, "is_buy": is_buy, "entry_bar": i,
        "entry": entry, "sl": sl, "tp": tp, "sl_init": sl, "tp_init": tp,
        "volume": sizing.volume, "volume_left": sizing.volume,
        "risk_amount": sizing.risk_amount, "r_dist": r_dist,
        "entry_spread": spread, "slip_entry": slip_pts,
        "mgmt": m, "atr_entry": atr_entry,
        "be_done": False, "partial_done": m.partial_fraction <= 0,
        "extreme": entry, "fills": [],
        "pnl_gross": 0.0, "cost_money": 0.0, "spread_paid": 0.0,
        "slip_paid": 0.0, "exit_spread_last": spread,
    }


def _fill(p: dict, bar: int, price: float, volume: float, spec: SymbolSpec,
          spread: float, slip_pts: float, reason: str) -> None:
    dist = (price - p["entry"]) if p["is_buy"] else (p["entry"] - price)
    gross = profit_for_volume(abs(dist), volume, spec) or 0.0
    signed = gross if dist >= 0 else -gross
    # cost: half-spread on this fill's exit leg (+ entry-leg half spread is
    # attributed once, on the first fill) + slippage legs.
    if not p["fills"]:
        entry_leg = profit_for_volume(p["entry_spread"] / 2.0 + p["slip_entry"],
                                      p["volume"], spec) or 0.0
        p["cost_money"] += entry_leg
        p["spread_paid"] += p["entry_spread"] / 2.0
        p["slip_paid"] += p["slip_entry"]
    exit_leg = profit_for_volume(spread / 2.0 + slip_pts, volume, spec) or 0.0
    p["cost_money"] += exit_leg
    p["spread_paid"] += (spread / 2.0) * (volume / p["volume"])
    p["slip_paid"] += slip_pts * (volume / p["volume"])
    p["pnl_gross"] += signed
    p["volume_left"] = round(p["volume_left"] - volume, 10)
    p["exit_spread_last"] = spread
    p["fills"].append(Fill(bar, price, volume, signed, exit_leg, reason))


def _manage_bar(p: dict, i: int, ba: BidAsk, spec: SymbolSpec, times, pos,
                atr_trail, slip_exit: float):
    """Advance the open position through bar ``i``. Returns ResearchTrade on close."""
    m: ManagementSpec = p["mgmt"]
    is_buy = p["is_buy"]
    spread = float(ba.spread[i])
    if is_buy:
        x_high, x_low = float(ba.bid_high[i]), float(ba.bid_low[i])
    else:
        x_high, x_low = float(ba.ask_high[i]), float(ba.ask_low[i])
    # track favourable extreme on the exit side
    p["extreme"] = max(p["extreme"], x_high) if is_buy else min(p["extreme"], x_low)

    # 1) SL check first (conservative)
    sl_hit = (x_low <= p["sl"]) if is_buy else (x_high >= p["sl"])
    tp_hit = False
    if p["tp"] is not None:
        tp_hit = (x_high >= p["tp"]) if is_buy else (x_low <= p["tp"])
    if sl_hit:
        _fill(p, i, p["sl"], p["volume_left"], spec, spread, slip_exit, "stop_loss")
        return _to_trade(p, i, times, pos, "stop_loss")
    # 2) management triggers (BE / partial / trail update)
    r = p["r_dist"]
    if r > 0:
        lvl_1r = p["entry"] + r * (m.be_trigger_r or 1.0) * (1 if is_buy else -1)
        reached_1r = (x_high >= lvl_1r) if is_buy else (x_low <= lvl_1r)
        if reached_1r:
            if m.be_trigger_r is not None and not p["be_done"]:
                new_sl = (p["entry"] + m.be_buffer) if is_buy else (p["entry"] - m.be_buffer)
                if (new_sl > p["sl"]) if is_buy else (new_sl < p["sl"]):
                    p["sl"] = new_sl
                p["be_done"] = True
            if m.partial_r is not None and not p["partial_done"]:
                plvl = p["entry"] + r * m.partial_r * (1 if is_buy else -1)
                hit = (x_high >= plvl) if is_buy else (x_low <= plvl)
                if hit:
                    vol = round(p["volume"] * m.partial_fraction, 10)
                    vol = min(vol, p["volume_left"])
                    if vol > 0:
                        _fill(p, i, plvl, vol, spec, spread, slip_exit, "partial")
                    p["partial_done"] = True
        if m.trail_mult is not None and p["atr_entry"] > 0 and p["be_done"]:
            trail_d = m.trail_mult * p["atr_entry"]
            cand = p["extreme"] - trail_d if is_buy else p["extreme"] + trail_d
            if (cand > p["sl"]) if is_buy else (cand < p["sl"]):
                p["sl"] = cand
            # re-check SL after trailing update (same bar, conservative)
            sl_hit2 = (x_low <= p["sl"]) if is_buy else (x_high >= p["sl"])
            if sl_hit2:
                _fill(p, i, p["sl"], p["volume_left"], spec, spread, slip_exit, "stop_loss")
                return _to_trade(p, i, times, pos, "stop_loss")
    # 3) TP check
    if tp_hit and p["tp"] is not None:
        _fill(p, i, p["tp"], p["volume_left"], spec, spread, slip_exit, "take_profit")
        return _to_trade(p, i, times, pos, "take_profit")
    return None


def _to_trade(p: dict, exit_bar: int, times, pos, reason: str) -> ResearchTrade:
    s = p["signal"]
    pnl_net = p["pnl_gross"] - p["cost_money"]
    risk = p["risk_amount"] if p["risk_amount"] > 0 else 1.0
    exit_vwap = (sum(f.price * f.volume for f in p["fills"])
                 / p["volume"] if p["fills"] else p["entry"])
    last_px = p["fills"][-1].price if p["fills"] else p["entry"]
    return ResearchTrade(
        entry_time=times.iloc[p["entry_bar"]], exit_time=times.iloc[exit_bar],
        side="BUY" if p["is_buy"] else "SELL", entry=p["entry"],
        exit_price=last_px, exit_vwap=exit_vwap,
        sl_initial=p["sl_init"], tp_initial=p["tp_init"],
        volume_initial=p["volume"], risk_amount=p["risk_amount"],
        pnl_gross=p["pnl_gross"], pnl_net=pnl_net,
        r_gross=p["pnl_gross"] / risk, r_net=pnl_net / risk,
        reason=reason if reason != "partial" else "take_profit",
        equity_after=0.0,  # set by caller
        entry_spread=p["entry_spread"], exit_spread=p["exit_spread_last"],
        cost=TradeCost(spread_paid=p["spread_paid"], slippage_paid=p["slip_paid"],
                       commission=0.0, swap=0.0,
                       total_distance=p["spread_paid"] + p["slip_paid"]),
        cost_money=p["cost_money"],
        duration_bars=exit_bar - p["entry_bar"],
        fills=list(p["fills"]), signal_index=s.index, entry_index=p["entry_bar"],
    )


def _force_close(p: dict, i: int, ba: BidAsk, spec: SymbolSpec, times, pos,
                 slip_exit: float) -> ResearchTrade:
    is_buy = p["is_buy"]
    spread = float(ba.spread[i])
    px = float(ba.bid_close[i]) if is_buy else float(ba.ask_close[i])
    _fill(p, i, px, p["volume_left"], spec, spread, slip_exit, "end_of_data")
    return _to_trade(p, i, times, pos, "end_of_data")
