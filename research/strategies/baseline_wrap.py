"""Baseline = current production M15 strategy, UNMODIFIED.

This module only *calls* the production functions
(``signals.compute_indicators`` / ``signals.evaluate_at`` / ``levels.build_sl_tp``)
with a production ``Config``. It contains no copy of the strategy rules, so it
cannot drift from production. The research backtest engine then prices these
signals with the unified Bid/Ask cost model (the production backtest engine's
fixed ``backtest_spread_cost`` is replaced for comparability — documented in
the report; the signal/SLTP rules are byte-identical to production).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from gold_trader.config import Config
from gold_trader.models import Signal
from gold_trader.strategy.levels import build_sl_tp
from gold_trader.strategy.signals import compute_indicators, evaluate_at

from .base import ResearchSignal, StrategyOutput


@dataclass(frozen=True)
class BaselineSpec:
    entry_model: str = "market at open of bar i+1 (production timing)"
    exit_model: str = "production ATR SL/TP (1.5x / 2.5x), SL-first on ambiguity"
    management_model: str = "none in backtest (production backtest has no management)"
    session_filter: str = "none (production strategy has no session filter)"


def generate_baseline_signals(df: pd.DataFrame, cfg: Config | None = None) -> StrategyOutput:
    """Evaluate the production strategy on every closed bar of ``df``.

    ``df`` must contain the warmup prefix; the engine discards signals before
    the segment's tradeable start. Indicators are precomputed once (causal —
    production maths, covered by production leakage tests).
    """
    cfg = cfg or Config()
    ind = compute_indicators(df, cfg)
    signals: list[ResearchSignal] = []
    evaluated = 0
    for i in range(cfg.min_candles_for_signal, len(df) - 1):
        evaluated += 1
        sig = evaluate_at(i, df, ind, cfg)
        if sig.signal is Signal.NO_TRADE:
            continue
        atr_value = float(ind.atr.iloc[i])
        if not math.isfinite(atr_value) or atr_value <= 0:
            continue
        # SL/TP distances from the production builder (entry reference = close
        # of signal bar; the engine re-anchors to the actual fill price while
        # keeping production ATR distances — same economics as live).
        ref = float(df["close"].iloc[i])
        plan = build_sl_tp(sig.signal, ref, atr_value, df.iloc[: i + 1], cfg)
        if sig.signal is Signal.BUY:
            sl_d, tp_d = ref - plan.sl, plan.tp - ref
        else:
            sl_d, tp_d = plan.sl - ref, ref - plan.tp
        if sl_d <= 0 or tp_d <= 0:
            continue
        signals.append(ResearchSignal(
            index=i, entry_index=i + 1,
            direction="BUY" if sig.signal is Signal.BUY else "SELL",
            sl_distance=float(sl_d), tp_distance=float(tp_d),
            meta={"reason": sig.reason},
        ))
    return StrategyOutput(signals, {"evaluated_bars": evaluated,
                                    "raw_signals": len(signals)})
