"""Strategy C — Regime Adaptive (research-only).

Hypothesis (frozen, see research/config.py):
* TREND regime: ADX14 > 25 AND |normalised EMA50 slope| > 0.05
  -> route to Strategy-A rules (same frozen A parameters);
* COMPRESSION regime: ATR-ratio < 0.70 -> route to Strategy-B rules;
* RANGE regime: ADX < 20 AND 0.8 <= ATR-ratio <= 1.3 -> NO_TRADE
  (conservative predefined choice: no mean-reversion system is invented in
  this round; a range module would be a new experiment round);
* otherwise -> NO_TRADE.

Implementation: candidate signals are generated with the frozen A/B rule
functions; a signal is kept only if its signal bar's regime routes to the
strategy that produced it. Regime is evaluated strictly on the signal bar
(all causal). Degrees of freedom are tracked in the report.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..config import StrategyCParams
from ..indicators_ext import adx, atr, atr_ratio, ema, ema_slope_norm, sma
from .base import StrategyOutput
from .strategy_a import generate_a_signals
from .strategy_b import generate_b_signals


def regime_series(df: pd.DataFrame, p: StrategyCParams) -> pd.DataFrame:
    """Per-bar regime labels (causal): TREND / COMPRESSION / RANGE / NONE."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    adx_v = adx(high, low, close, p.adx_period)
    atr_v = atr(high, low, close, p.atr_period)
    ratio = atr_v / sma(atr_v, p.atr_sma_period)
    slope = ema_slope_norm(ema(close, p.ema_slope_period), atr_v, p.ema_slope_lookback)
    adx_a = adx_v.to_numpy(dtype=float)
    ratio_a = ratio.to_numpy(dtype=float)
    slope_a = slope.to_numpy(dtype=float)
    labels = np.full(len(df), "NONE", dtype=object)
    valid = np.isfinite(adx_a) & np.isfinite(ratio_a) & np.isfinite(slope_a)
    trend = valid & (adx_a > p.adx_trend) & (np.abs(slope_a) > p.slope_threshold)
    comp = valid & (ratio_a < p.compression_ratio) & ~trend
    rng = (valid & (adx_a < p.adx_range_max)
           & (ratio_a >= p.range_ratio_lo) & (ratio_a <= p.range_ratio_hi)
           & ~trend & ~comp)
    labels[trend] = "TREND"
    labels[comp] = "COMPRESSION"
    labels[rng] = "RANGE"
    return pd.DataFrame({"regime": labels, "adx": adx_a,
                         "atr_ratio": ratio_a, "slope_norm": slope_a})


def generate_c_signals(df: pd.DataFrame, p: StrategyCParams) -> StrategyOutput:
    regimes = regime_series(df, p)["regime"].to_numpy()
    out_a = generate_a_signals(df, p.trend_params)
    out_b = generate_b_signals(df, p.compression_params)
    kept = []
    dropped_a = dropped_b = 0
    for s in out_a.signals:
        if regimes[s.index] == "TREND":
            kept.append(s)
        else:
            dropped_a += 1
    for s in out_b.signals:
        if regimes[s.index] == "COMPRESSION":
            kept.append(s)
        else:
            dropped_b += 1
    kept.sort(key=lambda s: s.entry_index)
    unique, seen = [], set()
    for s in kept:  # same-bar A+B collision -> keep first, count the drop
        if (s.entry_index, s.direction) in seen:
            continue
        seen.add((s.entry_index, s.direction))
        unique.append(s)
    mix = {k: int((regimes == k).sum()) for k in ("TREND", "COMPRESSION", "RANGE", "NONE")}
    diag: Dict[str, Any] = {"regime_mix": mix,
                             "a_candidates": len(out_a.signals),
                             "b_candidates": len(out_b.signals),
                             "a_dropped_off_regime": dropped_a,
                             "b_dropped_off_regime": dropped_b,
                             "range_action": p.range_action,
                             "raw_signals": len(unique)}
    return StrategyOutput(unique, diag)


def c_entry_model(p: StrategyCParams) -> str:
    return ("regime-routed: TREND->A rules, COMPRESSION->B rules, "
            "RANGE->NO_TRADE (predefined), else NO_TRADE")


def c_exit_model(p: StrategyCParams) -> str:
    return "inherited from routed strategy (A: swing stop/RR; B: range stop/RR)"


def c_management_model(p: StrategyCParams) -> str:
    return ("inherited from routed strategy "
            "(A: BE/partial/trail; B fixed unless trail diagnostic)")
