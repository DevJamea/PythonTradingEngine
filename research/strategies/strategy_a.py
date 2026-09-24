"""Strategy A — HTF Trend + Structural Pullback (research-only).

Hypothesis (frozen, see research/config.py):
* H4 bias: close vs EMA200 + EMA50 vs EMA200 alignment;
* H1 structure: HH+HL (BUY) / LH+LL (SELL) from confirmed swings (N=3);
* M15 pullback: |close - EMA50| <= 1.0*ATR14, structural swing stays valid;
* Trigger: M15 close beyond the last confirmed internal swing (N=2);
* PRIMARY entry at next-bar open ("close" variant); "retest" is diagnostic;
* Stop at pullback swing ± 0.3*ATR; target RR 2.0 (1.5/2.5 neighbours);
* PRIMARY management: BE@1R, 50% partial@1R, 1xATR trail after 1R.

All indicator series are causal; HTF values are mapped with the no-leakage
rule (only fully closed H1/H4 bars are visible to an M15 close).
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..indicators_ext import (HTFFrame, atr, confirmed_swings, ema,
                              map_htf_to_m15, market_structure, resample_htf)
from ..config import StrategyAParams
from .base import ResearchSignal, StrategyOutput


def _h4_bias(h4: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    close = h4["close"].astype(float)
    ef = ema(close, fast)
    es = ema(close, slow)
    out = pd.DataFrame({"close": close, "ema_fast": ef, "ema_slow": es})
    out["bias_long"] = (close > es) & (ef > es)
    out["bias_short"] = (close < es) & (ef < es)
    return out


def generate_a_signals(df: pd.DataFrame, p: StrategyAParams) -> StrategyOutput:
    n = len(df)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    m15_ema50 = ema(close, p.m15_ema_pullback)
    m15_atr = atr(high, low, close, p.m15_atr_period)
    m15_sw = confirmed_swings(high, low, p.trigger_swing_n)

    # --- H4 bias mapped onto M15 (causal) ---
    h4 = resample_htf(df, "4h").frame
    h4_ind = _h4_bias(h4, p.h4_ema_fast, p.h4_ema_slow)
    h4_mapped = map_htf_to_m15(
        df["time"], HTFFrame(h4),
        {"h4_long": h4_ind["bias_long"].astype(float),
         "h4_short": h4_ind["bias_short"].astype(float)})

    # --- H1 structure mapped onto M15 (causal) ---
    h1 = resample_htf(df, "h").frame
    h1_struct = market_structure(h1["high"].astype(float), h1["low"].astype(float),
                                 p.h1_swing_n)
    h1_sw = confirmed_swings(h1["high"].astype(float), h1["low"].astype(float), p.h1_swing_n)
    h1_mapped = map_htf_to_m15(
        df["time"], HTFFrame(h1),
        {"HH": h1_struct["HH"].astype(float), "HL": h1_struct["HL"].astype(float),
         "LH": h1_struct["LH"].astype(float), "LL": h1_struct["LL"].astype(float),
         "h1_last_low": h1_sw.low_price, "h1_last_high": h1_sw.high_price})

    o = df["open"].to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    hh = h1_mapped["HH"].to_numpy(dtype=float)
    hl = h1_mapped["HL"].to_numpy(dtype=float)
    lh = h1_mapped["LH"].to_numpy(dtype=float)
    ll = h1_mapped["LL"].to_numpy(dtype=float)
    h1_lo = h1_mapped["h1_last_low"].to_numpy(dtype=float)
    h1_hi = h1_mapped["h1_last_high"].to_numpy(dtype=float)
    h4_lo = h4_mapped["h4_long"].to_numpy(dtype=float)
    h4_sh = h4_mapped["h4_short"].to_numpy(dtype=float)
    e50 = m15_ema50.to_numpy(dtype=float)
    av = m15_atr.to_numpy(dtype=float)
    sw_h = m15_sw.high_price.to_numpy(dtype=float)
    sw_l = m15_sw.low_price.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    highs = high.to_numpy(dtype=float)

    signals: list[ResearchSignal] = []
    diag: Dict[str, Any] = {"evaluated_bars": 0, "h4_pass": 0, "h1_pass": 0,
                             "pullback_pass": 0, "trigger_pass": 0}
    start = max(p.m15_ema_pullback, p.m15_atr_period) + 2
    for i in range(start, n - 1):
        if not (np.isfinite(e50[i]) and np.isfinite(av[i]) and av[i] > 0):
            continue
        if np.isnan(h4_lo[i]) or np.isnan(hh[i]):
            continue  # HTF not ready — no signal (honest warmup)
        diag["evaluated_bars"] += 1
        want_long = bool(h4_lo[i] > 0.5) and bool(hh[i] > 0.5) and bool(hl[i] > 0.5)
        want_short = bool(h4_sh[i] > 0.5) and bool(lh[i] > 0.5) and bool(ll[i] > 0.5)
        if want_long or want_short:
            diag["h4_pass"] += 1
            diag["h1_pass"] += 1
        else:
            continue
        # --- pullback: close near EMA50 ---
        if abs(c[i] - e50[i]) > p.pullback_max_atr * av[i]:
            continue
        diag["pullback_pass"] += 1
        # --- structural validity + pullback swing over the lookback ---
        lo0 = max(0, i - p.pullback_lookback)
        if want_long:
            if np.isfinite(h1_lo[i]) and np.any(c[lo0:i + 1] < h1_lo[i]):
                continue  # H1 swing low violated -> structure invalid
            swing = float(np.min(lows[lo0:i + 1]))
            level = sw_h[i]  # last confirmed internal swing high
            if not np.isfinite(level):
                continue
            if not (c[i] > level):
                continue
            sl_d = (c[i] - swing) + p.stop_buffer_atr * av[i]
        else:
            if np.isfinite(h1_hi[i]) and np.any(c[lo0:i + 1] > h1_hi[i]):
                continue
            swing = float(np.max(highs[lo0:i + 1]))
            level = sw_l[i]
            if not np.isfinite(level):
                continue
            if not (c[i] < level):
                continue
            sl_d = (swing - c[i]) + p.stop_buffer_atr * av[i]
        if not (np.isfinite(sl_d) and sl_d > 0):
            continue
        diag["trigger_pass"] += 1
        tp_d = sl_d * p.rr_target / 1.0  # RR multiple of realised SL distance
        entry_ix = i + 1
        meta: Dict[str, Any] = {"params": "A",
                                "pullback_swing": swing,
                                "trigger_level": float(level)}
        if p.entry_variant == "retest":
            # Diagnostic: wait up to retest_window bars for a touch of the
            # trigger level; enter at that bar's open; else no signal.
            filled = False
            for k in range(1, p.retest_window + 1):
                if i + k >= n:
                    break
                touched = (lows[i + k] <= level) if want_long else (highs[i + k] >= level)
                # retest must not violate the pullback swing
                violated = (lows[i + k] < swing) if want_long else (highs[i + k] > swing)
                if violated:
                    break
                if touched:
                    entry_ix = i + k
                    filled = True
                    break
            if not filled:
                continue
            meta["retest_fill_bar"] = entry_ix
        signals.append(ResearchSignal(
            index=i, entry_index=entry_ix,
            direction="BUY" if want_long else "SELL",
            sl_distance=float(sl_d), tp_distance=float(tp_d), meta=meta))
    diag["raw_signals"] = len(signals)
    return StrategyOutput(signals, diag)


def a_entry_model(p: StrategyAParams) -> str:
    return ("market at next-bar open after M15 close beyond internal swing"
            if p.entry_variant == "close"
            else f"retest limit within {p.retest_window} bars (diagnostic)")


def a_exit_model(p: StrategyAParams) -> str:
    return (f"pullback-swing stop + {p.stop_buffer_atr}xATR buffer; "
            f"fixed RR {p.rr_target}")


def a_management_model(p: StrategyAParams) -> str:
    if p.management == "none":
        return "none (entry/exit-only diagnostic)"
    return (f"BE@{p.be_trigger_r}R(+{p.be_buffer}), "
            f"partial {p.partial_fraction:.0%}@{p.partial_r}R, "
            f"{p.trail_atr_mult}xATR trail after {p.trail_activation_r}R")
