"""Strategy B — Volatility Compression Breakout (research-only).

Hypothesis (frozen, see research/config.py):
* Compression: ATR14/SMA(ATR14,50) < 0.70 for >= 6 consecutive M15 bars;
* Range: Donchian(20) of the 20 bars *before* the signal bar;
* Breakout: close beyond the range with body >= 0.5*ATR;
* Confirmation: next bar must NOT close back inside the old range;
  entry at the open of the bar after confirmation (index+2);
* Stop at the other side of the range ± 0.2*ATR; PRIMARY target RR 2.0 fixed;
* PRIMARY session = all permitted hours; London/NY = declared diagnostic;
* exit_mode "trail" (1.5xATR trail, BE@1R) is a declared diagnostic.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..config import StrategyBParams
from ..indicators_ext import atr, atr_ratio, donchian, sma
from .base import ResearchSignal, StrategyOutput, inside_session


def generate_b_signals(df: pd.DataFrame, p: StrategyBParams) -> StrategyOutput:
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    times = pd.to_datetime(df["time"], utc=True)
    hi_s = pd.Series(h)
    lo_s = pd.Series(l)
    cl_s = pd.Series(c)
    av = atr(hi_s, lo_s, cl_s, p.atr_period).to_numpy(dtype=float)
    # ATR ratio needs the ATR series itself (not OHLC): compute directly.
    atr_ser = pd.Series(av)
    ratio = (atr_ser / sma(atr_ser, p.atr_sma_period)).to_numpy(dtype=float)
    d_hi, d_lo = donchian(hi_s, lo_s, p.donchian_period, shift=True)
    d_hi = d_hi.to_numpy(dtype=float)
    d_lo = d_lo.to_numpy(dtype=float)

    # consecutive compression run ending at each bar
    compressed = np.isfinite(ratio) & (ratio < p.compression_ratio)
    run = np.zeros(n, dtype=int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if compressed[i] else 0

    signals: list[ResearchSignal] = []
    diag: Dict[str, Any] = {"evaluated_bars": 0, "compressed_bars": int(compressed.sum()),
                             "breakout_pass": 0, "confirm_pass": 0}
    start = max(p.atr_period + p.atr_sma_period, p.donchian_period) + 2
    for i in range(start, n - 2):  # need i+1 for confirmation, i+2 for entry
        if not (np.isfinite(av[i]) and av[i] > 0):
            continue
        if run[i] < p.compression_min_bars:
            continue
        if not (np.isfinite(d_hi[i]) and np.isfinite(d_lo[i])):
            continue
        diag["evaluated_bars"] += 1
        body = abs(c[i] - o[i])
        want_long = (c[i] > d_hi[i]) and (body >= p.body_min_atr * av[i])
        want_short = (c[i] < d_lo[i]) and (body >= p.body_min_atr * av[i])
        if not (want_long or want_short):
            continue
        diag["breakout_pass"] += 1
        # --- confirmation: next close must not fall back inside old range ---
        if want_long and c[i + 1] < d_hi[i]:
            continue
        if want_short and c[i + 1] > d_lo[i]:
            continue
        diag["confirm_pass"] += 1
        entry_ix = i + 2
        # --- session filter on the entry bar (predefined variants) ---
        if p.session == "london_ny":
            lo, hi = p.session_london_ny
            if not inside_session(times.iloc[entry_ix], lo, hi):
                continue
        if want_long:
            sl_d = (c[i + 1] - d_lo[i]) + p.stop_buffer_atr * av[i]
        else:
            sl_d = (d_hi[i] - c[i + 1]) + p.stop_buffer_atr * av[i]
        if not (np.isfinite(sl_d) and sl_d > 0):
            continue
        tp_d = sl_d * p.rr_target
        signals.append(ResearchSignal(
            index=i, entry_index=entry_ix,
            direction="BUY" if want_long else "SELL",
            sl_distance=float(sl_d), tp_distance=float(tp_d),
            meta={"params": "B",
                   "range_high": float(d_hi[i]), "range_low": float(d_lo[i]),
                   "confirm_close": float(c[i + 1])}))
    diag["raw_signals"] = len(signals)
    return StrategyOutput(signals, diag)


def b_entry_model(p: StrategyBParams) -> str:
    return ("market at open of bar i+2 after Donchian(20) breakout + "
            "next-bar confirmation (no close back inside)")


def b_exit_model(p: StrategyBParams) -> str:
    if p.exit_mode == "trail":
        return (f"opposite range side + {p.stop_buffer_atr}xATR stop; "
                f"BE@1R + {p.trail_atr_mult}xATR trail, no fixed TP (diagnostic)")
    return (f"opposite range side + {p.stop_buffer_atr}xATR stop; "
            f"fixed RR {p.rr_target}")


def b_management_model(p: StrategyBParams) -> str:
    if p.exit_mode == "trail":
        return f"BE@{p.be_trigger_r}R(+{p.be_buffer}), {p.trail_atr_mult}xATR trail"
    return "none (fixed SL/TP)"


def b_session_filter(p: StrategyBParams) -> str:
    if p.session == "london_ny":
        lo, hi = p.session_london_ny
        return f"London/NY {lo}-{hi} UTC (diagnostic)"
    return "all permitted trading hours (00:00-23:59 UTC)"
