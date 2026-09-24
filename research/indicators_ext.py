"""Causal research indicators (all values at ``i`` use rows ``0..i`` only).

Reuses the production EMA/ATR/TR implementations by import (read-only) so the
maths matches the live engine, and adds SMA, ADX, Donchian, ATR-ratio,
normalised EMA slope, confirmed-swing detection and HTF (H1/H4) resampling
with strictly causal mapping back onto M15 bars.

Leakage rule for HTF mapping: an H1/H4 bar with end time E may only be used
by M15 bars whose own end time is >= E (i.e. the HTF bar has fully closed).
Implemented with ``merge_asof`` on bar-end timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Read-only reuse of production indicator maths (never modified, never copied).
from gold_trader.strategy.indicators import atr as prod_atr
from gold_trader.strategy.indicators import ema as prod_ema
from gold_trader.strategy.indicators import true_range as prod_tr


def ema(series: pd.Series, period: int) -> pd.Series:
    return prod_ema(series, period)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return prod_atr(high, low, close, period)


def sma(series: pd.Series, period: int) -> pd.Series:
    if period < 1:
        raise ValueError("SMA period must be >= 1")
    return series.rolling(window=period, min_periods=period).mean()


def atr_ratio(atr_s: pd.Series, period: int = 50) -> pd.Series:
    """ATR14 / SMA(ATR14, 50) — the compression gauge (causal)."""
    return atr_s / sma(atr_s, period)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ADX (causal). Returns 0..100, NaN during warmup."""
    if period < 1:
        raise ValueError("ADX period must be >= 1")
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = prod_tr(high, low, close)
    atr_s = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False,
                                   min_periods=period).mean() / atr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False,
                                     min_periods=period).mean() / atr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def donchian(high: pd.Series, low: pd.Series, period: int,
             shift: bool = True) -> Tuple[pd.Series, pd.Series]:
    """Donchian channel. ``shift=True`` (default) excludes the current bar so
    the channel at ``i`` uses bars ``i-period..i-1`` (no leakage)."""
    hi = high.rolling(window=period, min_periods=period).max()
    lo = low.rolling(window=period, min_periods=period).min()
    if shift:
        hi, lo = hi.shift(1), lo.shift(1)
    return hi, lo


def ema_slope_norm(ema_s: pd.Series, atr_s: pd.Series, lookback: int = 10) -> pd.Series:
    """Normalised EMA slope: (EMA_i - EMA_{i-k}) / (k * ATR_i).

    Dimensionless drift per bar in ATR units. Positive = rising."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    return (ema_s - ema_s.shift(lookback)) / (lookback * atr_s)


# ---------------------------------------------------------------------------
# confirmed swings (pivot at j is only known at j+N -> no leakage)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SwingSeries:
    """Last *confirmed* swing highs/lows as of each bar (forward-filled)."""
    high_price: pd.Series   # price of last confirmed swing high (NaN if none)
    high_pos: pd.Series     # its bar position
    low_price: pd.Series
    low_pos: pd.Series


def confirmed_swings(high: pd.Series, low: pd.Series, n: int) -> SwingSeries:
    """Causal swing detection with pivot window ``n``.

    A pivot high at bar j (high[j] is max of [j-n, j+n]) becomes *confirmed*
    at bar j+n. Before confirmation it is invisible. Output series hold the
    most recently confirmed swing as of each bar.
    """
    if n < 1:
        raise ValueError("swing N must be >= 1")
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    m = len(h)
    conf_high = np.full(m, np.nan)
    conf_high_pos = np.full(m, -1)
    conf_low = np.full(m, np.nan)
    conf_low_pos = np.full(m, -1)
    last_hp, last_hpos = np.nan, -1
    last_lp, last_lpos = np.nan, -1
    # pivots confirmed at bar i sit at j = i - n
    for i in range(n, m - n):
        j = i
        confirm_at = j + n
        if h[j] >= h[j - n: j + n + 1].max():
            # record confirmation event; forward-fill below applies it from confirm_at
            conf_high[confirm_at] = h[j]
            conf_high_pos[confirm_at] = j
        if l[j] <= l[j - n: j + n + 1].min():
            conf_low[confirm_at] = l[j]
            conf_low_pos[confirm_at] = j
    out_hp = np.full(m, np.nan)
    out_hpos = np.full(m, -1)
    out_lp = np.full(m, np.nan)
    out_lpos = np.full(m, -1)
    for i in range(m):
        if not np.isnan(conf_high[i]):
            last_hp, last_hpos = conf_high[i], int(conf_high_pos[i])
        if not np.isnan(conf_low[i]):
            last_lp, last_lpos = conf_low[i], int(conf_low_pos[i])
        out_hp[i], out_hpos[i] = last_hp, last_hpos
        out_lp[i], out_lpos[i] = last_lp, last_lpos
    idx = high.index
    return SwingSeries(
        high_price=pd.Series(out_hp, index=idx),
        high_pos=pd.Series(out_hpos, index=idx),
        low_price=pd.Series(out_lp, index=idx),
        low_pos=pd.Series(out_lpos, index=idx),
    )


@dataclass(frozen=True)
class StructureState:
    higher_high: bool
    higher_low: bool
    lower_high: bool
    lower_low: bool


def market_structure(high: pd.Series, low: pd.Series, n: int,
                     lookback_swings: int = 6) -> pd.DataFrame:
    """HH/HL/LH/LL flags per bar from *confirmed* swings only.

    Compares the last two confirmed swing highs (rising = HH) and last two
    confirmed swing lows. ``lookback_swings`` is unused for the comparison
    itself (kept for API clarity) — the comparison always uses the two most
    recent confirmed pivots, both strictly in the past.
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    m = len(h)
    # collect confirmed pivot events: list of (confirm_bar, pivot_bar, price, kind)
    events = []
    for j in range(n, m - n):
        if h[j] >= h[j - n: j + n + 1].max():
            events.append((j + n, j, h[j], "high"))
        if l[j] <= l[j - n: j + n + 1].min():
            events.append((j + n, j, l[j], "low"))
    events.sort(key=lambda e: (e[0], e[1]))
    hh = np.zeros(m, dtype=bool)
    hl = np.zeros(m, dtype=bool)
    lh = np.zeros(m, dtype=bool)
    ll = np.zeros(m, dtype=bool)
    highs: list[float] = []
    lows: list[float] = []
    e = 0
    cur_hh = cur_hl = cur_lh = cur_ll = False
    for i in range(m):
        while e < len(events) and events[e][0] <= i:
            _, _, price, kind = events[e]
            if kind == "high":
                if len(highs) >= 1:
                    cur_hh = price > highs[-1]
                    cur_lh = price < highs[-1]
                highs.append(price)
            else:
                if len(lows) >= 1:
                    cur_hl = price > lows[-1]
                    cur_ll = price < lows[-1]
                lows.append(price)
            e += 1
        hh[i], hl[i], lh[i], ll[i] = cur_hh, cur_hl, cur_lh, cur_ll
    return pd.DataFrame({"HH": hh, "HL": hl, "LH": lh, "LL": ll}, index=high.index)


# ---------------------------------------------------------------------------
# HTF resampling with causal mapping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HTFFrame:
    frame: pd.DataFrame  # columns: bar_start, bar_end, open, high, low, close


def resample_htf(df_m15: pd.DataFrame, rule: str) -> HTFFrame:
    """Resample M15 (columns time/open/high/low/close) to H1/H4 completed bars.

    ``rule``: "h" (H1) or "4h" (H4), UTC-anchored. A bar is *completed* at
    ``bar_end``; the mapping below only exposes it to M15 bars ending at/after
    that instant.
    """
    if rule not in ("h", "4h"):
        raise ValueError("rule must be 'h' or '4h'")
    t = pd.to_datetime(df_m15["time"], utc=True)
    tmp = pd.DataFrame({
        "time": t,
        "open": df_m15["open"].to_numpy(dtype=float),
        "high": df_m15["high"].to_numpy(dtype=float),
        "low": df_m15["low"].to_numpy(dtype=float),
        "close": df_m15["close"].to_numpy(dtype=float),
    }).set_index("time").sort_index()
    agg = tmp.resample(rule, origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    agg = agg.dropna()
    bars = pd.DataFrame({
        "bar_start": agg.index,
        "bar_end": agg.index + pd.Timedelta(hours=1 if rule == "h" else 4),
        "open": agg["open"].to_numpy(),
        "high": agg["high"].to_numpy(),
        "low": agg["low"].to_numpy(),
        "close": agg["close"].to_numpy(),
    })
    return HTFFrame(bars.reset_index(drop=True))


def map_htf_to_m15(m15_time: pd.Series, htf: HTFFrame,
                   columns: Dict[str, pd.Series]) -> pd.DataFrame:
    """Map HTF indicator columns onto M15 bars (strictly causal).

    ``columns`` maps output names -> HTF-length series aligned with
    ``htf.frame``. Each M15 bar (ending at time+15min) receives the values of
    the latest HTF bar with ``bar_end <= m15_end``. M15 bars before the first
    completed HTF bar get NaN.
    """
    m15_end = pd.to_datetime(m15_time, utc=True) + pd.Timedelta(minutes=15)
    keys = pd.DataFrame({"m15_end": m15_end}).sort_values("m15_end")
    htf_keys = pd.DataFrame({"bar_end": pd.to_datetime(htf.frame["bar_end"], utc=True)})
    for name, ser in columns.items():
        htf_keys[name] = ser.to_numpy()
    htf_keys = htf_keys.sort_values("bar_end")
    merged = pd.merge_asof(keys, htf_keys, left_on="m15_end", right_on="bar_end",
                           direction="backward")
    merged.index = keys.index
    merged = merged.sort_index()
    return merged[[c for c in columns]].reset_index(drop=True)
