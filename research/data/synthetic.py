"""Synthetic XAUUSD M15 proxy generator.

STATUS: SYNTHETIC — NOT real market data. Every output frame is stamped with
``source = SYNTHETIC_PROXY_v1``. This generator exists ONLY because the
sandbox has no MT5 terminal and no network access to market-data vendors, so
no real broker history can be obtained here. It lets us validate the full
research pipeline end-to-end; any numerical result on this proxy is a
pipeline-validation result, NEVER evidence about real-market edge. The real
experiment must be re-run on broker Bid/Ask history via ``loader.py``.

Model (documented, seeded, neutral by construction):
* zero-drift random walk (no planted edge, no tradable drift);
* GARCH(1,1)-style volatility clustering calibrated to roughly gold-like M15
  volatility (sigma_bar ~ 0.035% of price, vol-of-vol via GARCH persistence);
* intraday session multipliers (Asia quiet, London/NY busy);
* rare jumps (0.15% of bars, 3-6 sigma) for fat tails;
* OHLC built from open + close with wick extensions;
* weekend bars (Sat/Sun) excluded entirely (FX/metals market closed);
* synthetic spread series: base ~0.30 price units with lognormal noise,
  wider in quiet hours + rare spikes. Spread is SYNTHETIC, not broker data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (SEED_DATA, SYNTH_END, SYNTH_START, SYNTH_START_PRICE,
                      SYNTHETIC_TAG)

SOURCE_TAG = SYNTHETIC_TAG


def m15_weekday_grid(start: str, end: str) -> pd.DatetimeIndex:
    """Full M15 grid 00:00-23:45 UTC on Mon-Fri only (no weekends)."""
    full = pd.date_range(start=start, end=end, freq="15min", tz="UTC")
    mask = full.dayofweek < 5  # Monday=0 .. Friday=4
    return full[mask]


def _session_multiplier(hours: np.ndarray) -> np.ndarray:
    """Intraday volatility seasonality by UTC hour (documented, fixed)."""
    mult = np.ones_like(hours, dtype=float)
    # Asia (0-7): quiet
    mult[(hours >= 0) & (hours < 7)] = 0.70
    # London morning (7-12): busy
    mult[(hours >= 7) & (hours < 12)] = 1.20
    # London/NY overlap (12-16): busiest
    mult[(hours >= 12) & (hours < 16)] = 1.40
    # NY afternoon (16-21): moderate
    mult[(hours >= 16) & (hours < 21)] = 1.00
    # late NY / rollover (21-24): quiet
    mult[hours >= 21] = 0.75
    return mult


def generate_synthetic_m15(seed: int = SEED_DATA,
                           start: str = SYNTH_START,
                           end: str = SYNTH_END,
                           start_price: float = SYNTH_START_PRICE) -> pd.DataFrame:
    """Generate the neutral synthetic XAUUSD M15 proxy (seeded)."""
    rng = np.random.default_rng(seed)
    idx = m15_weekday_grid(start, end)
    n = len(idx)
    hours = idx.hour.to_numpy()

    # --- GARCH(1,1)-style variance process (stationary, zero drift) ---
    omega = 0.20
    alpha = 0.10
    beta = 0.85  # alpha + beta = 0.95 < 1 -> stationary
    base_sigma = 0.00035  # per-bar sigma at unit variance (~$0.70 at $2000)
    sess = _session_multiplier(hours)
    z = rng.standard_normal(n)
    var = np.ones(n)
    for i in range(1, n):
        var[i] = omega * 0.05 + alpha * (z[i - 1] ** 2) + beta * var[i - 1]
    var = np.clip(var, 0.05, 12.0)
    sigma = base_sigma * np.sqrt(var) * sess
    rets = sigma * z  # zero drift by construction

    # --- rare jumps (fat tails) ---
    jump_mask = rng.random(n) < 0.0015
    jump_dir = rng.choice([-1.0, 1.0], size=n)
    jump_mag = rng.uniform(3.0, 6.0, size=n) * sigma
    rets = rets + jump_mask * jump_dir * jump_mag

    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # --- wicks: extend high/low beyond body by half-normal noise ---
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick_scale = sigma * close  # price units
    high = body_hi + np.abs(rng.standard_normal(n)) * wick_scale * 0.9
    low = body_lo - np.abs(rng.standard_normal(n)) * wick_scale * 0.9
    # weekend-gap reopen noise is zero by construction (documented limitation:
    # no weekend gaps modelled; Friday close == next Monday open).

    # --- synthetic spread (price units), NOT broker data ---
    spread_base = 0.30
    spread_noise = rng.lognormal(mean=0.0, sigma=0.35, size=n)
    quiet_wide = np.where(sess < 1.0, 1.5, 1.0)
    spread = spread_base * spread_noise * quiet_wide
    spike = rng.random(n) < 0.002
    spread = np.where(spike, spread * rng.uniform(2.0, 4.0, size=n), spread)
    spread = np.clip(spread, 0.10, 3.00)
    spread = np.round(spread, 2)

    tick_volume = rng.integers(20, 400, size=n)

    df = pd.DataFrame({
        "time": idx,
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "tick_volume": tick_volume,
        "spread": spread,
    })
    # guarantee OHLC consistency after rounding
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df.attrs["source"] = SOURCE_TAG
    df.attrs["seed"] = seed
    return df.reset_index(drop=True)
