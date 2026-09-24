"""Unified execution-cost model (§5). Identical for every strategy.

Market BUY:  entry at Ask, exit at Bid.
Market SELL: entry at Bid, exit at Ask.

Bid/Ask per bar are derived from mid OHLC ± spread/2 (the only honest model
available: the synthetic proxy has no tick-level Bid/Ask, and any real CSV
without tick Bid/Ask uses its spread column the same way — the audit records
whichever provenance applies).

Commission and swap are NOT available and are NOT invented: they are recorded
as 0.0 with an explicit ``unavailable`` flag so totals stay honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import STRESS_SLIPPAGE_POINTS


@dataclass(frozen=True)
class BidAsk:
    bid_open: np.ndarray
    bid_high: np.ndarray
    bid_low: np.ndarray
    bid_close: np.ndarray
    ask_open: np.ndarray
    ask_high: np.ndarray
    ask_low: np.ndarray
    ask_close: np.ndarray
    spread: np.ndarray  # effective spread actually applied (after multiplier)


def build_bid_ask(df: pd.DataFrame, spread_mult: float = 1.0) -> BidAsk:
    """Build per-bar Bid/Ask OHLC from mid OHLC + spread column."""
    if spread_mult <= 0:
        raise ValueError("spread multiplier must be positive")
    spread = df["spread"].to_numpy(dtype=float) * spread_mult
    half = spread / 2.0
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    return BidAsk(o - half, h - half, l - half, c - half,
                  o + half, h + half, l + half, c + half, spread)


@dataclass(frozen=True)
class TradeCost:
    spread_paid: float      # price-distance paid via Bid/Ask on entry+exit
    slippage_paid: float    # adverse slippage distance (0 in baseline runs)
    commission: float       # always 0.0 — unavailable, NOT invented
    swap: float             # always 0.0 — unavailable/N-A, NOT invented
    total_distance: float   # spread_paid + slippage_paid (price units)

    @property
    def commission_unavailable(self) -> bool:
        return True

    @property
    def swap_unavailable(self) -> bool:
        return True


def round_trip_spread(entry_spread: float, exit_spread: float) -> float:
    """Spread distance paid for one round trip (half at entry, half at exit)."""
    return (entry_spread + exit_spread) / 2.0


def draw_slippage_points(rng: np.random.Generator, n: int,
                         lo: int = STRESS_SLIPPAGE_POINTS[0],
                         hi: int = STRESS_SLIPPAGE_POINTS[1]) -> np.ndarray:
    """Predefined stress distribution: uniform adverse 1..5 points per side."""
    return rng.integers(lo, hi + 1, size=n).astype(float)
