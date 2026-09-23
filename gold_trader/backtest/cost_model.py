"""Execution-cost model shared by the scalping backtest (and any future study).

Why this exists: for a scalper the *cost of doing the trade* is the dominant
term, not the entry signal. A single fixed "spread per side" number hides
exactly the effect that decides whether the strategy can win, so the cost is
modelled here explicitly and per bar:

* ``spread``      -- ``ask - bid`` in price units. One round trip (buy at ask,
                     sell at bid, or the mirror) costs exactly ONE spread.
* ``stress``      -- a multiplier applied to every spread value
                     (``Config.backtest_spread_stress_multiplier``). Brokers
                     widen gold spreads around news, rollover and thin hours;
                     a scalping study without this multiplier is not credible.
* ``commission``  -- per lot PER SIDE, so a round trip pays twice.
* ``slippage``    -- extra price distance per side applied on top of the
                     planned fill.

The legacy engine (``gold_trader.backtest.engine``) charges
``2 x backtest_spread_cost`` per round trip, i.e. it treats the configured
number as a per-side cost. ``fixed_round_trip_spread()`` reproduces that
convention so the two engines stay comparable on paper; the scalping engine
uses the *same* number as its conservative floor even in variable-spread mode,
so a study can never come out rosier than the old one just because the cost
model changed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..models import SymbolSpec
from ..risk.position_size import profit_for_volume


def fixed_round_trip_spread(cfg: Config) -> float:
    """Round-trip spread implied by the legacy per-side config value."""
    return 2.0 * float(cfg.backtest_spread_cost)


@dataclass(frozen=True)
class CostModel:
    """Per-bar round-trip cost of one trade, in price units and currency.

    ``variable_spread`` holds the real ``ask - bid`` distance per bar (index
    aligned with the candle frame). When it is ``None`` the model falls back
    to the conservative fixed legacy spread, never to zero.
    """

    variable_spread: Optional[np.ndarray] = None
    fixed_spread: float = 0.30
    stress_multiplier: float = 1.0
    slippage_per_side: float = 0.0
    commission_per_lot: float = 0.0

    # -- construction ------------------------------------------------------
    @classmethod
    def from_frame(cls, df: pd.DataFrame, cfg: Config, stress_multiplier: Optional[float] = None) -> "CostModel":
        """Build from a candle frame: uses its ``spread`` column when
        ``Config.backtest_use_recorded_spread`` is on (it is off by default, so
        the behaviour stays identical to the original fixed-spread engine).

        The fixed legacy spread is always kept as a floor: if the recorded
        spread of a bar is *narrower* than ``2 x backtest_spread_cost`` the
        model still charges the configured estimate. Optimistic data must not
        silently buy a better fill than the one we agreed to assume.
        """
        spread: Optional[np.ndarray] = None
        if getattr(cfg, "backtest_use_recorded_spread", False) and "spread" in df.columns:
            values = pd.to_numeric(df["spread"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).any():
                spread = values
        return cls(
            variable_spread=spread,
            fixed_spread=fixed_round_trip_spread(cfg),
            stress_multiplier=(
                float(cfg.backtest_spread_stress_multiplier)
                if stress_multiplier is None
                else float(stress_multiplier)
            ),
            slippage_per_side=float(cfg.backtest_slippage_per_side),
            commission_per_lot=float(cfg.backtest_commission_per_lot),
        )

    # -- queries -----------------------------------------------------------
    def spread_at(self, index: int) -> float:
        """Charged spread (price units) for a trade opened at bar ``index``."""
        raw = self.fixed_spread
        if self.variable_spread is not None and 0 <= index < len(self.variable_spread):
            value = float(self.variable_spread[index])
            if math.isfinite(value) and value > 0:
                # conservative floor: never cheaper than the fixed estimate
                raw = max(value, self.fixed_spread)
        return raw * self.stress_multiplier

    def round_trip_price_cost(self, index: int) -> float:
        """Total price-distance cost: spread (1x) + slippage (2x)."""
        return self.spread_at(index) + 2.0 * float(self.slippage_per_side)

    def round_trip_currency_cost(self, index: int, volume: float, spec: SymbolSpec) -> float:
        """Money cost of one round trip at bar ``index`` for ``volume`` lots."""
        price_cost = self.round_trip_price_cost(index)
        cost = 0.0
        if price_cost > 0:
            cost += profit_for_volume(price_cost, volume, spec) or 0.0
        cost += self.commission_currency(index, volume)
        return cost

    def commission_currency(self, index: int, volume: float) -> float:
        """Commission for one round trip (per side, so x2) in currency."""
        return 2.0 * float(self.commission_per_lot) * float(volume)

    def slippage_price(self) -> float:
        """Slippage charged per side (applied to the entry fill)."""
        return float(self.slippage_per_side)

    def describe(self) -> str:
        """One-line, human-readable summary for reports and logs."""
        mode = "real per-bar spread" if self.variable_spread is not None else "fixed"
        return (
            f"cost: {mode} (floor {self.fixed_spread:.5f}/round trip), "
            f"stress x{self.stress_multiplier:.2f}, "
            f"slippage {self.slippage_per_side:.5f}/side, "
            f"commission {self.commission_per_lot:.2f}/lot/side"
        )


def spread_summary(spread: Sequence[float] | pd.Series) -> dict:
    """Descriptive statistics of a spread series (for the reality report)."""
    values = pd.Series(list(spread) if not isinstance(spread, pd.Series) else spread)
    values = pd.to_numeric(values, errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }
