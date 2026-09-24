"""Research tests: Bid/Ask cost model, spread stress, slippage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.costs import (TradeCost, build_bid_ask, draw_slippage_points,
                            round_trip_spread)
from research.data.synthetic import generate_synthetic_m15


def test_bid_ask_sides():
    df = generate_synthetic_m15(end="2022-01-10 23:45")
    ba = build_bid_ask(df)
    o = df["open"].to_numpy(float)
    s = df["spread"].to_numpy(float)
    assert ba.ask_open == pytest.approx(o + s / 2)
    assert ba.bid_open == pytest.approx(o - s / 2)
    assert (ba.ask_open >= ba.bid_open).all()


def test_buy_entry_ask_exit_bid_sell_mirror():
    # engine-level direction check lives in test_research_engine; here we pin
    # the cost-model invariant: round-trip spread = (entry+exit spread)/2.
    assert round_trip_spread(0.30, 0.30) == pytest.approx(0.30)
    assert round_trip_spread(0.20, 0.40) == pytest.approx(0.30)


def test_spread_multiplier_scales_linearly():
    df = generate_synthetic_m15(end="2022-01-10 23:45")
    base = build_bid_ask(df, 1.0)
    stressed = build_bid_ask(df, 2.0)
    assert stressed.spread == pytest.approx(base.spread * 2.0)
    with pytest.raises(ValueError):
        build_bid_ask(df, 0.0)


def test_slippage_distribution_and_seed():
    rng1 = np.random.default_rng(131415)
    rng2 = np.random.default_rng(131415)
    a = draw_slippage_points(rng1, 1000)
    b = draw_slippage_points(rng2, 1000)
    assert (a == b).all()
    assert a.min() >= 1 and a.max() <= 5
    assert set(np.unique(a)) == {1.0, 2.0, 3.0, 4.0, 5.0}


def test_trade_cost_unavailable_flags():
    c = TradeCost(0.3, 0.0, 0.0, 0.0, 0.3)
    assert c.commission_unavailable and c.swap_unavailable
    assert c.total_distance == pytest.approx(0.3)
