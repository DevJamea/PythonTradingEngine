"""Unit tests for swing levels and ATR/level SL-TP construction."""
from __future__ import annotations

import pytest

from gold_trader.models import Signal
from gold_trader.strategy.levels import (
    build_sl_tp,
    find_swing_points,
    nearest_resistance,
    nearest_support,
)
from tests._helpers import build_df, make_cfg


def _zigzag(prices):
    rows = []
    prev_close = prices[0]
    for price in prices[1:]:
        o = prev_close
        c = price
        rows.append((o, max(o, c), min(o, c), c))
        prev_close = c
    return build_df(rows)


LEVEL_DATA = [
    1998.0, 2001.0, 2004.5, 2007.2, 2004.9, 2004.2, 2005.5, 2006.8,
    2004.6, 2004.3, 2005.3, 2006.2, 2004.7, 2004.4, 2005.2,
]


def test_swing_points_found():
    df = _zigzag(LEVEL_DATA)
    points = find_swing_points(df, pivot_window=2)
    highs = sorted({p.price for p in points if p.kind == "high"})
    lows = sorted({p.price for p in points if p.kind == "low"})
    assert highs == [2006.2, 2006.8, 2007.2]
    assert lows == [2004.2, 2004.3]


def test_nearest_support_and_resistance():
    df = _zigzag(LEVEL_DATA)
    assert nearest_support(df, 2005.0, lookback=100, pivot_window=2) == pytest.approx(2004.3)
    assert nearest_resistance(df, 2005.0, lookback=100, pivot_window=2) == pytest.approx(2006.2)


def test_atr_only_sl_tp_defaults():
    cfg = make_cfg()  # levels disabled by default
    plan = build_sl_tp(Signal.BUY, entry=2005.0, atr_value=1.0, df=None, cfg=cfg)
    assert plan.sl == pytest.approx(2005.0 - 1.5)
    assert plan.tp == pytest.approx(2005.0 + 2.5)
    assert plan.sl_source == "atr"
    assert plan.tp_source == "atr"


def test_sell_atr_only():
    cfg = make_cfg()
    plan = build_sl_tp(Signal.SELL, entry=2005.0, atr_value=1.0, df=None, cfg=cfg)
    assert plan.sl == pytest.approx(2005.0 + 1.5)
    assert plan.tp == pytest.approx(2005.0 - 2.5)


def test_levels_refine_sl_and_tp():
    df = _zigzag(LEVEL_DATA)
    cfg = make_cfg(use_levels_for_sl=True, use_levels_for_tp=True, level_pivot_window=2)
    plan = build_sl_tp(Signal.BUY, entry=2005.0, atr_value=1.0, df=df, cfg=cfg)
    # support 2004.3 -> SL = 2004.3 - 0.1 (tighter than the ATR SL 2003.5)
    assert plan.sl == pytest.approx(2004.2)
    assert plan.sl_source == "support_buffer"
    # resistance 2006.2 -> TP capped at 2006.2 - 0.1
    assert plan.tp == pytest.approx(2006.1)
    assert plan.tp_source == "resistance_buffer"


def test_never_widen_sl():
    """Levels only tighten the stop; a farther level leaves the ATR SL in place."""
    df = _zigzag(LEVEL_DATA)
    cfg = make_cfg(use_levels_for_sl=True, use_levels_for_tp=True, level_pivot_window=2)
    # SELL: ATR SL 2006.5; resistance 2006.2 + 0.1 = 2006.3 (closer -> applied)
    plan = build_sl_tp(Signal.SELL, entry=2005.0, atr_value=1.0, df=df, cfg=cfg)
    assert plan.sl == pytest.approx(2006.3)
    assert plan.sl_source == "resistance_buffer"
    # BUY with a far support: ATR SL stays (2003.5 < support buffer 2004.2? no -
    # support is closer than the ATR stop, so it tightens; use a wide ATR to
    # prove the ATR stop wins when the level is farther away)
    wide = build_sl_tp(Signal.BUY, entry=2005.0, atr_value=3.0, df=df, cfg=cfg)
    # ATR SL = 2005 - 4.5 = 2000.5, support buffer = 2004.3 - 0.3 = 2004.0
    # 2004.0 is closer to entry -> tightens
    assert wide.sl == pytest.approx(2004.0)
    far = build_sl_tp(
        Signal.BUY, entry=2005.0, atr_value=3.0, df=df,
        cfg=make_cfg(use_levels_for_sl=True, level_pivot_window=2, atr_sl_multiplier=0.3),
    )
    # ATR SL = 2005 - 0.9 = 2004.1 is closer than support buffer 2004.0 -> ATR wins
    assert far.sl == pytest.approx(2004.1)
    assert far.sl_source == "atr"


def test_invalid_atr_raises():
    with pytest.raises(ValueError):
        build_sl_tp(Signal.BUY, entry=2005.0, atr_value=0.0)
    with pytest.raises(ValueError):
        build_sl_tp(Signal.BUY, entry=2005.0, atr_value=float("nan"))
    with pytest.raises(ValueError):
        build_sl_tp(Signal.NO_TRADE, entry=2005.0, atr_value=1.0)
