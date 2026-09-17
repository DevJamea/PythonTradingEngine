"""Unit tests for signal generation (deterministic synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_trader.models import Signal
from gold_trader.strategy.signals import generate_signal
from tests._helpers import (
    build_df,
    make_cfg,
    make_downtrend_with_engulfing,
    make_uptrend_with_engulfing,
)


def test_uptrend_engulfing_gives_buy():
    signal = generate_signal(make_uptrend_with_engulfing(), make_cfg())
    assert signal.signal is Signal.BUY
    assert signal.details["trend"] == "UPTREND"
    assert "bullish_engulfing" in signal.details["patterns"]
    assert signal.details["rsi"] < 70.0
    assert signal.details["atr"] > 0


def test_downtrend_engulfing_gives_sell():
    signal = generate_signal(make_downtrend_with_engulfing(), make_cfg())
    assert signal.signal is Signal.SELL
    assert signal.details["trend"] == "DOWNTREND"
    assert "bearish_engulfing" in signal.details["patterns"]
    assert signal.details["rsi"] > 30.0


def test_engulfing_without_uptrend_gives_no_trade():
    """Replace the engulfing candle with a small neutral one -> no buy."""
    df = make_uptrend_with_engulfing()
    price = df["close"].iloc[-2]
    rows = [
        (
            float(df["open"].iloc[i]),
            float(df["high"].iloc[i]),
            float(df["low"].iloc[i]),
            float(df["close"].iloc[i]),
        )
        for i in range(len(df) - 1)
    ]
    rows.append((price, price + 0.07, price - 0.07, price + 0.05))
    signal = generate_signal(build_df(rows), make_cfg())
    assert signal.signal is Signal.NO_TRADE


def test_flat_market_bounce_is_blocked_by_rsi_gate():
    """A bounce after a long flat market can align the EMAs, but the RSI
    overbought gate blocks the BUY (and no true engulfing exists either)."""
    rows = [(2000.0, 2000.1, 1999.9, 2000.0) for _ in range(248)]
    rows.append((2000.5, 2000.6, 1999.8, 1999.9))  # bearish
    rows.append((1999.8, 2000.6, 1999.7, 2000.4))  # bullish, not a true engulfing
    signal = generate_signal(build_df(rows), make_cfg())
    assert signal.signal is Signal.NO_TRADE
    assert "overbought" in signal.reason


def test_insufficient_candles_give_no_trade():
    df = build_df([(2000, 2001, 1999, 2000.5) for _ in range(50)])
    signal = generate_signal(df, make_cfg())
    assert signal.signal is Signal.NO_TRADE
    assert "insufficient" in signal.reason


def test_too_few_candles_raises():
    with pytest.raises(ValueError):
        generate_signal(build_df([(2000, 2001, 1999, 2000.5)]), make_cfg())


def test_missing_columns_raise():
    df = pd.DataFrame({"open": [1.0], "close": [2.0]})
    with pytest.raises(ValueError):
        generate_signal(df, make_cfg())


def test_no_lookahead_between_window_and_precomputed():
    """Signal on the visible window == signal from precomputed indicators.

    Proves the causal precomputation used by the backtest engine cannot
    leak future data.
    """
    rng = np.random.default_rng(123)
    values = 2000.0 + rng.normal(0.0, 0.5, 320).cumsum()
    rows = []
    prev_close = 2000.0
    for value in values:
        o = prev_close
        c = value
        rows.append((o, max(o, c) + 0.1, min(o, c) - 0.1, c))
        prev_close = c
    df = build_df(rows)

    from gold_trader.strategy.signals import compute_indicators, evaluate_at

    ind = compute_indicators(df, make_cfg())
    for i in (215, 240, 300):
        window_signal = generate_signal(df.iloc[: i + 1].reset_index(drop=True), make_cfg())
        precomputed_signal = evaluate_at(i, df, ind, make_cfg())
        assert window_signal.signal == precomputed_signal.signal
        assert window_signal.details["trend"] == precomputed_signal.details["trend"]
        assert window_signal.details["rsi"] == pytest.approx(
            precomputed_signal.details["rsi"], rel=1e-9
        )
        assert window_signal.details["atr"] == pytest.approx(
            precomputed_signal.details["atr"], rel=1e-9
        )
