"""Research tests: causality / no-leakage of indicators, HTF mapping, swings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.data.synthetic import generate_synthetic_m15
from research.indicators_ext import (adx, atr, confirmed_swings, donchian,
                                     ema, map_htf_to_m15, resample_htf)


def _mini(n: int = 300) -> pd.DataFrame:
    return generate_synthetic_m15(end="2022-02-28 23:45").iloc[:n].copy()


class TestCausalIndicators:
    def test_ema_prefix_equivalence(self):
        df = _mini()
        full = ema(df["close"], 20)
        for cut in (50, 120, 299):
            part = ema(df["close"].iloc[: cut + 1], 20)
            assert full.iloc[cut] == pytest.approx(part.iloc[cut])

    def test_atr_prefix_equivalence(self):
        df = _mini()
        full = atr(df["high"], df["low"], df["close"], 14)
        for cut in (60, 200):
            part = atr(df["high"].iloc[: cut + 1], df["low"].iloc[: cut + 1],
                       df["close"].iloc[: cut + 1], 14)
            assert full.iloc[cut] == pytest.approx(part.iloc[cut])

    def test_adx_prefix_equivalence(self):
        df = _mini(500)
        full = adx(df["high"], df["low"], df["close"], 14)
        for cut in (100, 400):
            part = adx(df["high"].iloc[: cut + 1], df["low"].iloc[: cut + 1],
                       df["close"].iloc[: cut + 1], 14)
            a, b = full.iloc[cut], part.iloc[cut]
            assert (a == pytest.approx(b)) or (np.isnan(a) and np.isnan(b))

    def test_donchian_shifted_excludes_current_bar(self):
        df = _mini()
        hi, lo = donchian(df["high"], df["low"], 20, shift=True)
        i = 100
        assert hi.iloc[i] == pytest.approx(df["high"].iloc[i - 20:i].max())
        assert lo.iloc[i] == pytest.approx(df["low"].iloc[i - 20:i].min())


class TestHTFMapping:
    def test_h1_alignment_to_hour(self):
        df = generate_synthetic_m15(end="2022-01-31 23:45")
        h1 = resample_htf(df, "h").frame
        assert (h1["bar_start"].dt.minute == 0).all()
        assert (h1["bar_end"] - h1["bar_start"] == pd.Timedelta(hours=1)).all()

    def test_h4_alignment(self):
        df = generate_synthetic_m15(end="2022-01-31 23:45")
        h4 = resample_htf(df, "4h").frame
        assert ((h4["bar_end"] - h4["bar_start"]) == pd.Timedelta(hours=4)).all()

    def test_uncompleted_htf_bar_invisible(self):
        # HTF value for the in-progress H1 bar must NOT be visible to M15 bars
        # inside that same hour; only the previously completed bar is mapped.
        df = generate_synthetic_m15(end="2022-01-10 23:45")
        htf = resample_htf(df, "h")
        marker = pd.Series(np.arange(len(htf.frame), dtype=float))
        mapped = map_htf_to_m15(df["time"], htf, {"m": marker})
        m15_end = pd.to_datetime(df["time"], utc=True) + pd.Timedelta(minutes=15)
        h1_end = pd.to_datetime(htf.frame["bar_end"], utc=True)
        for k in range(0, len(df), 37):
            expect = h1_end[h1_end <= m15_end.iloc[k]]
            if len(expect) == 0:
                assert np.isnan(mapped["m"].iloc[k])
            else:
                assert mapped["m"].iloc[k] == len(expect) - 1

    def test_htf_prefix_equivalence(self):
        # mapping on a prefix equals the full mapping restricted to the prefix
        df = generate_synthetic_m15(end="2022-01-10 23:45")
        htf = resample_htf(df, "h")
        marker = pd.Series(np.arange(len(htf.frame), dtype=float))
        full = map_htf_to_m15(df["time"], htf, {"m": marker})
        cut = len(df) // 2
        htf_p = resample_htf(df.iloc[:cut], "h")
        marker_p = pd.Series(np.arange(len(htf_p.frame), dtype=float))
        part = map_htf_to_m15(df["time"].iloc[:cut], htf_p, {"m": marker_p})
        # all non-NaN prefix values must match the full mapping
        both = pd.DataFrame({"f": full["m"].iloc[:cut].to_numpy(),
                             "p": part["m"].to_numpy()})
        both = both.dropna()
        assert (both["f"] == both["p"]).all()


class TestConfirmedSwings:
    def test_pivot_invisible_before_confirmation(self):
        # clean V shape: pivot low at centre, N=2 -> confirmed 2 bars later
        n = 20
        closes = np.array([10 + abs(i - 10) * 0.5 for i in range(n)])  # V: low at 10
        highs = pd.Series(closes + 0.1)
        lows = pd.Series(closes - 0.1)
        # make bar 10 the strict low
        lows.iloc[10] = closes[10] - 1.0
        sw = confirmed_swings(highs, lows, 2)
        # before bar 12 (10+2) no swing low may reference bar 10
        assert (sw.low_pos.iloc[:12] != 10).all()
        assert sw.low_pos.iloc[12] == 10
        assert sw.low_price.iloc[12] == pytest.approx(lows.iloc[10])
