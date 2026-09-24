"""Research tests: engine exits, management, R accounting, skip counting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_trader.models import default_gold_spec
from research.backtest.engine import ManagementSpec, run_backtest
from research.costs import build_bid_ask
from research.strategies.base import ResearchSignal

SPEC = default_gold_spec()


def _df(rows, spread: float = 0.30):
    n = len(rows)
    return pd.DataFrame({
        "time": pd.date_range("2022-01-03", periods=n, freq="15min", tz="UTC"),
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "tick_volume": [100] * n, "spread": [spread] * n,
    })


def _sig(i, entry, direction="BUY", sl=2.0, tp=4.0):
    return ResearchSignal(index=i, entry_index=entry, direction=direction,
                          sl_distance=sl, tp_distance=tp, meta={"params": "X"})


class TestExits:
    def test_buy_enters_ask_and_hits_tp_on_bid(self):
        df = _df([(2000, 2001, 1999, 2000)] +           # 0 signal bar
                 [(2000, 2006, 1999.5, 2005)] +          # 1 entry bar, TP hit
                 [(2005, 2006, 2004, 2005)] * 3)
        ba = build_bid_ask(df)
        res = run_backtest(df, [_sig(0, 1)], ba, SPEC, 10_000.0, 0.005,
                           ManagementSpec(), 0)
        assert len(res.trades) == 1
        t = res.trades[0]
        assert t.side == "BUY"
        assert t.entry == pytest.approx(2000 + 0.15)  # ask open
        assert t.reason == "take_profit"
        assert t.r_net > 0

    def test_sl_first_on_ambiguity(self):
        df = _df([(2000, 2001, 1999, 2000)] +
                 [(2000, 2010, 1990, 2005)] +  # touches SL and TP same bar
                 [(2005, 2006, 2004, 2005)] * 3)
        ba = build_bid_ask(df)
        res = run_backtest(df, [_sig(0, 1, sl=2.0, tp=4.0)], ba, SPEC,
                           10_000.0, 0.005, ManagementSpec(), 0)
        assert res.trades[0].reason == "stop_loss"

    def test_sell_mirror(self):
        df = _df([(2000, 2001, 1999, 2000)] +
                 [(2000, 2000.5, 1994, 1995)] +
                 [(1995, 1996, 1994, 1995)] * 3)
        ba = build_bid_ask(df)
        res = run_backtest(df, [_sig(0, 1, "SELL")], ba, SPEC, 10_000.0,
                           0.005, ManagementSpec(), 0)
        t = res.trades[0]
        assert t.entry == pytest.approx(2000 - 0.15)  # bid open
        assert t.reason == "take_profit"

    def test_end_of_data_close(self):
        df = _df([(2000, 2001, 1999, 2000)] + [(2000, 2001, 1999, 2000)] * 4)
        ba = build_bid_ask(df)
        res = run_backtest(df, [_sig(0, 1, sl=50.0, tp=50.0)], ba, SPEC,
                           10_000.0, 0.005, ManagementSpec(), 0)
        assert res.trades[0].reason == "end_of_data"


class TestManagement:
    def test_partial_and_breakeven(self):
        # BUY: entry ~2000.15, SL 2 below, 1R level = entry+2; then falls back
        df = _df([(2000, 2001, 1999, 2000)] +
                 [(2000, 2000.5, 1999.9, 2000.2)] +   # entry, nothing
                 [(2000.2, 2003.5, 2000.0, 2003.0)] +  # reaches 1R -> partial+BE
                 [(2003.0, 2003.2, 1999.0, 1999.5)] +  # falls to BE stop
                 [(1999.5, 2000, 1999, 1999.5)])
        ba = build_bid_ask(df)
        m = ManagementSpec(use_fixed_tp=True, be_trigger_r=1.0, be_buffer=0.10,
                           partial_r=1.0, partial_fraction=0.5)
        res = run_backtest(df, [_sig(0, 1, sl=2.0, tp=10.0)], ba, SPEC,
                           10_000.0, 0.005, m, 0)
        t = res.trades[0]
        kinds = [f.reason for f in t.fills]
        assert "partial" in kinds
        assert t.reason == "stop_loss"  # remainder stopped at BE
        assert t.volume_initial > 0

    def test_trailing_moves_stop_only_forward(self):
        df = _df([(2000, 2001, 1999, 2000)] +
                 [(2000, 2001, 1999.9, 2000.5)] +
                 [(2000.5, 2005, 2000.4, 2004.5)] +  # big up -> trail up
                 [(2004.5, 2004.6, 2003.0, 2003.5)] +
                 [(2003.5, 2003.6, 2000.0, 2000.5)])  # falls -> trail stop
        ba = build_bid_ask(df)
        atr_arr = np.full(len(df), 1.0)
        m = ManagementSpec(use_fixed_tp=False, be_trigger_r=1.0, be_buffer=0.0,
                           trail_mult=1.0)
        res = run_backtest(df, [_sig(0, 1, sl=2.0, tp=100.0)], ba, SPEC,
                           10_000.0, 0.005, m, 0, atr_for_trail=atr_arr)
        t = res.trades[0]
        assert t.reason == "stop_loss"
        assert t.exit_price > t.sl_initial  # stop was trailed up, never down


class TestAccounting:
    def test_one_position_and_skip_counting(self):
        df = _df([(2000, 2001, 1999, 2000)] * 6)
        ba = build_bid_ask(df)
        sigs = [_sig(0, 1, sl=50, tp=50), _sig(1, 2, sl=50, tp=50), _sig(2, 3, sl=50, tp=50)]
        res = run_backtest(df, sigs, ba, SPEC, 10_000.0, 0.005,
                           ManagementSpec(), 0)
        assert len(res.trades) == 1  # first holds to end of data
        assert res.diagnostics.signals_skipped_in_position == 2

    def test_warmup_signals_discarded(self):
        df = _df([(2000, 2001, 1999, 2000)] * 6)
        df["_pos"] = range(6)
        ba = build_bid_ask(df)
        res = run_backtest(df, [_sig(0, 1)], ba, SPEC, 10_000.0, 0.005,
                           ManagementSpec(), trade_start_pos=4)
        assert len(res.trades) == 0
        assert res.diagnostics.signals_warmup_discarded == 1

    def test_r_multiple_net_of_costs(self):
        df = _df([(2000, 2001, 1999, 2000)] +
                 [(2000, 2006, 1999.5, 2005)] +
                 [(2005, 2006, 2004, 2005)] * 3, spread=0.0)
        ba = build_bid_ask(df)  # zero spread -> clean R math
        res = run_backtest(df, [_sig(0, 1, sl=2.0, tp=4.0)], ba, SPEC,
                           10_000.0, 0.005, ManagementSpec(), 0)
        t = res.trades[0]
        assert t.r_net == pytest.approx(2.0, rel=0.02)  # RR 2.0 winner
        assert t.cost_money == pytest.approx(0.0, abs=1e-6)
