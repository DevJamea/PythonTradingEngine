"""Research tests: chronological splitting, WFO segmentation, low-trade windows."""
from __future__ import annotations

import pandas as pd

from research.config import WARMUP_BARS
from research.data.splits import (assert_holdout_unlocked, chronological_splits,
                                  holdout_locked, slice_segment, wf_oos_windows)
from research.data.synthetic import generate_synthetic_m15


def test_chronological_fractions_and_no_shuffle():
    df = generate_synthetic_m15(end="2022-03-31 23:45")
    n = len(df)
    assert n > WARMUP_BARS + 100
    s = chronological_splits(n)
    assert s.dev.start == 0
    assert s.dev.end == int(n * 0.50)
    assert s.wf.start == s.dev.end
    assert s.oos.start == s.wf.end
    assert s.oos.end == n
    # contiguity + order (chronological, never shuffled)
    assert s.dev.start < s.dev.end <= s.wf.end <= s.oos.end
    # fractions respected
    assert abs((s.dev.end - s.dev.start) / n - 0.50) < 0.01


def test_warmup_prefix_never_traded():
    df = generate_synthetic_m15(end="2022-03-31 23:45")
    s = chronological_splits(len(df))
    seg = slice_segment(df, s.wf)
    assert seg.attrs["trade_start_pos"] == s.wf.start
    assert seg["_pos"].iloc[0] == s.wf.warmup_start
    assert (seg["_pos"] < s.wf.start).sum() == min(WARMUP_BARS, s.wf.start)


def test_wf_windows_non_overlapping_and_cover_zone():
    df = generate_synthetic_m15()  # full 3y range -> WF zone ~9 months
    s = chronological_splits(len(df))
    tradeable = df.iloc[s.wf.start:s.wf.end].copy()
    wins = wf_oos_windows(tradeable, 2)
    assert len(wins) >= 2
    for (a0, a1), (b0, b1) in zip(wins, wins[1:]):
        assert a1 <= b0, "OOS windows must not overlap"
    # coverage: first starts at/below zone start month, last ends at zone end
    t0 = pd.to_datetime(tradeable["time"].iloc[0], utc=True)
    t1 = pd.to_datetime(tradeable["time"].iloc[-1], utc=True)
    assert wins[0][0] <= t0
    assert wins[-1][1] > t1


def test_holdout_lock_api_exists_and_blocks_without_receipt(tmp_path, monkeypatch):
    import research.data.splits as sp
    monkeypatch.setattr(sp, "FREEZE_PATH", tmp_path / "nope.json")
    assert holdout_locked() is True
    try:
        assert_holdout_unlocked()
    except PermissionError:
        pass
    else:
        raise AssertionError("locked holdout must raise PermissionError")
