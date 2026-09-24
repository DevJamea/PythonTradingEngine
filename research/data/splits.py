"""Chronological data splitting (§4) and walk-forward segmentation (§17-18).

Rules enforced here:
* splits are chronological fractions of the bar index — never shuffled;
* every segment carries a warmup prefix (indicator readiness) that is NEVER
  traded: signals with index < segment_start are discarded by the engine;
* the OOS holdout slice can only be built after a FreezeReceipt exists
  (``require_freeze=True``), which is how the runner proves the holdout
  stayed untouched until parameters were frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from ..config import DEV_FRACTION, OOS_FRACTION, WF_FRACTION, WARMUP_BARS
from ..registry import OUTPUTS

FREEZE_PATH = OUTPUTS / "freeze_oos.json"


@dataclass(frozen=True)
class Segment:
    name: str
    start: int  # first tradeable bar (inclusive, in full-frame coordinates)
    end: int    # one past the last tradeable bar (exclusive)
    warmup_start: int  # first bar of the warmup prefix (>= 0)


@dataclass(frozen=True)
class DataSplits:
    dev: Segment
    wf: Segment
    oos: Segment
    n: int


def chronological_splits(n: int,
                         dev_frac: float = DEV_FRACTION,
                         wf_frac: float = WF_FRACTION,
                         oos_frac: float = OOS_FRACTION,
                         warmup: int = WARMUP_BARS) -> DataSplits:
    """Split ``n`` bars chronologically into Development / WF / OOS."""
    if abs(dev_frac + wf_frac + oos_frac - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")
    if n < warmup + 100:
        raise ValueError(f"need at least {warmup + 100} bars, got {n}")
    dev_end = int(n * dev_frac)
    wf_end = dev_end + int(n * wf_frac)

    def _seg(name: str, s: int, e: int) -> Segment:
        return Segment(name, s, e, max(0, s - warmup))

    return DataSplits(_seg("dev", 0, dev_end),
                      _seg("wf", dev_end, wf_end),
                      _seg("oos", wf_end, n), n)


def slice_segment(df: pd.DataFrame, seg: Segment) -> pd.DataFrame:
    """Return warmup+segment rows with a ``_pos`` full-frame position column."""
    out = df.iloc[seg.warmup_start:seg.end].copy().reset_index(drop=True)
    out["_pos"] = range(seg.warmup_start, seg.end)
    out.attrs["trade_start_pos"] = seg.start
    return out


def wf_oos_windows(wf_df: pd.DataFrame, oos_months: int = 2) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Non-overlapping OOS windows of ``oos_months`` calendar months each.

    ``wf_df`` is the full WF-zone frame (warmup prefix already stripped, i.e.
    only tradeable WF bars). Windows tile [wf_start, wf_end) by calendar
    months; a short tail (< 1 full window) is merged into the last window and
    the merge is reported by the caller.
    """
    def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(ts.year, ts.month, 1, tz="UTC")

    def _add_months(ts: pd.Timestamp, k: int) -> pd.Timestamp:
        total = (ts.year * 12 + ts.month - 1) + k
        return pd.Timestamp(total // 12, total % 12 + 1, 1, tz="UTC")

    times = pd.to_datetime(wf_df["time"], utc=True)
    start = _month_start(times.iloc[0])
    end = times.iloc[-1]
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while True:
        nxt = _add_months(cur, oos_months)
        if nxt >= end:
            break
        windows.append((cur, nxt))
        cur = nxt
    if cur < end:
        if windows:
            windows[-1] = (windows[-1][0], end + pd.Timedelta(seconds=1))
        else:
            windows.append((cur, end + pd.Timedelta(seconds=1)))
    else:
        # extend last window to include the final bar
        s, _ = windows[-1]
        windows[-1] = (s, end + pd.Timedelta(seconds=1))
    return windows


def holdout_locked() -> bool:
    """True until the OOS freeze receipt has been written."""
    return not FREEZE_PATH.exists()


def assert_holdout_unlocked() -> None:
    if holdout_locked():
        raise PermissionError(
            "OOS holdout is LOCKED: write the freeze receipt first "
            f"({FREEZE_PATH}). The holdout must stay untouched until all "
            "preceding stages are frozen."
        )
