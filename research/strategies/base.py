"""Common research-strategy contracts (no production behaviour here)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchSignal:
    """A raw entry signal on closed bar ``index`` (position in the passed frame).

    The backtest engine converts it into a priced trade: entry at the open of
    ``entry_index`` on the correct Bid/Ask side. ``sl_distance``/``tp_distance``
    are price distances; the engine derives SL/TP prices from the actual fill.
    """

    index: int            # signal bar (closed) — full-frame position via df["_pos"]
    entry_index: int      # entry bar (usually index+1; B uses index+2 after confirmation)
    direction: str        # "BUY" | "SELL"
    sl_distance: float
    tp_distance: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyOutput:
    signals: List[ResearchSignal]
    diagnostics: Dict[str, Any]  # evaluated bars, filter pass counts, regime mix...


def inside_session(time: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> bool:
    """UTC session check (inclusive start, exclusive end, overnight aware)."""
    sh, sm = int(start_hhmm[:2]), int(start_hhmm[3:5])
    eh, em = int(end_hhmm[:2]), int(end_hhmm[3:5])
    minutes = time.hour * 60 + time.minute
    lo, hi = sh * 60 + sm, eh * 60 + em
    if lo <= hi:
        return lo <= minutes < hi
    return minutes >= lo or minutes < hi
