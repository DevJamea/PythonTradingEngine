"""Daily cost controls for the scalping engine (pure, unit-testable).

A scalper's damage is not one bad trade, it is 40 trades x one spread. The live
loop therefore needs an entry *counter*, and a counter that cannot be read must
never be mistaken for "zero so far". Everything in this module is a pure
function over deal records so it can be tested without a terminal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ..strategy.scalping import SCALP_COMMENT_MARKER

#: Official MT5 ``DEAL_ENTRY_IN`` (an opening deal; OUT=1, INOUT=2, OUT_BY=3).
DEAL_ENTRY_IN = 0


@dataclass(frozen=True)
class DailyCapStatus:
    """Outcome of the daily scalp-entry gate."""

    allowed: bool
    entries_today: Optional[int]
    limit: int
    reason: str


def count_scalp_entries(
    deals: Optional[Iterable[object]],
    magic: int,
    entry_types: Sequence[int] = (DEAL_ENTRY_IN,),
    marker: str = SCALP_COMMENT_MARKER,
) -> int:
    """Count the bot's own *opening* scalp deals of today.

    Manual trades (different magic) and closing deals are ignored, so a
    partially closed scalp is never double-counted.
    """
    if not deals:
        return 0
    wanted = set(int(t) for t in entry_types)
    total = 0
    for deal in deals:
        try:
            if int(getattr(deal, "magic", -1)) != int(magic):
                continue
            if int(getattr(deal, "entry", -1)) not in wanted:
                continue
            if marker not in (getattr(deal, "comment", "") or ""):
                continue
        except (TypeError, ValueError):
            # A malformed deal record must not be counted as an entry, and it
            # must not crash the counter (which would block every later trade).
            continue
        total += 1
    return total


def scalp_entry_allowed(
    entries_today: Optional[int], limit: int, marker: str = SCALP_COMMENT_MARKER
) -> DailyCapStatus:
    """Fail-closed daily cap: unknown count => no new scalp entries today."""
    limit = int(limit)
    if limit < 1:
        return DailyCapStatus(False, entries_today, limit, "scalp_max_trades_per_day must be >= 1")
    if entries_today is None:
        return DailyCapStatus(
            False, None, limit, "today's scalp entry count is unknown (history unavailable)"
        )
    if entries_today >= limit:
        return DailyCapStatus(
            False,
            entries_today,
            limit,
            f"daily scalp cap reached ({entries_today} >= {limit})",
        )
    return DailyCapStatus(True, entries_today, limit, "ok")
