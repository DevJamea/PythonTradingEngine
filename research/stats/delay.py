"""Random execution-delay stress (§26) — documented PROXY.

M15 close-based bars cannot model sub-candle fills, queue position or
tick-level latency with any precision. Rather than pretending otherwise, the
predefined delay model is an explicit PROXY: 0s = no change, 1s = +0.25x the
entry-bar spread adverse, 2s = +0.5x spread adverse, applied at entry only
(trade count unchanged by construction in this model — stated, not hidden).

Measures PF, expectancy, trade count and MaxDD per delay level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..config import DELAY_PROXY_SPREAD_FRAC


@dataclass(frozen=True)
class DelayResult:
    levels: Dict[int, Dict[str, float]]
    limitation: str = ("M15 bars cannot resolve sub-candle execution; delay is "
                       "modelled as adverse entry slippage (0.25x/0.5x spread). "
                       "Signals are identical so trade count is normally "
                       "unchanged; any sizing-fail edge cases are reported.")

    def to_dict(self) -> Dict:
        return {"levels": self.levels, "limitation": self.limitation}


def delay_levels() -> Dict[int, float]:
    return dict(DELAY_PROXY_SPREAD_FRAC)
