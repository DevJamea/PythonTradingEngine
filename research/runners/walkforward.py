"""Walk-forward validation + Gate 3 (§17-20).

Gate-2 survivors only, WF zone only. Frozen parameters — NO re-optimisation,
NO adaptive decisions (tracked as 0 in the ledger). The WF zone is tiled into
non-overlapping 2-month OOS windows; the frozen config runs on each window
(with an honest warmup prefix for indicator readiness that is never traded).

Per-window: PF, expectancy, trades, net, MaxDD. LOW_SAMPLE windows (<30
trades) are reported separately and excluded from WFE/win-ratio numerators
but never silently dropped.

WFE (predefined) = median valid-window OOS expectancy(R) / Development
expectancy(R). Gate 3: WFE >= 50% AND winning windows >= 60% AND best-window
profit concentration <= 50% AND >= 2 valid windows (predefined).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import (GATE3_MAX_PROFIT_CONCENTRATION, GATE3_MIN_WFE,
                      GATE3_MIN_WIN_WINDOW_RATIO, MASTER_SEED, WARMUP_BARS,
                      WF_LOW_SAMPLE_TRADES, WF_MIN_VALID_WINDOWS,
                      WF_OOS_MONTHS)
from ..data.splits import Segment, wf_oos_windows
from ..registry import Registry, TrialLedger
from .common import ConfigResult, Handle, run_configuration


@dataclass(frozen=True)
class WFWindowResult:
    window: str
    n_trades: int
    pf: float
    expectancy_r: float
    net_profit: float
    maxdd: float
    low_sample: bool
    exp_id: str

    def to_dict(self) -> Dict:
        return {"window": self.window, "n_trades": self.n_trades, "pf": self.pf,
                "expectancy_r": self.expectancy_r, "net_profit": self.net_profit,
                "maxdd": self.maxdd, "low_sample": self.low_sample, "exp_id": self.exp_id}


@dataclass(frozen=True)
class WFSummary:
    wfe: float
    win_window_ratio: float
    median_oos_expectancy: float
    median_oos_pf: float
    oos_trades_total: int
    concentration_best_window: float
    max_window_drawdown: float
    n_windows: int
    n_valid_windows: int
    n_low_sample_windows: int
    dev_expectancy_r: float

    def to_dict(self) -> Dict:
        return {"wfe": self.wfe, "win_window_ratio": self.win_window_ratio,
                "median_oos_expectancy": self.median_oos_expectancy,
                "median_oos_pf": self.median_oos_pf,
                "oos_trades_total": self.oos_trades_total,
                "concentration_best_window": self.concentration_best_window,
                "max_window_drawdown": self.max_window_drawdown,
                "n_windows": self.n_windows, "n_valid_windows": self.n_valid_windows,
                "n_low_sample_windows": self.n_low_sample_windows,
                "dev_expectancy_r": self.dev_expectancy_r}


@dataclass(frozen=True)
class Gate3Verdict:
    passed: bool
    reasons: List[str]


def _window_frame(full_df: pd.DataFrame, start_pos: int, end_pos: int) -> pd.DataFrame:
    """Slice [start_pos, end_pos) with a warmup prefix (never traded)."""
    w0 = max(0, start_pos - WARMUP_BARS)
    out = full_df.iloc[w0:end_pos].copy().reset_index(drop=True)
    out["_pos"] = range(w0, end_pos)
    out.attrs["trade_start_pos"] = start_pos
    return out


def run_walkforward(full_df: pd.DataFrame, wf_seg: Segment, primary: Handle,
                    dev_expectancy_r: float, registry: Registry,
                    ledger: TrialLedger) -> Tuple[List[WFWindowResult], WFSummary]:
    tradeable = full_df.iloc[wf_seg.start:wf_seg.end].copy()
    windows = wf_oos_windows(tradeable, WF_OOS_MONTHS)
    times = pd.to_datetime(full_df["time"], utc=True).to_numpy()
    results: List[WFWindowResult] = []
    for wi, (ws, we) in enumerate(windows):
        mask = (pd.to_datetime(tradeable["time"], utc=True) >= ws) & \
               (pd.to_datetime(tradeable["time"], utc=True) < we)
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) == 0:
            continue
        start_pos = wf_seg.start + int(idx[0])
        end_pos = wf_seg.start + int(idx[-1]) + 1
        wdf = _window_frame(full_df, start_pos, end_pos)
        h = Handle(primary.strategy, f"wf-window-{wi + 1}", primary.params)
        res: ConfigResult = run_configuration(
            h, wdf, f"walkforward-window-{wi + 1}", registry, ledger,
            gate="walkforward", seed=MASTER_SEED, count_ledger="none")
        m = res.metrics
        low = m.total_trades < WF_LOW_SAMPLE_TRADES
        results.append(WFWindowResult(
            window=f"{ws.date()}..{we.date()}", n_trades=m.total_trades,
            pf=(float(m.profit_factor) if np.isfinite(m.profit_factor) else 999.0),
            expectancy_r=m.expectancy_r, net_profit=m.net_profit,
            maxdd=m.max_drawdown, low_sample=low, exp_id=res.experiment_id))
        ledger.wf_decisions += 1

    valid = [r for r in results if not r.low_sample]
    n_valid = len(valid)
    med_exp = float(np.median([r.expectancy_r for r in valid])) if valid else 0.0
    med_pf = float(np.median([r.pf for r in valid])) if valid else 0.0
    wfe = (med_exp / dev_expectancy_r) if dev_expectancy_r > 0 else 0.0
    wins = sum(1 for r in valid if r.net_profit > 0)
    win_ratio = (wins / n_valid) if n_valid else 0.0
    total_net = sum(r.net_profit for r in valid)
    best = max([r.net_profit for r in valid], default=0.0)
    concentration = (best / total_net) if total_net > 0 else 1.0
    max_wdd = max([r.maxdd for r in valid], default=0.0)
    summary = WFSummary(
        wfe=wfe, win_window_ratio=win_ratio, median_oos_expectancy=med_exp,
        median_oos_pf=med_pf, oos_trades_total=sum(r.n_trades for r in results),
        concentration_best_window=concentration, max_window_drawdown=max_wdd,
        n_windows=len(results), n_valid_windows=n_valid,
        n_low_sample_windows=len(results) - n_valid,
        dev_expectancy_r=dev_expectancy_r)
    return results, summary


def evaluate_gate3(summary: WFSummary) -> Gate3Verdict:
    reasons = []
    if summary.n_valid_windows < WF_MIN_VALID_WINDOWS:
        reasons.append(f"only {summary.n_valid_windows} valid windows "
                       f"(need {WF_MIN_VALID_WINDOWS}; low-sample windows cannot pass)")
    if not (summary.wfe >= GATE3_MIN_WFE):
        reasons.append(f"WFE {summary.wfe:.2%} < {GATE3_MIN_WFE:.0%}")
    if not (summary.win_window_ratio >= GATE3_MIN_WIN_WINDOW_RATIO):
        reasons.append(f"winning windows {summary.win_window_ratio:.0%} "
                       f"< {GATE3_MIN_WIN_WINDOW_RATIO:.0%}")
    if not (summary.concentration_best_window <= GATE3_MAX_PROFIT_CONCENTRATION):
        reasons.append(f"best-window concentration {summary.concentration_best_window:.0%} "
                       f"> {GATE3_MAX_PROFIT_CONCENTRATION:.0%}")
    return Gate3Verdict(passed=not reasons, reasons=reasons)
