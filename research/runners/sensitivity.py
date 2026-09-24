"""Sensitivity analysis + Gate 2 (§13-16).

Gate-1 survivors only, Development data only. For each declared free
parameter: one-at-a-time (OAT) neighbours from the predefined grid. Plus a 2D
heatmap over the two most important interacting parameters.

Gate-2 rule (predefined plateau rule, research/config.py):
* >=40% of OAT neighbours keep expectancy > 0;
* >=40% of OAT neighbours keep PF > 1;
* best PF <= 2x median neighbour PF (no isolated spike);
* heatmap largest connected PF>1 region >= 3 cells (4-connectivity).
No rescue, no re-tuning: failure => REJECTED — GATE 2.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import (GATE2_MAX_ISOLATION_RATIO, GATE2_MIN_CONNECTED_REGION,
                      GATE2_MIN_PF_GT1_FRAC, GATE2_MIN_POSITIVE_EXPECTANCY_FRAC,
                      MASTER_SEED, SENSITIVITY_GRIDS)
from ..registry import Registry, TrialLedger
from .common import ConfigResult, Handle, run_configuration, with_params


@dataclass(frozen=True)
class SensitivitySummary:
    oat_runs: int
    frac_positive_exp: float
    frac_pf_gt1: float
    best_pf: float
    median_neighbour_pf: float
    isolation_ratio: float
    largest_connected_pf_gt1: int
    heatmap_cells: int
    heatmap_pf_gt1_cells: int

    def to_dict(self) -> Dict:
        return {
            "oat_runs": self.oat_runs,
            "frac_positive_exp": self.frac_positive_exp,
            "frac_pf_gt1": self.frac_pf_gt1,
            "best_pf": self.best_pf, "median_neighbour_pf": self.median_neighbour_pf,
            "isolation_ratio": self.isolation_ratio,
            "largest_connected_pf_gt1": self.largest_connected_pf_gt1,
            "heatmap_cells": self.heatmap_cells,
            "heatmap_pf_gt1_cells": self.heatmap_pf_gt1_cells,
        }


@dataclass(frozen=True)
class Gate2Verdict:
    passed: bool
    reasons: List[str]


def _pf_of(res: ConfigResult) -> float:
    pf = res.metrics.profit_factor
    return float(pf) if np.isfinite(pf) else (999.0 if pf > 0 else 0.0)


def largest_connected(mask: np.ndarray) -> int:
    """Largest 4-connected True region in a 2D boolean array."""
    m, n = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    for i in range(m):
        for j in range(n):
            if not mask[i, j] or seen[i, j]:
                continue
            q = deque([(i, j)])
            seen[i, j] = True
            size = 0
            while q:
                x, y = q.popleft()
                size += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            best = max(best, size)
    return best


def _baseline_value(primary: Handle, param: str):
    return getattr(primary.params, param)


def run_sensitivity(primary: Handle, dev_df: pd.DataFrame, registry: Registry,
                    ledger: TrialLedger) -> Tuple[List[ConfigResult],
                                                  pd.DataFrame, SensitivitySummary]:
    """OAT sweep + heatmap for one surviving strategy. Returns runs, heatmap frame, summary."""
    grid = SENSITIVITY_GRIDS[primary.strategy]
    oat: List[ConfigResult] = []
    for param, values in grid.items():
        if param == "heatmap":
            continue
        base = _baseline_value(primary, param)
        for v in values:
            if v == base:
                continue  # baseline point already run at Gate 1
            h = with_params(primary, f"sens:{param}={v}", **{param: v})
            oat.append(run_configuration(h, dev_df, "development", registry,
                                         ledger, gate="sensitivity",
                                         seed=MASTER_SEED, count_ledger="sens"))
    # --- heatmap over the two declared interacting params ---
    px, py = grid["heatmap"]
    vx, vy = grid[px], grid[py]
    rows = []
    for xv in vx:
        for yv in vy:
            h = with_params(primary, f"heat:{px}={xv}|{py}={yv}", **{px: xv, py: yv})
            r = run_configuration(h, dev_df, "development", registry, ledger,
                                  gate="heatmap", seed=MASTER_SEED, count_ledger="heat")
            rows.append({px: xv, py: yv, "pf": _pf_of(r),
                         "expectancy_r": r.metrics.expectancy_r,
                         "trades": r.metrics.total_trades,
                         "maxdd": r.metrics.max_drawdown,
                         "exp_id": r.experiment_id})
    heat = pd.DataFrame(rows)

    pf_vals = np.array([_pf_of(r) for r in oat], dtype=float)
    exp_vals = np.array([r.metrics.expectancy_r for r in oat], dtype=float)
    frac_exp = float((exp_vals > 0).mean()) if len(oat) else 0.0
    frac_pf = float((pf_vals > 1).mean()) if len(oat) else 0.0
    best_pf = float(pf_vals.max()) if len(oat) else 0.0
    med_pf = float(np.median(pf_vals)) if len(oat) else 0.0
    isolation = (best_pf / med_pf) if med_pf > 0 else float("inf")

    piv = heat.pivot(index=py, columns=px, values="pf").to_numpy(dtype=float)
    mask = np.isfinite(piv) & (piv > 1.0)
    # treat inf as >1 (finite check above drops inf; add explicitly)
    mask = mask | np.isinf(piv) & (piv > 0)
    summary = SensitivitySummary(
        oat_runs=len(oat), frac_positive_exp=frac_exp, frac_pf_gt1=frac_pf,
        best_pf=best_pf, median_neighbour_pf=med_pf, isolation_ratio=isolation,
        largest_connected_pf_gt1=largest_connected(mask),
        heatmap_cells=int(mask.size), heatmap_pf_gt1_cells=int(mask.sum()))
    return oat, heat, summary


def evaluate_gate2(summary: SensitivitySummary) -> Gate2Verdict:
    reasons = []
    if summary.frac_positive_exp < GATE2_MIN_POSITIVE_EXPECTANCY_FRAC:
        reasons.append(f"positive-exp neighbours {summary.frac_positive_exp:.0%} "
                       f"< {GATE2_MIN_POSITIVE_EXPECTANCY_FRAC:.0%}")
    if summary.frac_pf_gt1 < GATE2_MIN_PF_GT1_FRAC:
        reasons.append(f"PF>1 neighbours {summary.frac_pf_gt1:.0%} "
                       f"< {GATE2_MIN_PF_GT1_FRAC:.0%}")
    if summary.isolation_ratio > GATE2_MAX_ISOLATION_RATIO:
        reasons.append(f"isolation ratio {summary.isolation_ratio:.2f} "
                       f"> {GATE2_MAX_ISOLATION_RATIO} (lone spike)")
    if summary.largest_connected_pf_gt1 < GATE2_MIN_CONNECTED_REGION:
        reasons.append(f"largest stable heatmap region {summary.largest_connected_pf_gt1} "
                       f"< {GATE2_MIN_CONNECTED_REGION} cells")
    return Gate2Verdict(passed=not reasons, reasons=reasons)
