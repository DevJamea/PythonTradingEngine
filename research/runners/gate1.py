"""Gate 1 — Development screening (§12).

Runs Baseline (reference) + A/B/C primaries + all predefined diagnostics on
the Development segment with initial parameters. Screening criteria:
PF >= 1.20, trades >= 50, expectancy > 0, MaxDD <= 1.5x Baseline MaxDD.
The t-statistic is reported as auxiliary only (never a gate).
Diagnostics inform the report; only primaries can pass Gate 1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..config import (A_DIAG_NO_MGMT, A_DIAG_RETEST, A_PRIMARY, A_RR15, A_RR25,
                      B_DIAG_SESSION, B_DIAG_TRAIL, B_PRIMARY, B_RR15, B_RR25,
                      C_PRIMARY, GATE1_MAXDD_MULT_BASELINE, GATE1_MIN_PF,
                      GATE1_MIN_TRADES, MASTER_SEED)
from ..registry import TrialLedger
from ..registry import Registry
from .common import ConfigResult, Handle, run_configuration


@dataclass(frozen=True)
class Gate1Verdict:
    passed: bool
    reasons: List[str]


def evaluate_gate1(m: dict, baseline_maxdd: float) -> Gate1Verdict:
    pf = m["profit_factor"]
    pf = float(pf) if isinstance(pf, (int, float)) else 0.0
    reasons = []
    if not (pf >= GATE1_MIN_PF):
        reasons.append(f"PF {pf:.3f} < {GATE1_MIN_PF}")
    if not (m["total_trades"] >= GATE1_MIN_TRADES):
        reasons.append(f"trades {m['total_trades']} < {GATE1_MIN_TRADES}")
    if not (m["expectancy_r"] > 0):
        reasons.append(f"expectancy {m['expectancy_r']:.4f}R <= 0")
    tol = GATE1_MAXDD_MULT_BASELINE * baseline_maxdd
    if not (m["max_drawdown"] <= tol):
        reasons.append(f"MaxDD {m['max_drawdown']:.2f} > 1.5x baseline ({tol:.2f})")
    return Gate1Verdict(passed=not reasons, reasons=reasons)


def primary_handles() -> Dict[str, Handle]:
    return {
        "baseline": Handle("baseline", "reference", None),
        "A": Handle("A", "primary", A_PRIMARY),
        "B": Handle("B", "primary", B_PRIMARY),
        "C": Handle("C", "primary", C_PRIMARY),
    }


def diagnostic_handles() -> Dict[str, Handle]:
    return {
        "A:RR15": Handle("A", "diagnostic:RR1.5", A_RR15),
        "A:RR25": Handle("A", "diagnostic:RR2.5", A_RR25),
        "A:no-mgmt": Handle("A", "diagnostic:entry-exit-only", A_DIAG_NO_MGMT),
        "A:retest": Handle("A", "diagnostic:retest-entry", A_DIAG_RETEST),
        "B:RR15": Handle("B", "diagnostic:RR1.5", B_RR15),
        "B:RR25": Handle("B", "diagnostic:RR2.5", B_RR25),
        "B:trail": Handle("B", "diagnostic:trailing-exit", B_DIAG_TRAIL),
        "B:session": Handle("B", "diagnostic:london-ny-session", B_DIAG_SESSION),
    }


def run_gate1(dev_df: pd.DataFrame, registry: Registry,
              ledger: TrialLedger) -> Dict[str, ConfigResult]:
    results: Dict[str, ConfigResult] = {}
    for key, h in list(primary_handles().items()) + list(diagnostic_handles().items()):
        results[key] = run_configuration(
            h, dev_df, "development", registry, ledger, gate="gate1",
            seed=MASTER_SEED, count_ledger="gate1")
    return results


def gate1_verdicts(results: Dict[str, ConfigResult]) -> Dict[str, Gate1Verdict]:
    baseline_maxdd = results["baseline"].metrics.max_drawdown
    return {k: evaluate_gate1(results[k].metrics.to_dict(), baseline_maxdd)
            for k in ("A", "B", "C")}
