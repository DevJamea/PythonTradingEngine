"""OOS holdout + Gates 4/5/6 (§21-30).

Order of operations (enforced):
1. Freeze receipt is written BEFORE the holdout slice is built; the splits
   module refuses to serve the holdout while it is locked.
2. Baseline + Gate-3 survivors run once on the full holdout. No peeking, no
   changes — any change would burn the OOS (tracked as a new round).
3. Gate 4: PF >= 1.0, expectancy degradation <= 50% vs WF median OOS,
   MaxDD within 1.5x OOS-baseline tolerance.
4. Gate-4 survivors: Monte Carlo (5000), bootstrap (5000), permutation
   (5000), execution-delay proxy (0/1/2s), DSR with ledger n_trials.
5. Gate 5: bootstrap CI>0 AND MC not fragile AND delay survives AND DSR>=0.95.
6. Gate-5 survivors: cost stress (spread x1.5/x2.0, slippage, combined).
7. Gate 6: all stress targets met => RESEARCH PROTOTYPE CANDIDATE
   (NOT a production strategy, NOT a live strategy, NOT a guaranteed edge).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import (BOOTSTRAP_SIMS, GATE1_MAXDD_MULT_BASELINE,
                      GATE4_MAX_EXPECTANCY_DEGRADATION, GATE4_MIN_PF,
                      GATE5_DELAY_MIN_PF, DSR_THRESHOLD,
                      GATE5_REQUIRE_BOOTSTRAP_CI_ABOVE_ZERO, MASTER_SEED,
                      MC_P95_DD_MULT_FAIL, MC_RUIN_EQUITY_FRAC,
                      MC_RUIN_PROB_FAIL, MC_SIMS, PERMUTATION_SIMS,
                      RISK_PER_TRADE, SEED_BOOTSTRAP, SEED_DELAY,
                      SEED_MONTECARLO, SEED_PERMUTATION, SEED_SLIPPAGE,
                      STRESS_SLIPPAGE_POINTS, STRESS_TARGETS, INITIAL_BALANCE)
from ..costs import draw_slippage_points
from ..data.splits import (FREEZE_PATH, Segment, assert_holdout_unlocked,
                           slice_segment)
from ..registry import FreezeReceipt, Registry, TrialLedger
from ..stats.bootstrap import BootstrapResult, bootstrap_trades
from ..stats.delay import DelayResult
from ..stats.dsr import DSRResult, deflated_sharpe
from ..stats.montecarlo import MonteCarloResult, trade_shuffle_montecarlo
from ..stats.permutation import PermutationResult, sign_flip_permutation
from .common import (ConfigResult, Handle, run_configuration, sorted_signals)
from .walkforward import WFSummary


@dataclass(frozen=True)
class Gate4Verdict:
    passed: bool
    reasons: List[str]
    degradation: float


@dataclass(frozen=True)
class Gate5Verdict:
    passed: bool
    reasons: List[str]


@dataclass(frozen=True)
class StressSummary:
    scenarios: Dict[str, Dict[str, float]]  # name -> {pf, expectancy_r, maxdd, trades, ...}

    def to_dict(self) -> Dict:
        return {"scenarios": self.scenarios}


@dataclass(frozen=True)
class Gate6Verdict:
    passed: bool
    reasons: List[str]


def open_holdout(full_df: pd.DataFrame, oos_seg: Segment, dataset_hash: str,
                 strategies: List[str]) -> Tuple[pd.DataFrame, FreezeReceipt]:
    """Write the freeze receipt, then (and only then) slice the holdout."""
    oos_range = {"start": str(full_df["time"].iloc[oos_seg.start]),
                 "end": str(full_df["time"].iloc[oos_seg.end - 1]),
                 "bars": oos_seg.end - oos_seg.start}
    receipt = FreezeReceipt.create("oos-open", dataset_hash, oos_range, strategies)
    receipt.save(FREEZE_PATH)
    assert_holdout_unlocked()
    return slice_segment(full_df, oos_seg), receipt


def evaluate_gate4(m, wf: WFSummary, oos_baseline_maxdd: float) -> Gate4Verdict:
    reasons = []
    pf = float(m.profit_factor) if np.isfinite(m.profit_factor) else 999.0
    if not (pf >= GATE4_MIN_PF):
        reasons.append(f"OOS PF {pf:.3f} < {GATE4_MIN_PF}")
    if wf.median_oos_expectancy <= 0:
        reasons.append("WF median OOS expectancy <= 0 — no positive reference to compare")
        degradation = float("inf")
    else:
        degradation = 1.0 - (m.expectancy_r / wf.median_oos_expectancy)
        if not (degradation <= GATE4_MAX_EXPECTANCY_DEGRADATION):
            reasons.append(f"expectancy degradation {degradation:.0%} "
                           f"> {GATE4_MAX_EXPECTANCY_DEGRADATION:.0%} vs WF")
    tol = GATE1_MAXDD_MULT_BASELINE * oos_baseline_maxdd
    if not (m.max_drawdown <= tol):
        reasons.append(f"OOS MaxDD {m.max_drawdown:.2f} > 1.5x OOS baseline ({tol:.2f})")
    return Gate4Verdict(passed=not reasons, reasons=reasons, degradation=degradation)


def run_oos_validation(oos_df: pd.DataFrame, handle: Handle,
                       oos_result: ConfigResult, ledger: TrialLedger,
                       registry: Registry):
    """Monte Carlo + bootstrap + permutation + delay + DSR for one survivor."""
    trades = oos_result.engine.trades
    rs = [t.r_net for t in trades]
    pnls = [t.pnl_net for t in trades]

    mc = trade_shuffle_montecarlo(rs, INITIAL_BALANCE, RISK_PER_TRADE,
                                  MC_SIMS, SEED_MONTECARLO)
    boot = bootstrap_trades(rs, pnls, RISK_PER_TRADE, BOOTSTRAP_SIMS, SEED_BOOTSTRAP)
    perm = sign_flip_permutation(rs, PERMUTATION_SIMS, SEED_PERMUTATION)

    # execution-delay proxy re-runs (signals identical; entry slipped)
    delay_levels_out: Dict[int, Dict[str, float]] = {}
    for secs, frac in ((0, 0.0), (1, 0.25), (2, 0.50)):
        h = Handle(handle.strategy, f"oos-delay-{secs}s", handle.params)
        r = run_configuration(h, oos_df, "oos", registry, ledger,
                              gate="delay", seed=SEED_DELAY,
                              delay_entry_spread_frac=frac, count_ledger="none")
        delay_levels_out[secs] = {
            "pf": float(r.metrics.profit_factor) if np.isfinite(r.metrics.profit_factor) else 999.0,
            "expectancy_r": r.metrics.expectancy_r,
            "trades": r.metrics.total_trades,
            "maxdd": r.metrics.max_drawdown,
            "exp_id": r.experiment_id,
        }
    delay = DelayResult(levels=delay_levels_out)

    dsr = deflated_sharpe([RISK_PER_TRADE * r for r in rs],
                          ledger.n_trials_for_dsr, threshold=DSR_THRESHOLD)
    return mc, boot, perm, delay, dsr


def evaluate_gate5(oos_result: ConfigResult, mc: MonteCarloResult,
                   boot: BootstrapResult, delay: DelayResult,
                   dsr: DSRResult) -> Gate5Verdict:
    reasons = []
    if GATE5_REQUIRE_BOOTSTRAP_CI_ABOVE_ZERO and not (boot.expectancy_ci95["lo"] > 0):
        reasons.append(f"bootstrap 95% CI [{boot.expectancy_ci95['lo']:.4f}, "
                       f"{boot.expectancy_ci95['hi']:.4f}] crosses zero")
    obs_dd = max(oos_result.metrics.max_drawdown_pct, 1e-9)
    if mc.p95_maxdd > MC_P95_DD_MULT_FAIL * obs_dd and oos_result.metrics.max_drawdown_pct > 0:
        reasons.append(f"MC P95 MaxDD {mc.p95_maxdd:.1%} > 3x observed {obs_dd:.1%}")
    if mc.prob_ruin_half > MC_RUIN_PROB_FAIL:
        reasons.append(f"MC P(terminal < 50% start) {mc.prob_ruin_half:.1%} > 5%")
    d2 = delay.levels[2]
    if not (d2["pf"] >= GATE5_DELAY_MIN_PF and d2["expectancy_r"] > 0):
        reasons.append(f"2s-delay proxy destroys result (PF {d2['pf']:.3f}, "
                       f"exp {d2['expectancy_r']:.4f}R)")
    if not (dsr.dsr >= DSR_THRESHOLD):
        reasons.append(f"DSR {dsr.dsr:.4f} < {DSR_THRESHOLD}")
    return Gate5Verdict(passed=not reasons, reasons=reasons)


def run_cost_stress(oos_df: pd.DataFrame, handle: Handle, ledger: TrialLedger,
                    registry: Registry) -> StressSummary:
    from .common import generate  # local import to avoid cycles at module load
    n_signals = len(sorted_signals(generate(handle, oos_df).signals))
    rng = np.random.default_rng(SEED_SLIPPAGE)
    slip_in = draw_slippage_points(rng, n_signals)
    slip_out = draw_slippage_points(rng, n_signals)

    scenarios: Dict[str, Dict[str, float]] = {}

    def _run(name: str, target: float, **kw):
        h = Handle(handle.strategy, f"stress:{name}", handle.params)
        r = run_configuration(h, oos_df, "oos", registry, ledger,
                              gate="stress", seed=SEED_SLIPPAGE, count_ledger="none", **kw)
        pf = float(r.metrics.profit_factor) if np.isfinite(r.metrics.profit_factor) else 999.0
        scenarios[name] = {"pf": pf, "expectancy_r": r.metrics.expectancy_r,
                           "maxdd": r.metrics.max_drawdown,
                           "trades": r.metrics.total_trades,
                           "costs": r.metrics.total_costs,
                           "target": target, "pass": bool(pf >= target),
                           "exp_id": r.experiment_id}

    _run("spread_1.5", STRESS_TARGETS["spread_1.5"], spread_mult=1.5)
    _run("spread_2.0", STRESS_TARGETS["spread_2.0"], spread_mult=2.0)
    _run("slippage", STRESS_TARGETS["slippage"],
         slip_entry_pts=slip_in, slip_exit_pts=slip_out)
    _run("combined", STRESS_TARGETS["combined"], spread_mult=1.5,
         slip_entry_pts=slip_in, slip_exit_pts=slip_out)
    return StressSummary(scenarios)


def evaluate_gate6(stress: StressSummary) -> Gate6Verdict:
    reasons = [f"{name}: PF {s['pf']:.3f} < target {s['target']}"
               for name, s in stress.scenarios.items() if not s["pass"]]
    return Gate6Verdict(passed=not reasons, reasons=reasons)
