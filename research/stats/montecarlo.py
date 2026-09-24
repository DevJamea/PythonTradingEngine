"""Trade-shuffle Monte Carlo (§23).

Null-ish resampling: randomly reorder the OOS net-R sequence and replay the
equity path with FIXED fractional risk per trade (risk_per_trade compounding:
equity *= (1 + risk_frac * R)). This is a standard approximation — it ignores
the path-dependence of the original equity-proportional sizing — and the
report states that limitation. Sim count and seed are predefined.

Reports full distributions (P10/P50/P90/P95), never a single number:
MaxDD, terminal equity, longest losing streak, P(DD > thresholds).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    n_sims: int
    seed: int
    maxdd_dist: Dict[str, float]       # p10/p50/p90/p95 (+mean)
    terminal_equity_dist: Dict[str, float]
    losing_streak_dist: Dict[str, float]
    prob_dd_gt_10pct: float
    prob_dd_gt_25pct: float
    prob_ruin_half: float              # P(terminal equity < 50% of start)
    observed_maxdd: float
    observed_terminal: float
    p95_maxdd: float

    def to_dict(self) -> Dict:
        return {
            "n_sims": self.n_sims, "seed": self.seed,
            "maxdd_dist": self.maxdd_dist,
            "terminal_equity_dist": self.terminal_equity_dist,
            "losing_streak_dist": self.losing_streak_dist,
            "prob_dd_gt_10pct": self.prob_dd_gt_10pct,
            "prob_dd_gt_25pct": self.prob_dd_gt_25pct,
            "prob_ruin_half": self.prob_ruin_half,
            "observed_maxdd": self.observed_maxdd,
            "observed_terminal": self.observed_terminal,
            "p95_maxdd": self.p95_maxdd,
        }


def _quantiles(x: np.ndarray) -> Dict[str, float]:
    return {
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "mean": float(np.mean(x)),
    }


def _max_dd_pct(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - equity) / peak, 0.0)
    return float(dd.max())


def _longest_losing_streak(rs: np.ndarray) -> int:
    best = cur = 0
    for r in rs:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return best


def trade_shuffle_montecarlo(net_rs: Sequence[float], initial_balance: float,
                             risk_frac: float, n_sims: int,
                             seed: int) -> MonteCarloResult:
    rs = np.asarray(list(net_rs), dtype=float)
    if rs.size == 0:
        raise ValueError("Monte Carlo needs at least one trade")
    rng = np.random.default_rng(seed)
    maxdds = np.empty(n_sims)
    terminals = np.empty(n_sims)
    streaks = np.empty(n_sims)
    for s in range(n_sims):
        perm = rng.permutation(rs)
        # fixed-fractional replay; clip single-trade loss at -100% (no leverage blowup)
        rets = np.clip(risk_frac * perm, -0.99, None)
        eq = initial_balance * float(np.prod(1.0 + rets))
        path = initial_balance * np.cumprod(1.0 + rets)
        maxdds[s] = _max_dd_pct(path)
        terminals[s] = eq
        streaks[s] = _longest_losing_streak(perm)
    obs_rets = np.clip(risk_frac * rs, -0.99, None)
    obs_path = initial_balance * np.cumprod(1.0 + obs_rets)
    return MonteCarloResult(
        n_sims=n_sims, seed=seed,
        maxdd_dist=_quantiles(maxdds),
        terminal_equity_dist=_quantiles(terminals),
        losing_streak_dist=_quantiles(streaks),
        prob_dd_gt_10pct=float((maxdds > 0.10).mean()),
        prob_dd_gt_25pct=float((maxdds > 0.25).mean()),
        prob_ruin_half=float((terminals < 0.5 * initial_balance).mean()),
        observed_maxdd=_max_dd_pct(obs_path),
        observed_terminal=float(obs_path[-1]),
        p95_maxdd=float(np.quantile(maxdds, 0.95)),
    )
