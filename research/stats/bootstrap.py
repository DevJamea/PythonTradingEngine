"""Bootstrap inference on OOS trades (§24).

Resample net-R WITH replacement (n_out = n_trades per replicate, seeded) and
derive 95% percentile confidence intervals for expectancy, mean R and total
return (fixed-fractional compounding replay). PF is bootstrapped as the ratio
of resampled gross profit to gross loss WITH an explicit caveat: a ratio of
sums is unstable when the resampled loss leg is near zero, so expectancy-based
inference is primary and PF intervals are reported alongside with that warning.

A CI crossing zero expectancy means uncertainty remains — reported as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    n_sims: int
    seed: int
    n_trades: int
    expectancy_ci95: Dict[str, float]   # lo/hi/mean
    mean_r_ci95: Dict[str, float]
    total_return_ci95: Dict[str, float]  # fraction, fixed-fractional replay
    pf_ci95: Dict[str, float]            # with caveat (see module docstring)
    prob_expectancy_gt0: float
    caveat_pf: str = ("PF is a ratio of sums; bootstrap intervals for ratios "
                      "are unstable when the resampled loss leg is small. "
                      "Expectancy-based inference is primary.")

    def to_dict(self) -> Dict:
        return {
            "n_sims": self.n_sims, "seed": self.seed, "n_trades": self.n_trades,
            "expectancy_ci95": self.expectancy_ci95,
            "mean_r_ci95": self.mean_r_ci95,
            "total_return_ci95": self.total_return_ci95,
            "pf_ci95": self.pf_ci95,
            "prob_expectancy_gt0": self.prob_expectancy_gt0,
            "caveat_pf": self.caveat_pf,
        }


def bootstrap_trades(net_rs: Sequence[float], net_pnls: Sequence[float],
                     risk_frac: float, n_sims: int,
                     seed: int) -> BootstrapResult:
    rs = np.asarray(list(net_rs), dtype=float)
    pnls = np.asarray(list(net_pnls), dtype=float)
    n = rs.size
    if n == 0:
        raise ValueError("bootstrap needs at least one trade")
    rng = np.random.default_rng(seed)
    exp_boot = np.empty(n_sims)
    ret_boot = np.empty(n_sims)
    pf_boot = np.empty(n_sims)
    for s in range(n_sims):
        idx = rng.integers(0, n, size=n)
        sample_r = rs[idx]
        sample_p = pnls[idx]
        exp_boot[s] = sample_r.mean()
        rets = np.clip(risk_frac * sample_r, -0.99, None)
        ret_boot[s] = float(np.prod(1.0 + rets) - 1.0)
        gp = sample_p[sample_p > 0].sum()
        gl = sample_p[sample_p < 0].sum()
        pf_boot[s] = (gp / abs(gl)) if gl < 0 else (np.inf if gp > 0 else 0.0)
    finite_pf = pf_boot[np.isfinite(pf_boot)]
    if finite_pf.size == 0:
        pf_ci = {"lo": float("inf"), "hi": float("inf"), "mean": float("inf")}
    else:
        pf_ci = {"lo": float(np.quantile(finite_pf, 0.025)),
                 "hi": float(np.quantile(finite_pf, 0.975)),
                 "mean": float(np.mean(finite_pf))}
    ci = lambda x: {"lo": float(np.quantile(x, 0.025)),
                    "hi": float(np.quantile(x, 0.975)),
                    "mean": float(np.mean(x))}
    return BootstrapResult(
        n_sims=n_sims, seed=seed, n_trades=n,
        expectancy_ci95=ci(exp_boot), mean_r_ci95=ci(exp_boot),
        total_return_ci95=ci(ret_boot), pf_ci95=pf_ci,
        prob_expectancy_gt0=float((exp_boot > 0).mean()),
    )
