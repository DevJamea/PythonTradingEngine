"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) — §27.

Standard implementation on per-trade net-R returns (units documented):
* SR = mean/std of per-trade fractional returns r = risk_frac * R (not
  annualised inside DSR — the formula is scale-consistent as long as T is the
  number of observations, here the OOS trade count);
* gamma3/gamma4 = sample skewness/kurtosis of r;
* T = number of OOS trades;
* n_trials = materially tested configurations from the TrialLedger
  (Gate-1 + sensitivity + heatmap configs), NOT just "Baseline+A+B+C".

DSR = Phi((SR - E[max SR_0]) / sd), where E[max SR_0] is the expected Sharpe
under the null given n_trials (benchmark SR0 = 0). The report states how
n_trials was defined and stresses DSR is a screening criterion, never a
probability of future profitability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    # Acklam's approximation for the standard normal quantile.
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


@dataclass(frozen=True)
class DSRResult:
    sharpe_observed: float
    skewness: float
    kurtosis: float
    n_observations: int
    n_trials: int
    expected_max_sharpe_null: float
    dsr: float
    threshold: float
    units: str = "per-trade fractional returns (risk_frac * R); T = OOS trades"

    def to_dict(self) -> Dict:
        return {
            "sharpe_observed": self.sharpe_observed, "skewness": self.skewness,
            "kurtosis": self.kurtosis, "n_observations": self.n_observations,
            "n_trials": self.n_trials,
            "expected_max_sharpe_null": self.expected_max_sharpe_null,
            "dsr": self.dsr, "threshold": self.threshold, "units": self.units,
        }


def deflated_sharpe(trade_returns: Sequence[float], n_trials: int,
                    benchmark_sr: float = 0.0,
                    threshold: float = 0.95) -> DSRResult:
    r = np.asarray(list(trade_returns), dtype=float)
    t = r.size
    if t < 3:
        raise ValueError("DSR needs at least 3 observations")
    n_trials = max(1, int(n_trials))
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    sr = mu / sd if sd > 0 else 0.0
    m2 = float(((r - mu) ** 2).mean())
    m3 = float(((r - mu) ** 3).mean())
    m4 = float(((r - mu) ** 4).mean())
    skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    kurt = m4 / (m2 ** 2) if m2 > 0 else 3.0  # raw kurtosis (normal = 3)
    gamma = 0.5772156649  # Euler-Mascheroni
    if n_trials == 1:
        e_max = benchmark_sr
    else:
        e_max = (math.sqrt(1.0) *
                 ((1 - gamma) * _phi_inv(1 - 1.0 / n_trials)
                  + gamma * _phi_inv(1 - 1.0 / (n_trials * math.e))))
    denom = math.sqrt(max(1e-12, (kurt - 1.0) / t))
    dsr = _phi((sr - e_max) / denom) if denom > 0 else 0.0
    return DSRResult(sr, skew, kurt, t, n_trials, e_max, dsr, threshold)
