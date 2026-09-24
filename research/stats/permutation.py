"""Permutation / null test (§25).

Predefined null model: the strategy's directional edge is zero, i.e. trade
signs are exchangeable. Operationalised as a sign-flip permutation on OOS
net-R: each replicate randomly flips the sign of each trade's R and recomputes
the mean. Empirical one-sided p-value = fraction of replicates with mean >=
observed mean.

Limitations (stated in the report): sign-flipping destroys serial dependence
and assumes a symmetric null; it tests "distinguishable from a no-directional-
edge process", NOT "proven market edge". No claim of proof is made.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class PermutationResult:
    n_perms: int
    seed: int
    n_trades: int
    observed_mean_r: float
    null_mean: float
    p_value_one_sided: float
    null_hypothesis: str = ("H0: trade signs are exchangeable (no exploitable "
                            "directional edge); test statistic = mean net-R.")
    limitation: str = ("Sign-flip destroys serial dependence and assumes a "
                       "symmetric null. A small p-value rejects this specific "
                       "null; it does not prove a tradeable market edge.")

    def to_dict(self) -> Dict:
        return {
            "n_perms": self.n_perms, "seed": self.seed, "n_trades": self.n_trades,
            "observed_mean_r": self.observed_mean_r, "null_mean": self.null_mean,
            "p_value_one_sided": self.p_value_one_sided,
            "null_hypothesis": self.null_hypothesis,
            "limitation": self.limitation,
        }


def sign_flip_permutation(net_rs: Sequence[float], n_perms: int,
                          seed: int) -> PermutationResult:
    rs = np.asarray(list(net_rs), dtype=float)
    n = rs.size
    if n == 0:
        raise ValueError("permutation test needs at least one trade")
    rng = np.random.default_rng(seed)
    obs = float(rs.mean())
    null_means = np.empty(n_perms)
    for k in range(n_perms):
        flips = rng.choice([-1.0, 1.0], size=n)
        null_means[k] = float((rs * flips).mean())
    # +1 smoothing (conservative, avoids p=0)
    p = float(((null_means >= obs).sum() + 1) / (n_perms + 1))
    return PermutationResult(n_perms, seed, n, obs, float(null_means.mean()), p)
