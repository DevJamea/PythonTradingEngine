"""Research tests: Monte Carlo, bootstrap, permutation, DSR, delay."""
from __future__ import annotations

import numpy as np
import pytest

from research.stats.bootstrap import bootstrap_trades
from research.stats.delay import delay_levels
from research.stats.dsr import deflated_sharpe
from research.stats.montecarlo import trade_shuffle_montecarlo
from research.stats.permutation import sign_flip_permutation

RS = [0.5, -1.0, 1.5, 0.2, -0.4, 2.0, -1.0, 0.8] * 10  # 80 trades
PNLS = [r * 50.0 for r in RS]


def test_montecarlo_deterministic_and_reports_distributions():
    a = trade_shuffle_montecarlo(RS, 10_000.0, 0.005, 200, 123456)
    b = trade_shuffle_montecarlo(RS, 10_000.0, 0.005, 200, 123456)
    assert a.to_dict() == b.to_dict()
    for key in ("p10", "p50", "p90", "p95"):
        assert key in a.maxdd_dist and key in a.terminal_equity_dist
    assert 0.0 <= a.prob_dd_gt_10pct <= 1.0
    with pytest.raises(ValueError):
        trade_shuffle_montecarlo([], 10_000.0, 0.005, 10, 1)


def test_bootstrap_ci_contains_sample_mean():
    res = bootstrap_trades(RS, PNLS, 0.005, 500, 789)
    mean = float(np.mean(RS))
    assert res.expectancy_ci95["lo"] <= mean <= res.expectancy_ci95["hi"]
    assert 0.0 <= res.prob_expectancy_gt0 <= 1.0
    assert "ratio" in res.caveat_pf


def test_permutation_p_value_sane_and_deterministic():
    a = sign_flip_permutation(RS, 1000, 101112)
    b = sign_flip_permutation(RS, 1000, 101112)
    assert a.p_value_one_sided == b.p_value_one_sided
    assert 0.0 < a.p_value_one_sided <= 1.0
    # strong positive edge -> small p; symmetric noise -> large p
    edge = sign_flip_permutation([1.0] * 60, 2000, 7)
    assert edge.p_value_one_sided < 0.05
    noise = sign_flip_permutation([1.0, -1.0] * 30, 2000, 7)
    assert noise.p_value_one_sided > 0.05


def test_dsr_inputs_and_trial_penalty():
    rets = [0.005 * r for r in RS]
    one = deflated_sharpe(rets, 1)
    many = deflated_sharpe(rets, 200)
    assert one.n_observations == len(RS) == many.n_observations
    assert many.expected_max_sharpe_null > one.expected_max_sharpe_null
    assert many.dsr < one.dsr  # more trials -> harsher deflation
    assert 0.0 <= many.dsr <= 1.0
    with pytest.raises(ValueError):
        deflated_sharpe([0.1, 0.2], 5)


def test_delay_levels_predefined():
    assert delay_levels() == {0: 0.0, 1: 0.25, 2: 0.50}
