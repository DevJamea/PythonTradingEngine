"""Research tests: gate evaluators accept/reject per the predefined rules."""
from __future__ import annotations

from research.runners.gate1 import evaluate_gate1
from research.runners.oos import evaluate_gate4, evaluate_gate6
from research.runners.sensitivity import SensitivitySummary, evaluate_gate2
from research.runners.walkforward import WFSummary, evaluate_gate3


def _m(pf=1.5, trades=100, exp=0.1, dd=100.0):
    return {"profit_factor": pf, "total_trades": trades,
            "expectancy_r": exp, "max_drawdown": dd}


class TestGate1:
    def test_pass(self):
        assert evaluate_gate1(_m(), 100.0).passed

    def test_each_criterion_rejects(self):
        assert not evaluate_gate1(_m(pf=1.19), 100.0).passed
        assert not evaluate_gate1(_m(trades=49), 100.0).passed
        assert not evaluate_gate1(_m(exp=0.0), 100.0).passed
        assert not evaluate_gate1(_m(dd=151.0), 100.0).passed


class TestGate2:
    def _s(self, **kw):
        base = dict(oat_runs=10, frac_positive_exp=0.8, frac_pf_gt1=0.8,
                    best_pf=1.5, median_neighbour_pf=1.2, isolation_ratio=1.25,
                    largest_connected_pf_gt1=5, heatmap_cells=25,
                    heatmap_pf_gt1_cells=10)
        base.update(kw)
        return SensitivitySummary(**base)

    def test_pass(self):
        assert evaluate_gate2(self._s()).passed

    def test_plateau_rules(self):
        assert not evaluate_gate2(self._s(frac_positive_exp=0.2)).passed
        assert not evaluate_gate2(self._s(frac_pf_gt1=0.2)).passed
        assert not evaluate_gate2(self._s(isolation_ratio=5.0)).passed
        assert not evaluate_gate2(self._s(largest_connected_pf_gt1=1)).passed


class TestGate3:
    def _w(self, **kw):
        base = dict(wfe=0.8, win_window_ratio=0.75, median_oos_expectancy=0.1,
                    median_oos_pf=1.4, oos_trades_total=200,
                    concentration_best_window=0.3, max_window_drawdown=50.0,
                    n_windows=4, n_valid_windows=4, n_low_sample_windows=0,
                    dev_expectancy_r=0.12)
        base.update(kw)
        return WFSummary(**base)

    def test_pass(self):
        assert evaluate_gate3(self._w()).passed

    def test_rules(self):
        assert not evaluate_gate3(self._w(wfe=0.4)).passed
        assert not evaluate_gate3(self._w(win_window_ratio=0.5)).passed
        assert not evaluate_gate3(self._w(concentration_best_window=0.8)).passed
        assert not evaluate_gate3(self._w(n_valid_windows=1)).passed


class _MM:
    def __init__(self, pf, exp, dd):
        self.profit_factor = pf
        self.expectancy_r = exp
        self.max_drawdown = dd


class TestGate4:
    def _w(self, med=0.1):
        return WFSummary(0.8, 0.75, med, 1.4, 200, 0.3, 50.0, 4, 4, 0, 0.12)

    def test_pass(self):
        assert evaluate_gate4(_MM(1.3, 0.08, 100.0), self._w(), 100.0).passed

    def test_rules(self):
        assert not evaluate_gate4(_MM(0.9, 0.08, 100.0), self._w(), 100.0).passed
        assert not evaluate_gate4(_MM(1.3, 0.01, 100.0), self._w(), 100.0).passed
        assert not evaluate_gate4(_MM(1.3, 0.08, 200.0), self._w(), 100.0).passed
        assert not evaluate_gate4(_MM(1.3, 0.08, 100.0), self._w(med=0.0), 100.0).passed


class TestGate6:
    def test_all_scenarios_must_pass(self):
        from research.runners.oos import StressSummary
        ok = StressSummary({k: {"pf": t, "target": t, "pass": True}
                            for k, t in (("a", 1.0), ("b", 0.9))})
        assert evaluate_gate6(ok).passed
        bad = StressSummary({"a": {"pf": 0.5, "target": 1.0, "pass": False}})
        assert not evaluate_gate6(bad).passed
