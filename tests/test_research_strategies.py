"""Research tests: A/B/C generation contracts + baseline fidelity."""
from __future__ import annotations

import pandas as pd

from gold_trader.config import Config
from gold_trader.strategy.signals import compute_indicators, evaluate_at
from research.config import A_PRIMARY, B_PRIMARY, C_PRIMARY
from research.data.synthetic import generate_synthetic_m15
from research.strategies.baseline_wrap import generate_baseline_signals
from research.strategies.strategy_a import generate_a_signals
from research.strategies.strategy_b import generate_b_signals
from research.strategies.strategy_c import generate_c_signals, regime_series
from tests._helpers import make_uptrend_with_engulfing


def _data() -> pd.DataFrame:
    return generate_synthetic_m15(end="2022-06-30 23:45")


class TestContracts:
    def test_a_output_contract(self):
        out = generate_a_signals(_data(), A_PRIMARY)
        assert out.diagnostics["evaluated_bars"] > 0
        for s in out.signals:
            assert s.direction in ("BUY", "SELL")
            assert s.entry_index == s.index + 1  # PRIMARY entry-at-close
            assert s.sl_distance > 0 and s.tp_distance > 0

    def test_b_confirmation_delay_and_range_stop(self):
        out = generate_b_signals(_data(), B_PRIMARY)
        for s in out.signals:
            assert s.entry_index == s.index + 2  # confirmation bar
            assert s.sl_distance > 0

    def test_c_range_is_no_trade_and_subset(self):
        df = _data()
        out_c = generate_c_signals(df, C_PRIMARY)
        out_a = generate_a_signals(df, C_PRIMARY.trend_params)
        out_b = generate_b_signals(df, C_PRIMARY.compression_params)
        assert len(out_c.signals) <= len(out_a.signals) + len(out_b.signals)
        assert out_c.diagnostics["range_action"] == "no_trade"
        mix = out_c.diagnostics["regime_mix"]
        assert set(mix) == {"TREND", "COMPRESSION", "RANGE", "NONE"}

    def test_c_regime_labels_cover_all_bars(self):
        labels = regime_series(_data(), C_PRIMARY)["regime"]
        assert set(labels.unique()) <= {"TREND", "COMPRESSION", "RANGE", "NONE"}
        assert len(labels) == len(_data())


class TestBaselineFidelity:
    def test_wrapper_matches_production_decisions(self):
        # The wrapper must reproduce production evaluate_at signal directions
        # exactly on a deterministic scenario (fidelity, not a copy).
        df = make_uptrend_with_engulfing()
        cfg = Config()
        ind = compute_indicators(df, cfg)
        prod = {}
        for i in range(cfg.min_candles_for_signal, len(df) - 1):
            prod[i] = evaluate_at(i, df, ind, cfg).signal.value
        out = generate_baseline_signals(df, cfg)
        wrapped = {s.index: s.direction for s in out.signals}
        for i, sig in prod.items():
            if sig == "NO_TRADE":
                assert i not in wrapped
            else:
                assert wrapped.get(i) == sig
