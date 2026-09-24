"""Research tests: registry, frozen params, ledger, freeze receipts."""
from __future__ import annotations

import json

import pytest

from research.config import A_PRIMARY, config_hash, frozen_config_dict
from research.registry import (ExperimentRecord, FreezeReceipt, Registry,
                               TrialLedger, git_commit, utc_now_iso)


def _rec(strategy="A", variant="primary"):
    return ExperimentRecord(
        experiment_id="EXP-0001-A-test", strategy=strategy, variant=variant,
        parameters={"rr": 2.0}, data_range={"segment": "development"},
        timeframe="M15", cost_model={"model": "bid_ask"},
        risk_model={"risk_per_trade": 0.005}, entry_model="e", exit_model="x",
        management_model="m", session_filter="all", seed=1,
        code_version="research-test", code_commit="abc",
        config_hash="hash", timestamp=utc_now_iso())


def test_registry_append_and_load(tmp_path):
    reg = Registry(tmp_path / "reg.jsonl")
    assert reg.next_id("A", "gate1") == "EXP-0001-A-gate1"
    rec = _rec()
    reg.record(rec)
    rows = reg.load_all()
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "EXP-0001-A-test"
    assert rows[0]["parameters"] == {"rr": 2.0}
    # machine-readable: every line is valid JSON with required keys
    for key in ("strategy", "parameters", "data_range", "cost_model",
                "risk_model", "seed", "code_commit", "timestamp"):
        assert key in rows[0]


def test_frozen_config_hash_stable():
    assert config_hash() == config_hash()
    d = frozen_config_dict()
    blob = json.dumps(d, sort_keys=True)
    assert "pullback_max_atr" in blob and "compression_ratio" in blob


def test_frozen_params_immutable():
    with pytest.raises(Exception):
        A_PRIMARY.rr_target = 99.0  # frozen dataclass


def test_ledger_counts_and_dsr_trials():
    ledger = TrialLedger()
    ledger.gate1_configs = 12
    ledger.sensitivity_configs = 30
    ledger.heatmap_configs = 25
    assert ledger.total_configurations == 67
    assert ledger.n_trials_for_dsr == 67
    d = ledger.to_dict()
    assert d["TOTAL_CONFIGURATIONS"] == 67
    assert d["N_TRIALS_FOR_DSR"] == 67


def test_freeze_receipt_roundtrip(tmp_path):
    r = FreezeReceipt.create("oos-open", "datahash",
                             {"start": "a", "end": "b"}, ["A"])
    p = r.save(tmp_path / "freeze.json")
    back = FreezeReceipt.load(p)
    assert back.receipt_hash == r.receipt_hash
    assert back.config_hash == r.config_hash
    # receipt hash commits to content
    r2 = FreezeReceipt.create("oos-open", "DIFFERENT", {"start": "a"}, ["A"])
    assert r2.receipt_hash != r.receipt_hash
