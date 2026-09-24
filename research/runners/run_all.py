"""Master orchestrator for the Phase-2 experiment (§40 outputs).

Runs the full gated pipeline in order, enforcing the holdout lock, and writes
every artifact under ``research/outputs/``:
  registry.jsonl, ledger.json, data_audit.json, splits.json, gate1.csv,
  sensitivity_<S>.csv, heatmap_<S>.csv, walkforward_<S>.csv, oos.csv,
  validation_<S>.json, stress_<S>.csv, verdicts.json, environment.json,
  production_snapshot.json, leakage_audit.json, master_table.md
The final scientific report is built by ``report.py``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..config import MASTER_SEED, config_hash, frozen_config_dict
from ..data.loader import audit_data, load_dataset
from ..data.splits import (FREEZE_PATH, chronological_splits, holdout_locked,
                           slice_segment)
from ..registry import OUTPUTS, REGISTRY_PATH, Registry, TrialLedger
from . import gate1 as g1
from . import oos as oos_mod
from . import sensitivity as sens
from . import walkforward as wf

STRATEGIES = ("A", "B", "C")


def _write_json(name: str, obj: Any) -> Path:
    p = OUTPUTS / name
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    return p


def _metrics_row(key: str, res) -> Dict[str, Any]:
    m = res.metrics.to_dict()
    return {"strategy": key, "variant": res.handle.variant,
            "exp_id": res.experiment_id, **m,
            "t_stat_aux": m.get("t_stat_aux"),
            "signals_raw": res.output.diagnostics.get("raw_signals"),
            "skipped_in_position": res.engine.diagnostics.signals_skipped_in_position,
            "sizing_failed": res.engine.diagnostics.signals_sizing_failed}


def _production_snapshot() -> Dict[str, str]:
    root = Path(__file__).resolve().parent.parent.parent
    snap = {}
    for sub in ("gold_trader", "tests"):
        for p in sorted((root / sub).rglob("*.py")):
            if "test_research" in p.name:
                continue  # research tests are new files, not production
            snap[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return snap


def main() -> Dict[str, Any]:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if FREEZE_PATH.exists():
        FREEZE_PATH.unlink()  # fresh round: holdout starts LOCKED
    if REGISTRY_PATH.exists():
        REGISTRY_PATH.unlink()
    assert holdout_locked(), "holdout must start locked"

    registry = Registry()
    ledger = TrialLedger()
    verdicts: Dict[str, Any] = {}

    print("== Phase 2 experiment starting ==", flush=True)
    print(f"config_hash={config_hash()}", flush=True)

    # --- production snapshot (safety audit baseline) ---
    prod_snap = _production_snapshot()
    _write_json("production_snapshot.json", prod_snap)

    # --- data + audit + splits ---
    df, manifest = load_dataset()
    audit = audit_data(df, manifest)
    _write_json("data_audit.json", {"manifest": manifest.__dict__, "audit": audit})
    splits = chronological_splits(len(df))
    _write_json("splits.json", {
        "dev": {"start": splits.dev.start, "end": splits.dev.end,
                "warmup_start": splits.dev.warmup_start},
        "wf": {"start": splits.wf.start, "end": splits.wf.end,
               "warmup_start": splits.wf.warmup_start},
        "oos": {"start": splits.oos.start, "end": splits.oos.end,
                "warmup_start": splits.oos.warmup_start},
        "n": splits.n, "dev_range": [str(df["time"].iloc[splits.dev.start]),
                                     str(df["time"].iloc[splits.dev.end - 1])],
        "wf_range": [str(df["time"].iloc[splits.wf.start]),
                     str(df["time"].iloc[splits.wf.end - 1])],
        "oos_range": [str(df["time"].iloc[splits.oos.start]),
                      str(df["time"].iloc[splits.oos.end - 1])]})
    print(f"data: {manifest.source} bars={manifest.n_bars} "
          f"[{manifest.start} .. {manifest.end}]", flush=True)
    print(f"splits: dev=[{splits.dev.start},{splits.dev.end}) "
          f"wf=[{splits.wf.start},{splits.wf.end}) "
          f"oos=[{splits.oos.start},{splits.oos.end})", flush=True)

    dev_df = slice_segment(df, splits.dev)

    # --- Gate 1 ---
    print("-- Gate 1 (development) --", flush=True)
    g1_results = g1.run_gate1(dev_df, registry, ledger)
    g1_verdicts = g1.gate1_verdicts(g1_results)
    gate1_rows = []
    for key, res in g1_results.items():
        row = _metrics_row(key, res)
        if key in g1_verdicts:
            v = g1_verdicts[key]
            row["gate1"] = "PASS" if v.passed else "REJECTED — GATE 1"
            row["gate1_reasons"] = "; ".join(v.reasons)
            if not v.passed:
                ledger.discarded_experiments.append(
                    f"{key}:primary REJECTED — GATE 1 ({'; '.join(v.reasons)})")
        gate1_rows.append(row)
    pd.DataFrame(gate1_rows).to_csv(OUTPUTS / "gate1.csv", index=False)
    survivors = [s for s in STRATEGIES if g1_verdicts[s].passed]
    verdicts["gate1"] = {s: ("PASS" if g1_verdicts[s].passed else "REJECTED — GATE 1")
                         for s in STRATEGIES}
    print(f"Gate 1 survivors: {survivors}", flush=True)

    # --- Sensitivity + Gate 2 ---
    sens_summaries: Dict[str, Any] = {}
    for s in survivors:
        print(f"-- Sensitivity {s} --", flush=True)
        oat, heat, summary = sens.run_sensitivity(
            g1.primary_handles()[s], dev_df, registry, ledger)
        heat.to_csv(OUTPUTS / f"heatmap_{s}.csv", index=False)
        pd.DataFrame([{"variant": r.handle.variant, "exp_id": r.experiment_id,
                       **r.metrics.to_dict()} for r in oat]).to_csv(
            OUTPUTS / f"sensitivity_{s}.csv", index=False)
        v = sens.evaluate_gate2(summary)
        sens_summaries[s] = {"summary": summary.to_dict(),
                             "verdict": "PASS" if v.passed else "REJECTED — GATE 2",
                             "reasons": v.reasons}
        if not v.passed:
            ledger.discarded_experiments.append(f"{s}:primary REJECTED — GATE 2")
    verdicts["gate2"] = {s: sens_summaries[s]["verdict"] for s in survivors}
    survivors = [s for s in survivors if sens_summaries[s]["verdict"] == "PASS"]
    _write_json("sensitivity.json", sens_summaries)
    print(f"Gate 2 survivors: {survivors}", flush=True)

    # --- Walk-forward + Gate 3 ---
    wf_summaries: Dict[str, Any] = {}
    for s in survivors:
        print(f"-- Walk-forward {s} --", flush=True)
        dev_exp = g1_results[s].metrics.expectancy_r
        windows, summary = wf.run_walkforward(
            df, splits.wf, g1.primary_handles()[s], dev_exp, registry, ledger)
        pd.DataFrame([w.to_dict() for w in windows]).to_csv(
            OUTPUTS / f"walkforward_{s}.csv", index=False)
        v = wf.evaluate_gate3(summary)
        wf_summaries[s] = {"summary": summary.to_dict(),
                           "verdict": "PASS" if v.passed else "REJECTED — GATE 3",
                           "reasons": v.reasons}
        if not v.passed:
            ledger.discarded_experiments.append(f"{s}:primary REJECTED — GATE 3")
    verdicts["gate3"] = {s: wf_summaries[s]["verdict"] for s in survivors}
    survivors = [s for s in survivors if wf_summaries[s]["verdict"] == "PASS"]
    _write_json("walkforward.json", wf_summaries)
    print(f"Gate 3 survivors: {survivors}", flush=True)

    # --- OOS holdout + Gate 4 (only if survivors exist; holdout stays locked otherwise) ---
    oos_rows: List[Dict[str, Any]] = []
    validations: Dict[str, Any] = {}
    oos_results: Dict[str, Any] = {}
    if survivors:
        print("-- Opening OOS holdout (freeze receipt) --", flush=True)
        oos_df, receipt = oos_mod.open_holdout(
            df, splits.oos, manifest.dataset_hash, survivors)
        _write_json("freeze_receipt.json", receipt.__dict__)
        from .common import run_configuration
        base_oos = run_configuration(
            g1.primary_handles()["baseline"], oos_df, "oos", registry,
            ledger, gate="oos", seed=MASTER_SEED, count_ledger="none")
        oos_rows.append(_metrics_row("baseline-OOS", base_oos))
        for s in survivors:
            print(f"-- OOS {s} --", flush=True)
            ledger.oos_strategies.append(s)
            r = run_configuration(g1.primary_handles()[s], oos_df, "oos",
                                  registry, ledger, gate="oos",
                                  seed=MASTER_SEED, count_ledger="none")
            oos_results[s] = r
            oos_rows.append(_metrics_row(s, r))
        pd.DataFrame(oos_rows).to_csv(OUTPUTS / "oos.csv", index=False)

        g4: Dict[str, Any] = {}
        for s in survivors:
            v = oos_mod.evaluate_gate4(
                oos_results[s].metrics,
                _wf_summary_from(wf_summaries[s]["summary"]),
                base_oos.metrics.max_drawdown)
            g4[s] = {"verdict": "PASS" if v.passed else "REJECTED — GATE 4",
                     "reasons": v.reasons, "degradation": v.degradation}
            if not v.passed:
                ledger.discarded_experiments.append(f"{s}:primary REJECTED — GATE 4")
        verdicts["gate4"] = {s: g4[s]["verdict"] for s in survivors}
        validations["gate4"] = g4
        survivors = [s for s in survivors if g4[s]["verdict"] == "PASS"]
        print(f"Gate 4 survivors: {survivors}", flush=True)

        # --- Statistical validation + Gate 5 ---
        g5: Dict[str, Any] = {}
        for s in survivors:
            print(f"-- Validation {s} (MC/bootstrap/perm/delay/DSR) --", flush=True)
            mc, boot, perm, delay, dsr = oos_mod.run_oos_validation(
                oos_df, g1.primary_handles()[s], oos_results[s], ledger, registry)
            v = oos_mod.evaluate_gate5(oos_results[s], mc, boot, delay, dsr)
            g5[s] = {"montecarlo": mc.to_dict(), "bootstrap": boot.to_dict(),
                     "permutation": perm.to_dict(), "delay": delay.to_dict(),
                     "dsr": dsr.to_dict(),
                     "verdict": "PASS" if v.passed else "REJECTED — GATE 5",
                     "reasons": v.reasons}
            if not v.passed:
                ledger.discarded_experiments.append(f"{s}:primary REJECTED — GATE 5")
        verdicts["gate5"] = {s: g5[s]["verdict"] for s in survivors}
        validations["gate5"] = g5
        survivors = [s for s in survivors if g5[s]["verdict"] == "PASS"]
        print(f"Gate 5 survivors: {survivors}", flush=True)

        # --- Cost stress + Gate 6 ---
        g6: Dict[str, Any] = {}
        for s in survivors:
            print(f"-- Cost stress {s} --", flush=True)
            stress = oos_mod.run_cost_stress(
                oos_df, g1.primary_handles()[s], ledger, registry)
            pd.DataFrame([{"scenario": k, **v}
                          for k, v in stress.scenarios.items()]).to_csv(
                OUTPUTS / f"stress_{s}.csv", index=False)
            v = oos_mod.evaluate_gate6(stress)
            label = ("RESEARCH PROTOTYPE CANDIDATE" if v.passed
                     else "REJECTED — GATE 6")
            g6[s] = {"stress": stress.to_dict(), "verdict": label,
                     "reasons": v.reasons}
            if not v.passed:
                ledger.discarded_experiments.append(f"{s}:primary REJECTED — GATE 6")
        verdicts["gate6"] = {s: g6[s]["verdict"] for s in survivors}
        validations["gate6"] = g6
        survivors = [s for s in survivors if g6[s]["verdict"] == "RESEARCH PROTOTYPE CANDIDATE"]
        print(f"Gate 6 candidates: {survivors}", flush=True)
    else:
        verdicts["gate4"] = {}
        verdicts["gate5"] = {}
        verdicts["gate6"] = {}
        _write_json("freeze_receipt.json", {"status": "NOT OPENED — no Gate-3 survivors"})

    _write_json("validations.json", validations)
    _write_json("verdicts.json", verdicts)
    _write_json("ledger.json", ledger.to_dict())
    _write_json("environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "research_version": "2.0.0-phase2",
        "config_hash": config_hash(),
        "frozen_config": frozen_config_dict(),
        "dataset_hash": manifest.dataset_hash,
        "data_source": manifest.source,
    })
    # leakage audit + production safety are written by report helpers
    from .leakage import write_leakage_audit
    from .safety import write_safety_audit
    write_leakage_audit()
    write_safety_audit(prod_snap)

    print("== Experiment complete ==", flush=True)
    print(f"final candidates: {survivors}", flush=True)
    return {"verdicts": verdicts, "ledger": ledger.to_dict(),
            "candidates": survivors, "manifest": manifest.__dict__}


def _wf_summary_from(d: Dict[str, Any]):
    from .walkforward import WFSummary
    return WFSummary(**d)
