"""Machine-readable experiment registry, trial ledger and freeze receipts.

Every backtest run appends one JSON line to ``outputs/registry.jsonl`` with:
experiment ID, strategy, full parameter values, data range, timeframe, cost
model, risk model, entry/exit/management models, session filter, seed,
code version/commit and timestamp.

The :class:`TrialLedger` counts every materially tested configuration for
multiple-testing accounting (DSR ``n_trials``). The :class:`FreezeReceipt`
proves the OOS holdout was only opened after parameters were frozen.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RESEARCH_VERSION, config_hash

OUTPUTS = Path(__file__).resolve().parent / "outputs"
REGISTRY_PATH = OUTPUTS / "registry.jsonl"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    strategy: str            # "baseline" | "A" | "B" | "C"
    variant: str             # e.g. "primary", "RR15", "diagnostic:retest", "sens:..."
    parameters: Dict[str, Any]
    data_range: Dict[str, Any]   # start/end time, bars, segment
    timeframe: str
    cost_model: Dict[str, Any]
    risk_model: Dict[str, Any]
    entry_model: str
    exit_model: str
    management_model: str
    session_filter: str
    seed: Optional[int]
    code_version: str
    code_commit: str
    config_hash: str
    timestamp: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    gate: str = ""
    gate_result: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


class Registry:
    """Append-only registry writer/reader."""

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._counter = self._existing_count()

    def _existing_count(self) -> int:
        if not self.path.exists():
            return 0
        n = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n

    def next_id(self, strategy: str, stage: str) -> str:
        self._counter += 1
        return f"EXP-{self._counter:04d}-{strategy}-{stage}"

    def record(self, rec: ExperimentRecord) -> ExperimentRecord:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(rec.to_json() + "\n")
        return rec

    def load_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self._counter = 0


@dataclass
class TrialLedger:
    """Complete ledger of everything materially tested (nothing hidden)."""

    strategies_tested: List[str] = field(default_factory=list)
    gate1_configs: int = 0
    sensitivity_configs: int = 0
    heatmap_configs: int = 0
    wf_decisions: int = 0
    adaptive_decisions: int = 0  # 0: frozen params, no re-optimisation by design
    oos_strategies: List[str] = field(default_factory=list)
    discarded_experiments: List[str] = field(default_factory=list)

    @property
    def total_configurations(self) -> int:
        return self.gate1_configs + self.sensitivity_configs + self.heatmap_configs

    @property
    def n_trials_for_dsr(self) -> int:
        # Materially considered configurations (Gate-1 + sensitivity + heatmap).
        # Walk-forward / OOS runs reuse frozen configs and are not new trials.
        return max(1, self.total_configurations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategies_tested": self.strategies_tested,
            "gate1_configs": self.gate1_configs,
            "sensitivity_configs": self.sensitivity_configs,
            "heatmap_configs": self.heatmap_configs,
            "wf_decisions": self.wf_decisions,
            "adaptive_decisions": self.adaptive_decisions,
            "oos_strategies": self.oos_strategies,
            "discarded_experiments": self.discarded_experiments,
            "TOTAL_EXPERIMENTS": len(self.strategies_tested),
            "TOTAL_CONFIGURATIONS": self.total_configurations,
            "TOTAL_STRATEGIES": len({s.split(":")[0] for s in self.strategies_tested}),
            "TOTAL_OOS_STRATEGIES": len(self.oos_strategies),
            "N_TRIALS_FOR_DSR": self.n_trials_for_dsr,
        }


@dataclass(frozen=True)
class FreezeReceipt:
    """Proof that strategy/params/costs were frozen before the OOS was opened."""

    stage: str
    config_hash: str
    dataset_hash: str
    oos_range: Dict[str, Any]
    strategies: List[str]
    code_commit: str
    timestamp: str
    receipt_hash: str

    @staticmethod
    def create(stage: str, dataset_hash: str, oos_range: Dict[str, Any],
               strategies: List[str]) -> "FreezeReceipt":
        cfg = config_hash()
        commit = git_commit()
        ts = utc_now_iso()
        blob = json.dumps(
            {"stage": stage, "config_hash": cfg, "dataset_hash": dataset_hash,
             "oos_range": oos_range, "strategies": strategies,
             "commit": commit, "timestamp": ts},
            sort_keys=True, separators=(",", ":"),
        )
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return FreezeReceipt(stage, cfg, dataset_hash, oos_range, strategies,
                             commit, ts, digest)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)
        return path

    @staticmethod
    def load(path: Path) -> "FreezeReceipt":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return FreezeReceipt(**data)
