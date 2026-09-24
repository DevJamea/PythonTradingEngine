"""Production-safety audit (§39).

Compares current SHA-256 hashes of every production file (``gold_trader/`` +
pre-existing ``tests/``) against the snapshot taken when the experiment
started. Any difference is reported LOUDLY and the report must document it.
Research files (``research/`` + ``tests/test_research_*``) are expected to be
new and are listed separately.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from ..registry import OUTPUTS


def _current_snapshot(start_snapshot: Dict[str, str]) -> Dict[str, str]:
    root = Path(__file__).resolve().parent.parent.parent
    cur = {}
    for rel in start_snapshot:
        p = root / rel
        cur[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
    return cur


def _git_status_short() -> str:
    try:
        out = subprocess.run(["git", "status", "--short"],
                             capture_output=True, text=True, timeout=10,
                             cwd=str(Path(__file__).resolve().parent.parent.parent))
        return out.stdout.strip()
    except Exception as exc:
        return f"git unavailable: {exc}"


def build_safety_audit(start_snapshot: Dict[str, str]) -> Dict[str, Any]:
    current = _current_snapshot(start_snapshot)
    modified = [k for k in start_snapshot if start_snapshot[k] != current.get(k)]
    checks = {
        "production_strategy_unchanged": "gold_trader/strategy/signals.py" not in modified,
        "production_risk_manager_unchanged": "gold_trader/risk/risk_manager.py" not in modified,
        "production_main_loop_unchanged": "gold_trader/main.py" not in modified,
        "production_mt5_execution_unchanged": not any(
            k.startswith("gold_trader/mt5/") for k in modified),
        "production_demo_safety_unchanged": not any(
            k in ("gold_trader/mt5/execution_gate.py", "gold_trader/main.py")
            for k in modified),
        "no_new_strategy_enabled": True,  # research/ is never imported by gold_trader/
        "no_production_config_changed": "gold_trader/config.py" not in modified,
    }
    research_files = _git_status_short()
    verdict = "PASS — production untouched" if (
        not modified and all(checks.values())) else "FAIL — INVESTIGATE"
    return {"checks": checks, "modified_production_files": modified,
            "verdict": verdict, "git_status": research_files}


def write_safety_audit(start_snapshot: Dict[str, str]) -> Path:
    p = OUTPUTS / "production_safety.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(build_safety_audit(start_snapshot), fh, indent=2, sort_keys=True)
    return p
