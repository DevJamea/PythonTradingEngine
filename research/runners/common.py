"""Shared runner utilities: strategy dispatch, single-config execution.

Every executed configuration is recorded in the machine-readable registry
with full provenance (params, data range, cost/risk models, seeds, commit).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold_trader.config import Config
from gold_trader.models import default_gold_spec

from ..backtest.engine import EngineResult, ManagementSpec, run_backtest
from ..config import (INITIAL_BALANCE, RISK_PER_TRADE, StrategyAParams,
                      StrategyBParams, StrategyCParams, config_hash)
from ..costs import BidAsk, build_bid_ask
from ..indicators_ext import atr
from ..metrics.performance import PerfMetrics, compute_performance
from ..registry import ExperimentRecord, Registry, TrialLedger, git_commit, utc_now_iso
from ..strategies import strategy_a as mod_a
from ..strategies import strategy_b as mod_b
from ..strategies import strategy_c as mod_c
from ..strategies.base import ResearchSignal, StrategyOutput
from ..strategies.baseline_wrap import BaselineSpec, generate_baseline_signals

SPEC = default_gold_spec()
PROD_CFG = Config()


def sorted_signals(signals: List[ResearchSignal]) -> List[ResearchSignal]:
    return sorted(signals, key=lambda s: (s.entry_index, s.index))


# ---------------------------------------------------------------------------
# strategy handles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Handle:
    strategy: str   # "baseline" | "A" | "B" | "C"
    variant: str    # "primary" | diagnostics | sensitivity labels
    params: Any     # None (baseline) | StrategyAParams | StrategyBParams | StrategyCParams


def _mgmt_for_signal(strategy: str, params: Any, signal: ResearchSignal) -> ManagementSpec:
    if strategy == "baseline":
        return ManagementSpec()
    if strategy == "A":
        p: StrategyAParams = params
        if p.management == "none":
            return ManagementSpec()
        return ManagementSpec(use_fixed_tp=True, be_trigger_r=p.be_trigger_r,
                              be_buffer=p.be_buffer, partial_r=p.partial_r,
                              partial_fraction=p.partial_fraction,
                              trail_mult=p.trail_atr_mult,
                              trail_activation_r=p.trail_activation_r)
    if strategy == "B":
        p = params
        if p.exit_mode == "trail":
            return ManagementSpec(use_fixed_tp=False, be_trigger_r=p.be_trigger_r,
                                  be_buffer=p.be_buffer,
                                  trail_mult=p.trail_atr_mult,
                                  trail_activation_r=p.trail_activation_r)
        return ManagementSpec()
    if strategy == "C":
        origin = (signal.meta or {}).get("params", "A")
        if origin == "B":
            return ManagementSpec()
        cp: StrategyCParams = params
        ap = cp.trend_params
        return ManagementSpec(use_fixed_tp=True, be_trigger_r=ap.be_trigger_r,
                              be_buffer=ap.be_buffer, partial_r=ap.partial_r,
                              partial_fraction=ap.partial_fraction,
                              trail_mult=ap.trail_atr_mult,
                              trail_activation_r=ap.trail_activation_r)
    raise ValueError(f"unknown strategy {strategy}")


def describe(handle: Handle) -> Tuple[str, str, str, str]:
    """(entry_model, exit_model, management_model, session_filter)."""
    if handle.strategy == "baseline":
        b = BaselineSpec()
        return b.entry_model, b.exit_model, b.management_model, b.session_filter
    if handle.strategy == "A":
        p: StrategyAParams = handle.params
        return (mod_a.a_entry_model(p), mod_a.a_exit_model(p),
                mod_a.a_management_model(p), "all permitted trading hours (00:00-23:59 UTC)")
    if handle.strategy == "B":
        p = handle.params
        return (mod_b.b_entry_model(p), mod_b.b_exit_model(p),
                mod_b.b_management_model(p), mod_b.b_session_filter(p))
    cp: StrategyCParams = handle.params
    return (mod_c.c_entry_model(cp), mod_c.c_exit_model(cp),
            mod_c.c_management_model(cp), "inherited from routed strategy")


def params_dict(handle: Handle) -> Dict[str, Any]:
    if handle.strategy == "baseline":
        return {"production_config": "Config() defaults (unmodified)",
                "note": "signal/SLTP rules byte-identical to production"}
    d = asdict(handle.params)
    if handle.strategy == "C":
        d["trend_params"] = asdict(handle.params.trend_params)
        d["compression_params"] = asdict(handle.params.compression_params)
    return d


def generate(handle: Handle, df: pd.DataFrame) -> StrategyOutput:
    if handle.strategy == "baseline":
        return generate_baseline_signals(df, PROD_CFG)
    if handle.strategy == "A":
        return mod_a.generate_a_signals(df, handle.params)
    if handle.strategy == "B":
        return mod_b.generate_b_signals(df, handle.params)
    return mod_c.generate_c_signals(df, handle.params)


def _trail_atr(df: pd.DataFrame) -> np.ndarray:
    return atr(df["high"].astype(float), df["low"].astype(float),
               df["close"].astype(float), 14).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# single-configuration execution + registry recording
# ---------------------------------------------------------------------------
@dataclass
class ConfigResult:
    handle: Handle
    output: StrategyOutput
    engine: EngineResult
    metrics: PerfMetrics
    span_days: float
    experiment_id: str


def run_configuration(
    handle: Handle,
    seg_df: pd.DataFrame,
    segment_name: str,
    registry: Registry,
    ledger: TrialLedger,
    gate: str,
    seed: Optional[int] = None,
    spread_mult: float = 1.0,
    slip_entry_pts: Optional[np.ndarray] = None,
    slip_exit_pts: Optional[np.ndarray] = None,
    delay_entry_spread_frac: float = 0.0,
    initial_balance: float = INITIAL_BALANCE,
    risk_per_trade: float = RISK_PER_TRADE,
    count_ledger: str = "none",  # "gate1" | "sens" | "heat" | "none"
) -> ConfigResult:
    """Generate signals, run the research engine, compute metrics, register."""
    trade_start = int(seg_df.attrs.get("trade_start_pos", 0))
    t0 = pd.to_datetime(seg_df["time"], utc=True)
    # span over the TRADEABLE part only
    pos = (seg_df["_pos"].to_numpy(dtype=int) if "_pos" in seg_df.columns
           else np.arange(len(seg_df)))
    tradeable = seg_df[pos >= trade_start]
    if len(tradeable) >= 2:
        span_days = max(
            (pd.to_datetime(tradeable["time"].iloc[-1], utc=True)
             - pd.to_datetime(tradeable["time"].iloc[0], utc=True)).total_seconds() / 86400.0,
            1 / 96)
    else:
        span_days = 1 / 96

    output = generate(handle, seg_df)
    signals = sorted_signals(output.signals)
    ba = build_bid_ask(seg_df, spread_mult)
    engine = run_backtest(
        seg_df, signals, ba, SPEC, initial_balance, risk_per_trade,
        mgmt=lambda s: _mgmt_for_signal(handle.strategy, handle.params, s),
        trade_start_pos=trade_start, atr_for_trail=_trail_atr(seg_df),
        slip_entry_pts=slip_entry_pts, slip_exit_pts=slip_exit_pts,
        delay_entry_spread_frac=delay_entry_spread_frac)
    metrics = compute_performance(engine.trades, initial_balance, span_days)

    entry_model, exit_model, mgmt_model, session = describe(handle)
    exp_id = registry.next_id(handle.strategy, gate)
    registry.record(ExperimentRecord(
        experiment_id=exp_id, strategy=handle.strategy, variant=handle.variant,
        parameters=params_dict(handle),
        data_range={"segment": segment_name,
                    "start": str(tradeable["time"].iloc[0]) if len(tradeable) else None,
                    "end": str(tradeable["time"].iloc[-1]) if len(tradeable) else None,
                    "bars_tradeable": int(len(tradeable)),
                    "bars_warmup": int((pos < trade_start).sum()),
                    "warmup_bars_excluded_from_trading": True},
        timeframe="M15",
        cost_model={"model": "bid_ask", "spread_mult": spread_mult,
                    "entry_slippage_pts": ("uniform_1_5" if slip_entry_pts is not None else 0),
                    "exit_slippage_pts": ("uniform_1_5" if slip_exit_pts is not None else 0),
                    "delay_entry_spread_frac": delay_entry_spread_frac,
                    "commission": "unavailable (recorded 0, NOT invented)",
                    "swap": "unavailable (recorded 0, NOT invented)"},
        risk_model={"initial_balance": initial_balance, "risk_per_trade": risk_per_trade,
                    "max_open_positions": 1,
                    "sizing": "production calculate_position_size (read-only)"},
        entry_model=entry_model, exit_model=exit_model,
        management_model=mgmt_model, session_filter=session,
        seed=seed, code_version="research-2.0.0-phase2",
        code_commit=git_commit(), config_hash=config_hash(),
        timestamp=utc_now_iso(), metrics=metrics.to_dict(), gate=gate))

    ledger.strategies_tested.append(f"{handle.strategy}:{handle.variant}")
    if count_ledger == "gate1":
        ledger.gate1_configs += 1
    elif count_ledger == "sens":
        ledger.sensitivity_configs += 1
    elif count_ledger == "heat":
        ledger.heatmap_configs += 1
    return ConfigResult(handle, output, engine, metrics, span_days, exp_id)


def with_params(handle: Handle, variant: str, **overrides) -> Handle:
    """Return a copy of the handle with dataclass params replaced."""
    if handle.strategy == "baseline":
        raise ValueError("baseline has no research params")
    return Handle(handle.strategy, variant, replace(handle.params, **overrides))
