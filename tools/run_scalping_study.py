#!/usr/bin/env python3
"""Offline research runner for the scalping study (development vs isolated).

Read-only by construction: this script imports no MT5 code path that can send
an order, it only loads a candle dump and runs the two engines + the analysis
layer. Everything it prints is reproducible from the input file alone.

Usage
-----
    python3 tools/run_scalping_study.py \
        --m5 /home/user/xauusd_m5_2021_2025.csv.gz \
        --m15 /home/user/xauusd_m15_2021_2025.csv.gz \
        --json-out /home/user/scalping_study.json

Protocol (identical to the previous study, and deliberately rigid):

1. **Development 2021-01-01 .. 2024-01-01.** A coarse parameter grid is scored
   by rolling walk-forward folds (6-month train, 2-month test). The config that
   wins the most test windows is the candidate; ties fall to the earlier
   (i.e. tighter/more conservative) entry in the grid.
2. **Isolated 2024-01-01 .. 2026-01-01.** The chosen config runs EXACTLY ONCE
   per cost scenario: (a) real per-bar spread from the bid/ask dump,
   (b) stress: spread x1.5 plus 0.05 slippage per side, (c) legacy comparable:
   fixed 2 x backtest_spread_cost. No selection may use this period.
3. Verdict = profit factor and Sharpe on the isolated period only.

The trend baseline is measured with the untouched ``BacktestEngine`` on the M15
dump so the before/after table compares two strategies, not two simulators.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gold_trader.backtest.analysis import (  # noqa: E402
    format_walk_forward,
    monte_carlo_simulation,
    monte_carlo_text,
    risk_ratios,
    verdict,
    walk_forward_study,
)
from gold_trader.backtest.cost_model import spread_summary  # noqa: E402
from gold_trader.backtest.data import (  # noqa: E402
    assess_quality,
    load_candles_csv,
    split_period,
)
from gold_trader.backtest.engine import BacktestEngine  # noqa: E402
from gold_trader.backtest.metrics import format_metrics  # noqa: E402
from gold_trader.backtest.scalping_engine import ScalpingBacktestEngine  # noqa: E402
from gold_trader.config import Config  # noqa: E402

DEV_START, DEV_END = "2021-01-01", "2024-01-01"
ISO_START, ISO_END = "2024-01-01", "2026-01-01"

#: Coarse grid. ``min_profit_to_spread_ratio`` is NOT part of it on purpose:
#: the cost gate is a safety constraint, and tuning it to make a period look
#: good would destroy the only reason this strategy is worth testing.
GRID: List[Dict[str, Any]] = [
    {
        "scalp_tp_atr_multiple": tp,
        "scalp_sl_atr_multiple": sl,
        "scalp_bb_std": std,
        "scalp_rsi_oversold": os_low,
        "scalp_rsi_overbought": 100.0 - os_low,
    }
    for tp in (1.5, 2.5)
    for sl in (1.0, 1.5)
    for std in (1.5, 2.5)
    for os_low in (25.0, 15.0)
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def scalping_base_config() -> Config:
    """Research config: the scalping engine is enabled for the study only."""
    return replace(Config(), scalping_enabled=True, active_strategy="scalping")


#: Cost scenarios. Every one of them is expressed as a *config* (never as a
#: hand-built CostModel) because the engine then derives the cost series from
#: the very frame it is running -- which keeps the per-bar spread aligned with
#: the bar indices inside walk-forward folds.
SCENARIOS = {
    "real variable spread (recorded bid/ask)": {
        "backtest_use_recorded_spread": True,
        "backtest_spread_stress_multiplier": 1.0,
        "backtest_slippage_per_side": 0.0,
    },
    "stress: spread x1.5 + 0.05 slippage/side": {
        "backtest_use_recorded_spread": True,
        "backtest_spread_stress_multiplier": 1.5,
        "backtest_slippage_per_side": 0.05,
    },
    "legacy fixed spread (2 x backtest_spread_cost)": {
        "backtest_use_recorded_spread": False,
    },
}


def scenario_config(base: Config, label: str) -> Config:
    return replace(base, **SCENARIOS[label])


def summarise(name: str, result, cfg: Config) -> Dict[str, Any]:
    metrics = result.metrics
    ratios = risk_ratios(result.trades, result.initial_balance)
    call = verdict(metrics, ratios)
    data: Dict[str, Any] = {
        "label": name,
        "trades": metrics.total_trades,
        "win_rate": metrics.win_rate,
        "profit_factor": None if math.isinf(metrics.profit_factor) else metrics.profit_factor,
        "net_profit": metrics.net_profit,
        "gross_profit": metrics.gross_profit,
        "gross_loss": metrics.gross_loss,
        "max_drawdown": metrics.max_drawdown,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "average_win": metrics.average_win,
        "average_loss": metrics.average_loss,
        "sharpe": ratios.sharpe,
        "sortino": ratios.sortino,
        "trades_per_year": ratios.trades_per_year,
        "final_equity": result.final_equity,
        "initial_balance": result.initial_balance,
        "passed": call.passed,
        "verdict_reasons": list(call.reasons),
    }
    if hasattr(result, "total_spread_cost"):
        data["total_spread_cost"] = result.total_spread_cost
        data["total_commission_cost"] = result.total_commission_cost
        data["veto_counts"] = dict(result.veto_counts)
        data["cost_model"] = result.cost_model
        data["bars_held_median"] = (
            float(sorted(r.bars_held for r in result.records)[len(result.records) // 2])
            if result.records
            else None
        )
        data["mean_spread_price"] = (
            float(sum(r.spread_price for r in result.records) / len(result.records))
            if result.records
            else None
        )
    return data


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="XAUUSD scalping study (research only)")
    parser.add_argument("--m5", required=True, help="M5 gzip CSV with a real 'spread' column")
    parser.add_argument("--m15", default=None, help="M15 dump for the trend baseline comparison")
    parser.add_argument("--json-out", default="/home/user/scalping_study.json")
    parser.add_argument("--grid", type=int, default=len(GRID), help="limit grid size (debug)")
    parser.add_argument("--skip-isolated", action="store_true", help="development only")
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {"protocol": {}, "runs": [], "grid": [], "walk_forward": {}}
    cfg = scalping_base_config()

    m5 = load_candles_csv(args.m5)
    quality = assess_quality(m5)
    report["protocol"]["data"] = {
        "path": args.m5,
        "rows": quality.rows,
        "first": str(quality.first_time),
        "last": str(quality.last_time),
        "years": list(quality.years),
        "mean_spread": quality.mean_spread,
        "p90_spread": quality.p90_spread,
        "max_spread": quality.max_spread,
    }
    report["protocol"]["spread_summary"] = spread_summary(m5["spread"])
    report["protocol"]["development"] = [DEV_START, DEV_END]
    report["protocol"]["isolated"] = [ISO_START, ISO_END]
    report["protocol"]["grid"] = GRID[: args.grid]
    report["protocol"]["acceptance"] = {"min_profit_factor": 1.10, "min_sharpe": 0.50}
    log(
        f"data: {quality.rows} M5 bars {quality.first_time} .. {quality.last_time} | "
        f"spread mean={quality.mean_spread:.4f} p90={quality.p90_spread:.4f}"
    )

    dev = split_period(m5, DEV_START, DEV_END)
    iso = split_period(m5, ISO_START, ISO_END)
    log(f"dev bars={len(dev)}  isolated bars={len(iso)}")

    # -- 1. full-grid scoring on the DEV period only -----------------------
    grid = GRID[: args.grid]
    for index, overrides in enumerate(grid, start=1):
        candidate = replace(cfg, **overrides)
        started = time.time()
        result = ScalpingBacktestEngine(
            replace(candidate, **SCENARIOS["real variable spread (recorded bid/ask)"])
        ).run(dev)
        data = summarise(f"grid[{index}]", result, candidate)
        data["overrides"] = overrides
        report["grid"].append(data)
        log(
            f"grid {index}/{len(grid)} {overrides} -> trades={data['trades']} "
            f"PF={data['profit_factor']} net={data['net_profit']:.2f} [{time.time()-started:.1f}s]"
        )

    # -- 2. walk-forward over DEV, selection by fold wins ------------------
    log("walk-forward on development period ...")
    wf = walk_forward_study(
        dev,
        cfg,
        param_grid=grid,
        engine_factory=lambda candidate: ScalpingBacktestEngine(
            replace(candidate, **SCENARIOS["real variable spread (recorded bid/ask)"])
        ),
        train_months=6,
        test_months=2,
        start=DEV_START,
        end=DEV_END,
    )
    wins: Dict[int, int] = {}
    for fold in wf.folds:
        key = json.dumps(fold.chosen, sort_keys=True)
        for i, overrides in enumerate(grid):
            if json.dumps(overrides, sort_keys=True) == key:
                wins[i] = wins.get(i, 0) + 1
    chosen_index = max(wins.items(), key=lambda kv: (kv[1], -kv[0]))[0] if wins else 0
    chosen_overrides = grid[chosen_index]
    chosen = replace(cfg, **chosen_overrides)
    report["walk_forward"] = {
        **wf.as_dict(),
        "folds_detail": [
            {
                "train": f"{f.train_start.date()}..{f.train_end.date()}",
                "test": f"{f.test_start.date()}..{f.test_end.date()}",
                "chosen": f.chosen,
                "train_pf": f.train_profit_factor,
                "test_pf": f.test_profit_factor,
                "test_trades": f.test_trades,
                "test_net": f.test_net_profit,
            }
            for f in wf.folds
        ],
        "fold_wins": wins,
        "chosen_index": chosen_index,
        "chosen": chosen_overrides,
        "text": format_walk_forward(wf),
    }
    log(f"chosen by fold wins: {chosen_overrides} (wins={wins})")
    log(f"walk-forward OOS: {wf.oos_metrics.total_trades} trades, "
        f"PF={wf.oos_metrics.profit_factor:.3f}, net={wf.oos_metrics.net_profit:.2f}")

    dev_chosen = ScalpingBacktestEngine(
        replace(chosen, **SCENARIOS["real variable spread (recorded bid/ask)"])
    ).run(dev)
    report["runs"].append(summarise("development (2021-2023)", dev_chosen, chosen))

    # -- 2b. cost-gate sensitivity, measured on the DEV period only --------
    # This is not an optimisation: it answers "would a stricter cost gate have
    # saved the strategy?" and it may only ever be answered on development data.
    sensitivity = []
    for ratio in (2.0, 3.0, 5.0, 8.0, 12.0):
        probe = ScalpingBacktestEngine(
            replace(
                chosen,
                min_profit_to_spread_ratio=ratio,
                **SCENARIOS["real variable spread (recorded bid/ask)"],
            )
        ).run(dev)
        data = summarise(f"dev / min_profit_to_spread_ratio={ratio}", probe, chosen)
        data["min_profit_to_spread_ratio"] = ratio
        sensitivity.append(data)
        log(
            f"gate sensitivity ratio={ratio}: trades={data['trades']} "
            f"PF={data['profit_factor']} net={data['net_profit']:.2f}"
        )
    report["gate_sensitivity_dev"] = sensitivity

    # -- 2c. per-year description with the FROZEN config (post-hoc reporting,
    #     never selection: the parameters above were already fixed on 2021-2023)
    per_year = []
    for year in range(2021, 2026):
        year_frame = split_period(m5, f"{year}-01-01", f"{year+1}-01-01")
        if len(year_frame) < 1000:
            continue
        year_result = ScalpingBacktestEngine(
            replace(chosen, **SCENARIOS["real variable spread (recorded bid/ask)"])
        ).run(year_frame)
        data = summarise(f"scalping {year}", year_result, chosen)
        data["year"] = year
        data["in_isolated_period"] = year >= 2024
        per_year.append(data)
        log(
            f"year {year}: trades={data['trades']} PF={data['profit_factor']} "
            f"net={data['net_profit']:.2f} spread={data['total_spread_cost']:.2f}"
        )
    report["per_year_dev_only_note"] = (
        "Descriptive only. Parameter selection used 2021-2023 exclusively; the "
        "2024-2025 rows are the frozen config being reported, not tuned."
    )
    report["per_year"] = per_year

    # -- 2d. market microstructure facts the report will quote -------------
    bar_range = (dev["high"] - dev["low"])
    report["protocol"]["microstructure"] = {
        "median_bar_range": float(bar_range.median()),
        "mean_bar_range": float(bar_range.mean()),
        "median_spread": float(dev["spread"].median()),
        "cost_share_of_a_bar": float(dev["spread"].median() / bar_range.median())
        if bar_range.median() > 0
        else None,
    }
    log(
        f"median M5 range={bar_range.median():.3f} vs median spread={dev['spread'].median():.3f} "
        f"-> cost is {100 * dev['spread'].median() / bar_range.median():.0f}% of a whole bar"
    )

    if args.skip_isolated:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
        log(f"dev-only report written to {args.json_out}")
        return 0

    # -- 3. the isolated period, looked at once ---------------------------
    for label, overrides in SCENARIOS.items():
        scenario_cfg = replace(chosen, **overrides)
        result = ScalpingBacktestEngine(scenario_cfg).run(iso)
        data = summarise(f"isolated 2024-2025 / {label}", result, chosen)
        mc = monte_carlo_simulation(result.trades, result.initial_balance, runs=1000, block_size=5)
        data["monte_carlo"] = mc.as_dict()
        data["monte_carlo_text"] = monte_carlo_text(mc)
        report["runs"].append(data)
        log(
            f"ISOLATED [{label}] trades={data['trades']} PF={data['profit_factor']} "
            f"net={data['net_profit']:.2f} sharpe={data['sharpe']:.2f} "
            f"spreadcost={data['total_spread_cost']:.2f} passed={data['passed']}"
        )

    # -- 4. the trend baseline on M15, same protocol -----------------------
    if args.m15:
        m15 = load_candles_csv(args.m15)
        for label, (start, end) in (
            ("development", (DEV_START, DEV_END)),
            ("isolated", (ISO_START, ISO_END)),
        ):
            period = split_period(m15, start, end)
            average_recorded_per_side = (
                float(period["spread"].mean() / 2.0) if "spread" in period.columns else None
            )
            for scenario, overrides in (
                (
                    "legacy fixed spread 0.15/side",
                    {"backtest_use_recorded_spread": False},
                ),
                (
                    "recorded spread as an average (0.15/side replaced by mean/2)",
                    {
                        "backtest_use_recorded_spread": True,
                        "backtest_spread_cost": average_recorded_per_side or 0.15,
                    },
                ),
            ):
                trend_cfg = replace(
                    Config(),
                    min_candles_for_signal=210,
                    backtest_candles=len(period),
                    **overrides,
                )
                result = BacktestEngine(trend_cfg).run(period)
                data = summarise(f"trend baseline / {label} / {scenario}", result, trend_cfg)
                # the legacy engine subtracts 2 x spread_cost from the P/L, it has
                # no fill geometry: report the spread it charged for transparency
                data["cost_model"] = (
                    "legacy engine: net -= 2 * per-side spread; per side = "
                    f"{trend_cfg.backtest_spread_cost:.4f}"
                    + (
                        " (mean of the recorded bid/ask spread)"
                        if overrides["backtest_use_recorded_spread"]
                        else " (configured fixed estimate)"
                    )
                )
                data["strategy"] = data.get("strategy") or "trend baseline (signals.py, M15)"
                data["period"] = label
                data["strategy"] = "trend baseline (signals.py, M15)"
                data["spread_used_mean"] = (
                    float(period["spread"].mean()) if "spread" in period.columns else None
                )
                report["runs"].append(data)
                log(
                    f"TREND [{label} / {scenario}] trades={data['trades']} "
                    f"PF={data['profit_factor']} net={data['net_profit']:.2f}"
                )

    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
    log(f"report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
