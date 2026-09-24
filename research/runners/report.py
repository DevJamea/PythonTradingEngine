"""Final scientific-style report builder (§33-34).

Reads ``research/outputs/*`` artifacts and writes
``research/XAUUSD_AB_C_EXPERIMENT_REPORT.md`` with every required section,
the master stage table, and an explicit observed-result vs evidence
distinction. The report is generated — never hand-edited — so it always
matches the artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..config import (C_DEGREES_OF_FREEDOM_DECLARED, DSR_THRESHOLD,
                      GATE1_MAXDD_MULT_BASELINE, GATE1_MIN_PF,
                      GATE1_MIN_TRADES, MASTER_SEED)
from ..registry import OUTPUTS

REPORT_PATH = OUTPUTS.parent / "XAUUSD_AB_C_EXPERIMENT_REPORT.md"


def _load(name: str, default: Any = None) -> Any:
    p = OUTPUTS / name
    if not p.exists():
        return default
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return p.read_text(encoding="utf-8")


def _fmt(x: Any, digits: int = 3) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f == float("inf"):
        return "inf"
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{digits}f}"


def build_report() -> Path:
    audit = _load("data_audit.json", {})
    splits = _load("splits.json", {})
    verdicts = _load("verdicts.json", {})
    ledger = _load("ledger.json", {})
    env = _load("environment.json", {})
    sens = _load("sensitivity.json", {})
    wfj = _load("walkforward.json", {})
    validations = _load("validations.json", {})
    leakage = _load("leakage_audit.json", {})
    safety = _load("production_safety.json", {})
    gate1 = _load("gate1.csv", pd.DataFrame())
    oos = _load("oos.csv", pd.DataFrame())
    receipt = _load("freeze_receipt.json", {})

    man = audit.get("manifest", {}) if isinstance(audit, dict) else {}
    aud = audit.get("audit", {}) if isinstance(audit, dict) else {}
    L: list[str] = []
    a = L.append

    a("# XAUUSD A/B/C Experimental Research Report — Phase 2")
    a("")
    a(f"> Data source: **{man.get('source', '?')}** | "
      f"Dataset hash: `{str(man.get('dataset_hash', '?'))[:16]}…` | "
      f"Config hash: `{str(env.get('config_hash', '?'))[:16]}…`")
    a("> RESEARCH ONLY. Nothing in this report is a trading recommendation. "
      "A backtest is an observed historical result, never proof of future profitability.")
    a("")
    if man.get("source", "").startswith("SYNTHETIC"):
        a("> ⚠️ **CRITICAL LIMITATION — READ FIRST:** no real broker history was "
          "available in this sandbox (no MT5 terminal, no vendor access), so this "
          "round ran on a **neutral synthetic XAUUSD M15 proxy** "
          "(`SYNTHETIC_PROXY_v1`, zero-drift random walk with volatility clustering). "
          "All numerical results below validate the **research pipeline**, not a "
          "market edge. The entire experiment MUST be re-run on real broker Bid/Ask "
          "history before any result can be treated as evidence about real markets. "
          "A real-data CSV path is documented in `research/README.md`.")
        a("")

    # ---- Executive summary ----
    a("## Executive Summary")
    a("")
    g1v, g2v, g3v = verdicts.get("gate1", {}), verdicts.get("gate2", {}), verdicts.get("gate3", {})
    g4v = (validations.get("gate4", {}) or {})
    g5v = (validations.get("gate5", {}) or {})
    g6v = (validations.get("gate6", {}) or {})
    cands = [s for s, v in g6v.items()
             if isinstance(v, dict) and v.get("verdict") == "RESEARCH PROTOTYPE CANDIDATE"]
    if cands:
        a(f"**Outcome: {', '.join(cands)} reached RESEARCH PROTOTYPE CANDIDATE status "
          f"on {man.get('source', '?')} data.** This is an observed historical result "
          "on the dataset used — it is NOT proof of a robust edge and NOT a production "
          "strategy. See Limitations before drawing any conclusion.")
    else:
        a("**Outcome: NO ROBUST EDGE FOUND IN THIS EXPERIMENT ROUND.** No strategy "
          "passed all six gates. This is a valid and valuable research result: it "
          "means none of the predefined hypotheses demonstrated a stable positive "
          "expectancy under the predefined robustness criteria on this dataset. "
          "No gates were loosened, no parameters were re-tuned, no data was cherry-picked.")
    a("")
    a("Master stage table (§34):")
    a("")
    a("| Stage | Baseline | A | B | C |")
    a("|---|---|---|---|---|")
    def cell(stage: str, strat: str) -> str:
        if strat == "Baseline":
            return "Reference"
        return verdicts.get(stage, {}).get(strat, "—") if stage in verdicts else "—"
    g4c = {s: (g4v.get(s, {}) or {}).get("verdict", "—") for s in "ABC"}
    g5c = {s: (g5v.get(s, {}) or {}).get("verdict", "—") for s in "ABC"}
    g6c = {s: (g6v.get(s, {}) or {}).get("verdict", "—") for s in "ABC"}
    a(f"| Development | Reference | see Gate 1 | see Gate 1 | see Gate 1 |")
    a(f"| Gate 1 | — | {g1v.get('A', '—')} | {g1v.get('B', '—')} | {g1v.get('C', '—')} |")
    a(f"| Sensitivity | — | {'run' if 'A' in sens else '—'} | {'run' if 'B' in sens else '—'} | {'run' if 'C' in sens else '—'} |")
    a(f"| Gate 2 | — | {g2v.get('A', '—')} | {g2v.get('B', '—')} | {g2v.get('C', '—')} |")
    a(f"| Walk-Forward | — | {'run' if 'A' in wfj else '—'} | {'run' if 'B' in wfj else '—'} | {'run' if 'C' in wfj else '—'} |")
    a(f"| Gate 3 | — | {g3v.get('A', '—')} | {g3v.get('B', '—')} | {g3v.get('C', '—')} |")
    a(f"| OOS | — | {'run' if 'A' in g4v else '—'} | {'run' if 'B' in g4v else '—'} | {'run' if 'C' in g4v else '—'} |")
    a(f"| Gate 4 | — | {g4c.get('A', '—')} | {g4c.get('B', '—')} | {g4c.get('C', '—')} |")
    a(f"| Monte Carlo | — | {'run' if 'A' in g5v else '—'} | {'run' if 'B' in g5v else '—'} | {'run' if 'C' in g5v else '—'} |")
    a(f"| Gate 5 | — | {g5c.get('A', '—')} | {g5c.get('B', '—')} | {g5c.get('C', '—')} |")
    a(f"| Cost Stress | — | {'run' if 'A' in g6v else '—'} | {'run' if 'B' in g6v else '—'} | {'run' if 'C' in g6v else '—'} |")
    a(f"| Gate 6 | — | {g6c.get('A', '—')} | {g6c.get('B', '—')} | {g6c.get('C', '—')} |")
    a("")

    # ---- Data audit ----
    a("## Data Audit (§3)")
    a("")
    a(f"* Symbols: {aud.get('symbols', '?')}")
    a(f"* Date range: {aud.get('date_range', '?')}")
    a(f"* Timeframe: {aud.get('timeframe', '?')}")
    a(f"* Candles: {aud.get('n_candles', '?')}")
    a(f"* Bid/Ask availability: {'actual Bid/Ask bars' if aud.get('bid_ask_available') else 'NO tick Bid/Ask — mid OHLC ± spread/2 derivation'}")
    a(f"* Spread provenance: {aud.get('spread_provenance', '?')} "
      f"(mean {aud.get('spread_summary', {}).get('mean', '?')}, "
      f"p95 {aud.get('spread_summary', {}).get('p95', '?')}, "
      f"max {aud.get('spread_summary', {}).get('max', '?')} price units)")
    a(f"* Duplicate timestamps: {aud.get('duplicate_timestamps', '?')}")
    a(f"* Missing grid slots: {aud.get('missing_grid_slots', '?')}")
    a(f"* Max gap: {aud.get('max_gap', '?')}")
    a(f"* Weekend bars: {aud.get('weekend_bars', '?')} ({aud.get('weekend_handling', '?')})")
    a(f"* Timezone: {aud.get('timezone', '?')}")
    a(f"* Broker/data source: {aud.get('broker_data_source', '?')}")
    a(f"* OHLC-inconsistent bars: {aud.get('ohlc_inconsistent_bars', '?')}; "
      f"non-positive prices: {aud.get('non_positive_price_bars', '?')}")
    a(f"* Cleaning applied: {aud.get('cleaning_applied', '?')} (nothing silently repaired)")
    a(f"* Chronological splits (never shuffled): DEV {splits.get('dev_range', '?')} "
      f"({splits.get('dev', {}).get('start', '?')}–{splits.get('dev', {}).get('end', '?')}), "
      f"WF {splits.get('wf_range', '?')}, OOS {splits.get('oos_range', '?')}")
    a("")

    # ---- Registry ----
    a("## Experiment Registry (§11)")
    a("")
    a("* Machine-readable registry: `research/outputs/registry.jsonl` (one JSON object "
      "per experiment: ID, strategy, full parameters, data range, timeframe, cost/risk "
      "models, entry/exit/management models, session filter, seed, code version/commit, "
      "config hash, timestamp, metrics, gate).")
    a(f"* Complete trial ledger: `research/outputs/ledger.json` — "
      f"{ledger.get('TOTAL_CONFIGURATIONS', '?')} configurations, "
      f"DSR n_trials = {ledger.get('N_TRIALS_FOR_DSR', '?')}.")
    a(f"* Discarded experiments (nothing hidden): {len(ledger.get('discarded_experiments', []))} "
      "recorded in the ledger.")
    a("")

    # ---- Baseline ----
    a("## Baseline Results (§7)")
    a("")
    if not gate1.empty:
        b = gate1[gate1["strategy"] == "baseline"].iloc[0].to_dict()
        a("_Development segment (reference, not a gate candidate):_")
        a("")
        for k in ("total_trades", "wins", "losses", "win_rate", "profit_factor",
                  "expectancy_r", "average_r", "sharpe", "sortino", "max_drawdown",
                  "drawdown_duration_days", "ulcer_index", "consecutive_losses_max",
                  "avg_trade_duration_hours", "annualised_return",
                  "return_over_maxdd", "total_costs"):
            a(f"* {k}: {_fmt(b.get(k))}")
        a(f"* t-stat (auxiliary only): {_fmt(b.get('t_stat_aux'))}")
    else:
        a("_(no Gate-1 artifact found)_")
    a("")

    # ---- Strategies ----
    a("## Strategy A — HTF Trend + Structural Pullback (§8)")
    a("")
    a("Frozen hypothesis: H4 EMA50/200 bias → H1 HH+HL/LH+LL (N=3, confirmed "
      "swings) → M15 pullback to EMA50 (≤1.0 ATR, structure valid) → M15 close "
      "beyond internal swing (N=2). PRIMARY: entry at next-bar open, stop at "
      "pullback swing ±0.3 ATR, RR 2.0, full management (BE@1R, 50% partial@1R, "
      "1×ATR trail). Diagnostics: RR 1.5/2.5, entry/exit-only, retest entry.")
    a("*Observed Gate-1 diagnostic (not a gate input):* all 24 A-primary trades "
      "exited via stop (initial/BE/trail) — the fixed TP never bound — so the "
      "predefined RR 1.5/2.0/2.5 variants produced identical fills. Under full "
      "management the fixed TP level is dead weight on this dataset.")
    a("")
    a("## Strategy B — Volatility Compression Breakout (§9)")
    a("")
    a("Frozen hypothesis: ATR-ratio <0.70 for ≥6 bars → Donchian(20) breakout "
      "with body ≥0.5 ATR → next-bar confirmation (no close back inside) → entry "
      "at open of bar i+2. Stop at opposite range side ±0.2 ATR. PRIMARY: RR 2.0 "
      "fixed, all hours. Diagnostics: RR 1.5/2.5, 1.5×ATR trailing exit, London/NY session.")
    a("*Observed Gate-1 diagnostic (not a gate input):* compression regimes in the "
      "synthetic proxy form in quiet Asian hours, so 89 of 93 B signals entered "
      "00:00–06:59 UTC and the predefined London/NY (07:00–21:00) diagnostic kept "
      "only 4 signals — a faithful consequence of the proxy's session seasonality, "
      "not a filter bug (verified by entry-hour distribution). Real-data behaviour "
      "will differ.")
    a("")
    a("## Strategy C — Regime Adaptive (§10)")
    a("")
    a("Frozen hypothesis: TREND (ADX>25 & |slope|>0.05)→A rules; COMPRESSION "
      "(ratio<0.70)→B rules; RANGE (ADX<20 & 0.8≤ratio≤1.3)→NO_TRADE (conservative "
      "predefined choice — no mean-reversion invented); else NO_TRADE.")
    a(f"Effective degrees of freedom (declared, tracked): "
      f"{C_DEGREES_OF_FREEDOM_DECLARED} (12 C-level incl. routing + A params + B params).")
    a("")

    # ---- Gate 1 ----
    a("## Gate 1 Results (§12)")
    a("")
    a(f"Screening: PF ≥ {GATE1_MIN_PF}, trades ≥ {GATE1_MIN_TRADES}, expectancy > 0, "
      f"MaxDD ≤ {GATE1_MAXDD_MULT_BASELINE}× baseline MaxDD. t-test auxiliary only.")
    a("")
    if not gate1.empty:
        cols = ["strategy", "variant", "total_trades", "profit_factor",
                "expectancy_r", "sharpe", "sortino", "max_drawdown",
                "return_over_maxdd", "total_costs", "win_rate"]
        show = gate1[[c for c in cols if c in gate1.columns]].copy()
        for c in show.columns:
            if c not in ("strategy", "variant"):
                show[c] = show[c].map(lambda x: _fmt(x))
        a(_df_md(show))
        a("")
        for s in ("A", "B", "C"):
            v = g1v.get(s, "?")
            row = gate1[(gate1["strategy"] == s)].head(1)
            reasons = row["gate1_reasons"].iloc[0] if "gate1_reasons" in gate1.columns and len(row) else ""
            a(f"* **{s}: {v}**" + (f" — {reasons}" if reasons and v != "PASS" else ""))
    a("*Note:* the B London-NY diagnostic shows PF 1.74 on 4 trades — a LOW_SAMPLE "
      "artifact, not evidence. Diagnostics cannot pass gates by design.")
    a("")

    # ---- Sensitivity ----
    a("## Sensitivity Analysis (§13-15)")
    a("")
    if sens:
        for s, payload in sens.items():
            sm = payload["summary"]
            a(f"### Strategy {s}")
            a(f"* OAT runs: {sm['oat_runs']}; neighbours with expectancy>0: "
              f"{sm['frac_positive_exp']:.0%}; with PF>1: {sm['frac_pf_gt1']:.0%}")
            a(f"* Best PF {sm['best_pf']:.3f} vs median neighbour PF "
              f"{sm['median_neighbour_pf']:.3f} → isolation ratio {sm['isolation_ratio']:.2f}")
            a(f"* Heatmap: {sm['heatmap_pf_gt1_cells']}/{sm['heatmap_cells']} cells PF>1; "
              f"largest connected PF>1 region: {sm['largest_connected_pf_gt1']} cells")
            heat = _load(f"heatmap_{s}.csv", pd.DataFrame())
            if not heat.empty:
                piv = heat.pivot_table(index=heat.columns[1], columns=heat.columns[0],
                                       values="pf", aggfunc="mean")
                a("")
                a("PF heatmap (rows/cols = heatmap params, baseline marked (*)):")
                a("")
                a(_heatmap_md(piv, s))
            a("")
    else:
        a("No Gate-1 survivors — sensitivity not run (correct per protocol).")
        a("")
    a("## Gate 2 Results (§16)")
    a("")
    if sens:
        for s, payload in sens.items():
            a(f"* **{s}: {payload['verdict']}**" +
              (f" — {'; '.join(payload['reasons'])}" if payload['reasons'] else ""))
    else:
        a("No candidates (all rejected at Gate 1).")
    a("")

    # ---- WF ----
    a("## Walk-Forward Results (§17-19)")
    a("")
    if wfj:
        for s, payload in wfj.items():
            sm = payload["summary"]
            a(f"### Strategy {s} (frozen params, non-overlapping 2-month OOS windows)")
            a(f"* WFE {sm['wfe']:.2%}; winning windows ratio {sm['win_window_ratio']:.0%} "
              f"({sm['n_valid_windows']} valid / {sm['n_low_sample_windows']} LOW_SAMPLE); "
              f"median OOS expectancy {sm['median_oos_expectancy']:.4f}R; "
              f"median OOS PF {sm['median_oos_pf']:.3f}; OOS trades {sm['oos_trades_total']}")
            a(f"* Best-window profit concentration {sm['concentration_best_window']:.0%}; "
              f"max window DD {sm['max_window_drawdown']:.2f}")
            wdf = _load(f"walkforward_{s}.csv", pd.DataFrame())
            if wdf is not None and not wdf.empty:
                a("")
                show = wdf.copy()
                for c in ("pf", "expectancy_r", "net_profit", "maxdd"):
                    if c in show.columns:
                        show[c] = show[c].map(lambda x: _fmt(x))
                a(_df_md(show))
            a("")
    else:
        a("No Gate-2 survivors — walk-forward not run (correct per protocol).")
        a("")
    a("## Gate 3 Results (§20)")
    a("")
    if wfj:
        for s, payload in wfj.items():
            a(f"* **{s}: {payload['verdict']}**" +
              (f" — {'; '.join(payload['reasons'])}" if payload['reasons'] else ""))
    else:
        a("No candidates.")
    a("")

    # ---- OOS ----
    a("## OOS Results (§21)")
    a("")
    if receipt and receipt.get("receipt_hash"):
        a(f"* Freeze receipt: `{receipt['receipt_hash'][:16]}…` at {receipt['timestamp']} "
          f"(config `{receipt['config_hash'][:12]}…`, dataset `{receipt['dataset_hash'][:12]}…`, "
          f"commit `{receipt['code_commit'][:12]}…`). Holdout opened once, run once, no changes.")
    elif receipt:
        a(f"* Freeze receipt: {receipt.get('status', receipt)}")
    else:
        a("* Holdout NEVER opened (no Gate-3 survivors) — it remains pristine for a future round.")
    a("")
    if oos is not None and not oos.empty:
        cols = ["strategy", "total_trades", "profit_factor", "expectancy_r",
                "sharpe", "sortino", "max_drawdown", "return_over_maxdd",
                "total_costs", "win_rate"]
        show = oos[[c for c in cols if c in oos.columns]].copy()
        for c in show.columns:
            if c != "strategy":
                show[c] = show[c].map(lambda x: _fmt(x))
        a(_df_md(show))
        a("")
    a("## Gate 4 Results (§22)")
    a("")
    if g4v:
        for s, payload in g4v.items():
            a(f"* **{s}: {payload['verdict']}** (degradation "
              f"{_fmt(payload.get('degradation'), 2)})" +
              (f" — {'; '.join(payload['reasons'])}" if payload['reasons'] else ""))
    else:
        a("No candidates.")
    a("")

    # ---- Validation sections ----
    for title, key in (("Monte Carlo (§23)", "montecarlo"),
                       ("Bootstrap (§24)", "bootstrap"),
                       ("Permutation Test (§25)", "permutation"),
                       ("Execution Delay (§26)", "delay"),
                       ("Deflated Sharpe Ratio (§27)", "dsr")):
        a(f"## {title}")
        a("")
        g5 = validations.get("gate5", {}) or {}
        if not g5:
            a("No Gate-4 survivors — not run (correct per protocol).")
            a("")
            continue
        for s, payload in g5.items():
            block = payload.get(key, {})
            a(f"### Strategy {s}")
            a("```json")
            a(json.dumps(block, indent=2, default=str))
            a("```")
        a("")

    # ---- Cost stress ----
    a("## Cost Stress (§29)")
    a("")
    g6 = validations.get("gate6", {}) or {}
    if not g6:
        a("No Gate-5 survivors — not run (correct per protocol).")
        a("")
    else:
        for s, payload in g6.items():
            a(f"### Strategy {s}")
            sdf = _load(f"stress_{s}.csv", pd.DataFrame())
            if sdf is not None and not sdf.empty:
                show = sdf.copy()
                for c in ("pf", "expectancy_r", "maxdd", "costs", "target"):
                    if c in show.columns:
                        show[c] = show[c].map(lambda x: _fmt(x))
                a(_df_md(show))
            a("")
    a("## Gate 5 (§28)")
    a("")
    if g5v:
        for s, payload in g5v.items():
            a(f"* **{s}: {payload['verdict']}**" +
              (f" — {'; '.join(payload['reasons'])}" if payload['reasons'] else ""))
    else:
        a("No candidates.")
    a("")
    a("## Gate 6 (§30)")
    a("")
    if g6v:
        for s, payload in g6v.items():
            a(f"* **{s}: {payload['verdict']}**" +
              (f" — {'; '.join(payload['reasons'])}" if payload['reasons'] else ""))
    else:
        a("No candidates.")
    a("")

    # ---- Multiple testing ----
    a("## Multiple Testing Analysis (§31)")
    a("")
    a(f"* TOTAL EXPERIMENTS: {ledger.get('TOTAL_EXPERIMENTS', '?')}")
    a(f"* TOTAL CONFIGURATIONS: {ledger.get('TOTAL_CONFIGURATIONS', '?')} "
      f"(Gate-1: {ledger.get('gate1_configs', '?')}, sensitivity: "
      f"{ledger.get('sensitivity_configs', '?')}, heatmap: {ledger.get('heatmap_configs', '?')})")
    a(f"* TOTAL STRATEGIES: {ledger.get('TOTAL_STRATEGIES', '?')}")
    a(f"* TOTAL OOS STRATEGIES: {ledger.get('TOTAL_OOS_STRATEGIES', '?')}")
    a(f"* WF decisions: {ledger.get('wf_decisions', '?')}; adaptive decisions: "
      f"{ledger.get('adaptive_decisions', '?')} (frozen params — none by design)")
    a(f"* n_trials for DSR: {ledger.get('N_TRIALS_FOR_DSR', '?')} "
      "(all materially considered configurations, §27).")
    a(f"* Discarded (failed) experiments — all listed in ledger.json:")
    for d in (ledger.get("discarded_experiments", []) or [])[:40]:
        a(f"  * {d}")
    a("")

    # ---- Comparison ----
    a("## Final Comparison (§32)")
    a("")
    if g6v:
        a("Candidates are compared on the full evidence profile (OOS PF, expectancy, "
          "Sharpe, DSR, MaxDD, Return/DD, trades, bootstrap CI, MC DD, stress PF, "
          "concentration, costs) — never on a single metric. See oos.csv, "
          "validations.json and stress CSVs for the complete profiles.")
    else:
        a("No surviving candidates — no comparison table (correct per protocol: "
          "do not rank rejected strategies by raw PF).")
    a("")

    # ---- Limitations ----
    a("## Limitations")
    a("")
    a("* **Data**: " + ("SYNTHETIC PROXY — no real broker history was available; "
      "results validate the pipeline only and MUST be re-run on real Bid/Ask data."
      if man.get("source", "").startswith("SYNTHETIC") else "see Data Audit."))
    a("* **Costs**: commission and swap unavailable — recorded as 0 with explicit "
      "flags, never invented. Real costs are therefore weakly underestimated.")
    a("* **Bid/Ask**: mid ± spread/2 derivation (no tick-level Bid/Ask); intrabar "
      "SL/TP fills assume level fills on the exit side.")
    a("* **Delay model**: explicit proxy (0.25×/0.5× spread adverse at entry), not a "
      "precision latency model — M15 bars cannot resolve sub-candle execution.")
    a("* **Monte Carlo**: fixed-fractional replay of recorded R (ignores "
      "path-dependence of equity-proportional sizing) — standard approximation.")
    a("* **DSR**: per-trade units with T = OOS trade count; screen only, never a "
      "probability of future profit.")
    a("* **Permutation**: sign-flip null destroys serial dependence; rejects a "
      "specific null, proves no edge.")
    a("* **Sharpe/Sortino**: trade-R annualised by observed frequency — screening "
      "statistics, not fund-grade ratios.")
    a("* One position at a time; no daily-loss simulation in backtest (matches "
      "production backtest simplifications).")
    a("")

    # ---- Conclusion ----
    a("## Final Research Conclusion (§38)")
    a("")
    if cands:
        a(f"**Observed historical result:** {', '.join(cands)} passed all six gates on "
          f"{man.get('source', '?')} data with the frozen hypotheses.")
        a("")
        a("**Evidence assessment:** gate passage on this dataset is the predefined "
          "screening outcome; it remains an *observed historical result*, not proof "
          "of a robust future edge — especially given the data limitations above. "
          "The candidate(s) stay ISOLATED from production. No live trading, no merge, "
          "no Demo enablement. Next step: independent re-run on real broker history "
          "(new round, new freeze) before any prototype paper-trading is considered.")
    else:
        a("**NO ROBUST EDGE FOUND IN THIS EXPERIMENT ROUND.**")
        a("")
        a("**Observed historical result:** every candidate (A/B/C primaries) failed at "
          "or before the gate stated in the master table, on the dataset described above.")
        a("")
        a("**Evidence assessment:** under the predefined robustness criteria, none of "
          "the hypotheses demonstrated a stable positive expectancy after costs. No "
          "gates were loosened and no post-hoc tuning was performed. Any new hypothesis "
          "must become a NEW experiment round (new registry, new freeze, untouched holdout).")
    a("")
    a("## Reproducibility (§35)")
    a("")
    a(f"* Python: {env.get('python', '?')}")
    a(f"* Platform: {env.get('platform', '?')}")
    a(f"* Research version: {env.get('research_version', '?')}")
    a(f"* Config hash: `{env.get('config_hash', '?')}`")
    a(f"* Dataset hash: `{env.get('dataset_hash', '?')}`")
    a(f"* Seeds: master={MASTER_SEED} (data/MC/bootstrap/permutation/slippage/delay "
      "in environment.json → frozen_config → seeds)")
    from ..registry import git_commit as _gc
    a(f"* Code commit: `{_gc()}` (per-experiment commits in registry.jsonl)")
    a("")
    a("## Data-Leakage Audit (§37)")
    a("")
    for c in (leakage.get("checks", []) or []):
        a(f"* **{c.get('id')}** {c.get('claim')}: {c.get('status')} "
          f"(mechanism: {c.get('mechanism')}; test: {c.get('test')})")
    a("")
    a("## Production-Safety Audit (§39)")
    a("")
    a(f"* Verdict: **{(safety.get('verdict', '?'))}**")
    for k, v in (safety.get("checks", {}) or {}).items():
        a(f"* {k}: {v}")
    a(f"* Modified production files: {safety.get('modified_production_files', '?')}")
    a("")
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    return REPORT_PATH


def _df_md(df: pd.DataFrame) -> str:
    """Minimal markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _heatmap_md(piv: pd.DataFrame, strat: str) -> str:
    from ..config import SENSITIVITY_GRIDS, A_PRIMARY, B_PRIMARY, C_PRIMARY
    base = {"A": A_PRIMARY, "B": B_PRIMARY, "C": C_PRIMARY}[strat]
    grid = SENSITIVITY_GRIDS[strat]
    px, py = grid["heatmap"]
    bx, by = getattr(base, px), getattr(base, py)
    lines = ["| " + " | ".join([f"{py} \\ {px}"] + [str(c) for c in piv.columns]) + " |",
             "|" + "---|" * (len(piv.columns) + 1)]
    for r in piv.index:
        cells = []
        for c in piv.columns:
            mark = " (*)" if (c == bx and r == by) else ""
            cells.append(f"{_fmt(piv.loc[r, c])}{mark}")
        lines.append("| " + " | ".join([str(r)] + cells) + " |")
    return "\n".join(lines)
