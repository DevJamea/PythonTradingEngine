# XAUUSD A/B/C Experimental Research Report — Phase 2

> Data source: **SYNTHETIC_PROXY_v1** | Dataset hash: `3cae208d6b9fabe6…` | Config hash: `8dce5e86f872ff2a…`
> RESEARCH ONLY. Nothing in this report is a trading recommendation. A backtest is an observed historical result, never proof of future profitability.

> ⚠️ **CRITICAL LIMITATION — READ FIRST:** no real broker history was available in this sandbox (no MT5 terminal, no vendor access), so this round ran on a **neutral synthetic XAUUSD M15 proxy** (`SYNTHETIC_PROXY_v1`, zero-drift random walk with volatility clustering). All numerical results below validate the **research pipeline**, not a market edge. The entire experiment MUST be re-run on real broker Bid/Ask history before any result can be treated as evidence about real markets. A real-data CSV path is documented in `research/README.md`.

## Executive Summary

**Outcome: NO ROBUST EDGE FOUND IN THIS EXPERIMENT ROUND.** No strategy passed all six gates. This is a valid and valuable research result: it means none of the predefined hypotheses demonstrated a stable positive expectancy under the predefined robustness criteria on this dataset. No gates were loosened, no parameters were re-tuned, no data was cherry-picked.

Master stage table (§34):

| Stage | Baseline | A | B | C |
|---|---|---|---|---|
| Development | Reference | see Gate 1 | see Gate 1 | see Gate 1 |
| Gate 1 | — | REJECTED — GATE 1 | REJECTED — GATE 1 | REJECTED — GATE 1 |
| Sensitivity | — | — | — | — |
| Gate 2 | — | — | — | — |
| Walk-Forward | — | — | — | — |
| Gate 3 | — | — | — | — |
| OOS | — | — | — | — |
| Gate 4 | — | — | — | — |
| Monte Carlo | — | — | — | — |
| Gate 5 | — | — | — | — |
| Cost Stress | — | — | — | — |
| Gate 6 | — | — | — | — |

## Data Audit (§3)

* Symbols: ['XAUUSD']
* Date range: {'end': '2024-12-31 23:45:00+00:00', 'start': '2022-01-03 00:00:00+00:00'}
* Timeframe: M15
* Candles: 75072
* Bid/Ask availability: NO tick Bid/Ask — mid OHLC ± spread/2 derivation
* Spread provenance: synthetic (mean 0.38679334505541346, p95 0.7, max 3.0 price units)
* Duplicate timestamps: 0
* Missing grid slots: 0
* Max gap: 2 days 00:15:00
* Weekend bars: 0 (excluded (no Sat/Sun bars expected))
* Timezone: UTC
* Broker/data source: NONE — synthetic proxy (no broker data available)
* OHLC-inconsistent bars: 0; non-positive prices: 0
* Cleaning applied: [] (nothing silently repaired)
* Chronological splits (never shuffled): DEV ['2022-01-03 00:00:00+00:00', '2023-07-03 23:45:00+00:00'] (0–37536), WF ['2023-07-04 00:00:00+00:00', '2024-04-02 11:45:00+00:00'], OOS ['2024-04-02 12:00:00+00:00', '2024-12-31 23:45:00+00:00']

## Experiment Registry (§11)

* Machine-readable registry: `research/outputs/registry.jsonl` (one JSON object per experiment: ID, strategy, full parameters, data range, timeframe, cost/risk models, entry/exit/management models, session filter, seed, code version/commit, config hash, timestamp, metrics, gate).
* Complete trial ledger: `research/outputs/ledger.json` — 12 configurations, DSR n_trials = 12.
* Discarded experiments (nothing hidden): 3 recorded in the ledger.

## Baseline Results (§7)

_Development segment (reference, not a gate candidate):_

* total_trades: 1160.000
* wins: 335.000
* losses: 825.000
* win_rate: 0.289
* profit_factor: 0.513
* expectancy_r: -0.390
* average_r: -0.390
* sharpe: -9.580
* sortino: -67.694
* max_drawdown: 8979.979
* drawdown_duration_days: 544.583
* ulcer_index: 0.656
* consecutive_losses_max: 26.000
* avg_trade_duration_hours: 3.620
* annualised_return: -0.782
* return_over_maxdd: -1.000
* total_costs: 4325.140
* t-stat (auxiliary only): -11.724

## Strategy A — HTF Trend + Structural Pullback (§8)

Frozen hypothesis: H4 EMA50/200 bias → H1 HH+HL/LH+LL (N=3, confirmed swings) → M15 pullback to EMA50 (≤1.0 ATR, structure valid) → M15 close beyond internal swing (N=2). PRIMARY: entry at next-bar open, stop at pullback swing ±0.3 ATR, RR 2.0, full management (BE@1R, 50% partial@1R, 1×ATR trail). Diagnostics: RR 1.5/2.5, entry/exit-only, retest entry.
*Observed Gate-1 diagnostic (not a gate input):* all 24 A-primary trades exited via stop (initial/BE/trail) — the fixed TP never bound — so the predefined RR 1.5/2.0/2.5 variants produced identical fills. Under full management the fixed TP level is dead weight on this dataset.

## Strategy B — Volatility Compression Breakout (§9)

Frozen hypothesis: ATR-ratio <0.70 for ≥6 bars → Donchian(20) breakout with body ≥0.5 ATR → next-bar confirmation (no close back inside) → entry at open of bar i+2. Stop at opposite range side ±0.2 ATR. PRIMARY: RR 2.0 fixed, all hours. Diagnostics: RR 1.5/2.5, 1.5×ATR trailing exit, London/NY session.
*Observed Gate-1 diagnostic (not a gate input):* compression regimes in the synthetic proxy form in quiet Asian hours, so 89 of 93 B signals entered 00:00–06:59 UTC and the predefined London/NY (07:00–21:00) diagnostic kept only 4 signals — a faithful consequence of the proxy's session seasonality, not a filter bug (verified by entry-hour distribution). Real-data behaviour will differ.

## Strategy C — Regime Adaptive (§10)

Frozen hypothesis: TREND (ADX>25 & |slope|>0.05)→A rules; COMPRESSION (ratio<0.70)→B rules; RANGE (ADX<20 & 0.8≤ratio≤1.3)→NO_TRADE (conservative predefined choice — no mean-reversion invented); else NO_TRADE.
Effective degrees of freedom (declared, tracked): 47 (12 C-level incl. routing + A params + B params).

## Gate 1 Results (§12)

Screening: PF ≥ 1.2, trades ≥ 50, expectancy > 0, MaxDD ≤ 1.5× baseline MaxDD. t-test auxiliary only.

| strategy | variant | total_trades | profit_factor | expectancy_r | sharpe | sortino | max_drawdown | return_over_maxdd | total_costs | win_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | reference | 1160.000 | 0.513 | -0.390 | -9.580 | -67.694 | 8979.979 | -1.000 | 4325.140 | 0.289 |
| A | primary | 24.000 | 0.307 | -0.533 | -2.367 | -8.607 | 622.603 | -1.000 | 137.155 | 0.292 |
| B | primary | 67.000 | 0.646 | -0.281 | -1.422 | -38.563 | 1137.959 | -0.802 | 425.325 | 0.284 |
| C | primary | 62.000 | 0.563 | -0.358 | -1.814 | -39.569 | 1254.839 | -0.846 | 381.575 | 0.258 |
| A:RR15 | diagnostic:RR1.5 | 24.000 | 0.307 | -0.533 | -2.367 | -8.607 | 622.603 | -1.000 | 137.155 | 0.292 |
| A:RR25 | diagnostic:RR2.5 | 24.000 | 0.307 | -0.533 | -2.367 | -8.607 | 622.603 | -1.000 | 137.155 | 0.292 |
| A:no-mgmt | diagnostic:entry-exit-only | 21.000 | 0.519 | -0.396 | -1.163 | -5.966 | 470.305 | -0.876 | 132.310 | 0.238 |
| A:retest | diagnostic:retest-entry | 23.000 | 0.270 | -0.586 | -2.604 | -9.675 | 654.165 | -1.000 | 129.310 | 0.261 |
| B:RR15 | diagnostic:RR1.5 | 67.000 | 0.667 | -0.236 | -1.344 | -29.346 | 1000.548 | -0.772 | 430.605 | 0.358 |
| B:RR25 | diagnostic:RR2.5 | 65.000 | 0.708 | -0.236 | -1.053 | -33.449 | 996.751 | -0.758 | 423.025 | 0.262 |
| B:trail | diagnostic:trailing-exit | 67.000 | 0.533 | -0.268 | -2.013 | -28.754 | 892.801 | -0.970 | 430.175 | 0.478 |
| B:session | diagnostic:london-ny-session | 4.000 | 1.744 | 0.387 | 0.391 | 76.942 | 103.105 | 0.744 | 21.550 | 0.500 |

* **A: REJECTED — GATE 1** — PF 0.307 < 1.2; trades 24 < 50; expectancy -0.5330R <= 0
* **B: REJECTED — GATE 1** — PF 0.646 < 1.2; expectancy -0.2812R <= 0
* **C: REJECTED — GATE 1** — PF 0.563 < 1.2; expectancy -0.3578R <= 0
*Note:* the B London-NY diagnostic shows PF 1.74 on 4 trades — a LOW_SAMPLE artifact, not evidence. Diagnostics cannot pass gates by design.

## Sensitivity Analysis (§13-15)

No Gate-1 survivors — sensitivity not run (correct per protocol).

## Gate 2 Results (§16)

No candidates (all rejected at Gate 1).

## Walk-Forward Results (§17-19)

No Gate-2 survivors — walk-forward not run (correct per protocol).

## Gate 3 Results (§20)

No candidates.

## OOS Results (§21)

* Freeze receipt: NOT OPENED — no Gate-3 survivors

## Gate 4 Results (§22)

No candidates.

## Monte Carlo (§23)

No Gate-4 survivors — not run (correct per protocol).

## Bootstrap (§24)

No Gate-4 survivors — not run (correct per protocol).

## Permutation Test (§25)

No Gate-4 survivors — not run (correct per protocol).

## Execution Delay (§26)

No Gate-4 survivors — not run (correct per protocol).

## Deflated Sharpe Ratio (§27)

No Gate-4 survivors — not run (correct per protocol).

## Cost Stress (§29)

No Gate-5 survivors — not run (correct per protocol).

## Gate 5 (§28)

No candidates.

## Gate 6 (§30)

No candidates.

## Multiple Testing Analysis (§31)

* TOTAL EXPERIMENTS: 12
* TOTAL CONFIGURATIONS: 12 (Gate-1: 12, sensitivity: 0, heatmap: 0)
* TOTAL STRATEGIES: 4
* TOTAL OOS STRATEGIES: 0
* WF decisions: 0; adaptive decisions: 0 (frozen params — none by design)
* n_trials for DSR: 12 (all materially considered configurations, §27).
* Discarded (failed) experiments — all listed in ledger.json:
  * A:primary REJECTED — GATE 1 (PF 0.307 < 1.2; trades 24 < 50; expectancy -0.5330R <= 0)
  * B:primary REJECTED — GATE 1 (PF 0.646 < 1.2; expectancy -0.2812R <= 0)
  * C:primary REJECTED — GATE 1 (PF 0.563 < 1.2; expectancy -0.3578R <= 0)

## Final Comparison (§32)

No surviving candidates — no comparison table (correct per protocol: do not rank rejected strategies by raw PF).

## Limitations

* **Data**: SYNTHETIC PROXY — no real broker history was available; results validate the pipeline only and MUST be re-run on real Bid/Ask data.
* **Costs**: commission and swap unavailable — recorded as 0 with explicit flags, never invented. Real costs are therefore weakly underestimated.
* **Bid/Ask**: mid ± spread/2 derivation (no tick-level Bid/Ask); intrabar SL/TP fills assume level fills on the exit side.
* **Delay model**: explicit proxy (0.25×/0.5× spread adverse at entry), not a precision latency model — M15 bars cannot resolve sub-candle execution.
* **Monte Carlo**: fixed-fractional replay of recorded R (ignores path-dependence of equity-proportional sizing) — standard approximation.
* **DSR**: per-trade units with T = OOS trade count; screen only, never a probability of future profit.
* **Permutation**: sign-flip null destroys serial dependence; rejects a specific null, proves no edge.
* **Sharpe/Sortino**: trade-R annualised by observed frequency — screening statistics, not fund-grade ratios.
* One position at a time; no daily-loss simulation in backtest (matches production backtest simplifications).

## Final Research Conclusion (§38)

**NO ROBUST EDGE FOUND IN THIS EXPERIMENT ROUND.**

**Observed historical result:** every candidate (A/B/C primaries) failed at or before the gate stated in the master table, on the dataset described above.

**Evidence assessment:** under the predefined robustness criteria, none of the hypotheses demonstrated a stable positive expectancy after costs. No gates were loosened and no post-hoc tuning was performed. Any new hypothesis must become a NEW experiment round (new registry, new freeze, untouched holdout).

## Reproducibility (§35)

* Python: 3.11.2 (main, Apr  8 2026, 01:58:00) [GCC 12.2.0]
* Platform: Linux-6.1.158+-x86_64-with-glibc2.36
* Research version: 2.0.0-phase2
* Config hash: `8dce5e86f872ff2ad0993ebfa4b3a9de9d8cfe8adb2c28642106e793a49d146d`
* Dataset hash: `3cae208d6b9fabe63015b98202ed4025dad61c64cce85cbd372f881891ef4045`
* Seeds: master=20260924 (data/MC/bootstrap/permutation/slippage/delay in environment.json → frozen_config → seeds)
* Code commit: `63fc2fb3a7e80af03cc1a33ad5a388b26a001258` (per-experiment commits in registry.jsonl)

## Data-Leakage Audit (§37)

* **L1** no future candle enters a signal: PASS (by construction + unit test) (mechanism: causal indicators (EMA/ATR/ADX/Donchian-shifted/SMA); signals evaluated per closed bar i using rows 0..i; entry at open of bar >= i+1; test: tests/test_research_leakage.py::TestCausalIndicators)
* **L2** no future spread enters an earlier trade: PASS (by construction + unit test) (mechanism: per-bar spread array indexed by the trade's own entry/exit bars only; stress multipliers applied bar-wise; test: tests/test_research_costs.py)
* **L3** no future ATR enters an earlier signal: PASS (by construction + unit test) (mechanism: Wilder ATR is causal; trailing uses entry-bar ATR fixed per trade; test: tests/test_research_leakage.py)
* **L4** no future EMA values are used: PASS (by construction + unit test) (mechanism: pandas ewm causal; HTF EMA mapped via merge_asof on fully-closed HTF bar ends only; test: tests/test_research_leakage.py::TestHTFMapping)
* **L5** no OOS data enters parameter selection: PASS (by construction + receipt) (mechanism: parameters frozen in research/config.py before execution; freeze receipt hashes config before OOS slice is built; test: holdout lock (splits.assert_holdout_unlocked) + freeze_receipt.json)
* **L6** no OOS data enters sensitivity: PASS (verified from registry) (mechanism: sensitivity runner receives the Development slice only; test: registry data_range audit (all sensitivity rows segment=development))
* **L7** no OOS data enters strategy construction: PASS (single code version; see environment.json) (mechanism: strategies implemented from the brief before any backtest; no post-hoc rules; test: registry: no strategy code/version change across stages)
* **L8** no future information enters trade management: PASS (by construction + unit test) (mechanism: BE/partial/trail triggers evaluated bar-by-bar on the exit-side Bid/Ask of the current bar only; test: tests/test_research_engine.py)

## Production-Safety Audit (§39)

* Verdict: **PASS — production untouched**
* no_new_strategy_enabled: True
* no_production_config_changed: True
* production_demo_safety_unchanged: True
* production_main_loop_unchanged: True
* production_mt5_execution_unchanged: True
* production_risk_manager_unchanged: True
* production_strategy_unchanged: True
* Modified production files: []

