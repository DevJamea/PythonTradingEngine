# Phase 2 — XAUUSD A/B/C Experimental Research (ISOLATED)

**RESEARCH ONLY. This directory is not part of the production trading engine.**

## Isolation rules

* Code in `research/` MUST NOT be imported by `gold_trader/`.
* Code in `research/` MUST NOT modify `gold_trader/` behaviour (no monkeypatching,
  no writes to production config, no order execution paths).
* The production strategy (`strategy/signals.py`), risk manager
  (`risk/risk_manager.py`), main loop (`main.py`) and MT5 execution code are
  read-only references for this experiment.
* The Baseline strategy reuses production signal/SLTP functions **by import**
  so its rules are byte-identical to production; only the cost accounting is
  unified (Bid/Ask) for fair comparison — documented in the final report.
* No candidate strategy from this phase may be enabled for Demo or Live trading.

## Layout

```
research/
├── config.py            # frozen hypotheses + gates (declared BEFORE first backtest)
├── registry.py          # machine-readable experiment registry / ledger / freeze receipts
├── costs.py             # unified Bid/Ask cost model + stress scenarios
├── indicators_ext.py    # causal research indicators (ADX, Donchian, swings, HTF mapping)
├── data/                # loader, synthetic proxy generator, audit, chronological splits
├── strategies/          # baseline wrapper (read-only) + research-only A / B / C
├── backtest/            # research backtest engine (Bid/Ask, R-tracking, management)
├── metrics/             # performance statistics (R-based + equity-based)
├── stats/               # Monte Carlo, bootstrap, permutation, DSR, execution delay
├── runners/             # gate1 → sensitivity → walk-forward → OOS → stress pipeline
├── outputs/             # generated artifacts (registry.jsonl, CSVs, hashes, report inputs)
├── XAUUSD_AB_C_EXPERIMENT_REPORT.md   # final scientific-style report
└── run_experiment.py    # master entry point: python -m research.run_experiment
```

## Data

The sandbox has no MT5 terminal and no network access to market-data vendors,
so **no real broker history is available**. The experiment therefore runs on a
clearly-labelled **synthetic XAUUSD M15 proxy** (`data/synthetic.py`) to validate
the full research pipeline end-to-end. Every artifact is stamped
`SYNTHETIC_PROXY_v1`, and the final report states that any result must be
re-run on real broker Bid/Ask history before it can be treated as evidence.

To re-run on real data, drop a CSV at `research/data/xauusd_m15.csv` with columns:

```
time,open,high,low,close,tick_volume,spread[,bid,ask]
```

(`time` UTC ISO-8601, `spread` in price units; `bid`/`ask` optional. If absent,
bid/ask are derived as mid ± spread/2 and the audit records that derivation.)

## Reproduce

```bash
python -m pytest tests/test_research_*.py -q   # research unit tests
python -m research.run_experiment              # full gated experiment
```
