"""Frozen research configuration for Phase 2.

EVERYTHING in this file was declared BEFORE the first backtest was executed.
Values here are the *initial hypotheses* from the experiment brief plus the
small number of disambiguating choices the brief explicitly requires to be
predefined (entry variant, primary RR, session, range-regime behaviour, WFE
definition, null model, delay proxy, slippage distribution, DSR trial rule).

Do not edit after execution starts. The freeze receipt (registry.py) hashes
the canonical JSON of this config; any post-hoc change invalidates the OOS.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# master seeds (every stochastic process is seeded and recorded)
# ---------------------------------------------------------------------------
MASTER_SEED = 20260924
SEED_DATA = 42            # synthetic proxy generator
SEED_MONTECARLO = 123456  # trade-shuffle Monte Carlo
SEED_BOOTSTRAP = 789      # bootstrap CIs
SEED_PERMUTATION = 101112  # sign-flip permutation null test
SEED_SLIPPAGE = 131415    # cost-stress slippage draws
SEED_DELAY = 161718       # execution-delay proxy draws (if randomised)

RESEARCH_VERSION = "2.0.0-phase2"

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
REAL_DATA_CSV = "research/data/xauusd_m15.csv"  # optional real-data override
SYNTHETIC_TAG = "SYNTHETIC_PROXY_v1"
SYMBOL = "XAUUSD"
TIMEFRAME = "M15"
SYNTH_START = "2022-01-03 00:00"  # first Monday of 2022 (UTC)
SYNTH_END = "2024-12-31 23:45"    # last M15 bar of 2024 (UTC)
SYNTH_START_PRICE = 2000.0

# chronological splits (fractions of the full bar count, never shuffled)
DEV_FRACTION = 0.50
WF_FRACTION = 0.25
OOS_FRACTION = 0.25

# warmup bars prepended (for indicator readiness) but never traded
WARMUP_BARS = 4000  # > H4 EMA200 (200*16=3200) + H1 swing confirmation margin

# ---------------------------------------------------------------------------
# capital / risk (identical to the production baseline experiment)
# ---------------------------------------------------------------------------
INITIAL_BALANCE = 10_000.0
RISK_PER_TRADE = 0.005  # 0.5 % of equity per trade
MAX_OPEN_POSITIONS = 1  # same as production default

# ---------------------------------------------------------------------------
# cost model (identical for every strategy)
# ---------------------------------------------------------------------------
COMMISSION_PER_TRADE = None  # unavailable — explicitly NOT invented
SWAP_PER_TRADE = None        # unavailable / N/A intraday — explicitly NOT invented
BASE_SLIPPAGE_POINTS = 0     # baseline runs use pure Bid/Ask, no added slippage


# ---------------------------------------------------------------------------
# Strategy A — HTF trend + structural pullback (PRIMARY + diagnostics)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyAParams:
    # H4 bias
    h4_ema_fast: int = 50
    h4_ema_slow: int = 200
    # H1 structure
    h1_swing_n: int = 3
    # M15 pullback
    m15_ema_pullback: int = 50
    pullback_max_atr: float = 1.0   # distance(close, EMA50) <= 1.0 * ATR14
    m15_atr_period: int = 14
    pullback_lookback: int = 96     # max bars to look back for pullback start
    # trigger (internal M15 swing)
    trigger_swing_n: int = 2
    trigger_lookback: int = 96
    # entry variant: "close" (PRIMARY) or "retest" (diagnostic only)
    entry_variant: str = "close"
    retest_window: int = 3          # diagnostic retest: fill within N bars else cancel
    # stop / target
    stop_buffer_atr: float = 0.3
    rr_target: float = 2.0          # PRIMARY central hypothesis; 1.5/2.5 predefined neighbours
    # management (PRIMARY = full management ON)
    management: str = "full"        # "full" | "none" (diagnostic entry/exit-only)
    be_trigger_r: float = 1.0
    be_buffer: float = 0.10         # price units, same as production default
    partial_r: float = 1.0
    partial_fraction: float = 0.50
    trail_atr_mult: float = 1.0
    trail_activation_r: float = 1.0


# Predefined Strategy-A experimental candidates (all registered before execution).
A_PRIMARY = StrategyAParams()  # gates use ONLY this one
A_RR15 = StrategyAParams(rr_target=1.5)   # Gate-1 diagnostic (predefined neighbour)
A_RR25 = StrategyAParams(rr_target=2.5)   # Gate-1 diagnostic (predefined neighbour)
A_DIAG_NO_MGMT = StrategyAParams(management="none")  # entry/exit-only diagnostic
A_DIAG_RETEST = StrategyAParams(entry_variant="retest")  # retest-entry diagnostic


# ---------------------------------------------------------------------------
# Strategy B — volatility compression breakout (PRIMARY + diagnostics)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyBParams:
    atr_period: int = 14
    atr_sma_period: int = 50
    compression_ratio: float = 0.70
    compression_min_bars: int = 6
    donchian_period: int = 20
    body_min_atr: float = 0.5
    stop_buffer_atr: float = 0.2
    rr_target: float = 2.0          # PRIMARY central hypothesis; 1.5/2.5 predefined neighbours
    exit_mode: str = "fixed"        # "fixed" (PRIMARY) | "trail" (diagnostic)
    trail_atr_mult: float = 1.5     # diagnostic trailing config (predefined)
    trail_activation_r: float = 1.0
    be_trigger_r: float = 1.0
    be_buffer: float = 0.10
    # session: "all" (PRIMARY, matches production 00:00-23:59) vs "london_ny" (diagnostic)
    session: str = "all"
    session_london_ny: Tuple[str, str] = ("07:00", "21:00")  # UTC, diagnostic only


B_PRIMARY = StrategyBParams()  # gates use ONLY this one
B_RR15 = StrategyBParams(rr_target=1.5)  # Gate-1 diagnostic
B_RR25 = StrategyBParams(rr_target=2.5)  # Gate-1 diagnostic
B_DIAG_TRAIL = StrategyBParams(exit_mode="trail")  # trailing diagnostic
B_DIAG_SESSION = StrategyBParams(session="london_ny")  # session diagnostic


# ---------------------------------------------------------------------------
# Strategy C — regime adaptive (PRIMARY)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyCParams:
    adx_period: int = 14
    adx_trend: float = 25.0
    adx_range_max: float = 20.0
    atr_period: int = 14
    atr_sma_period: int = 50
    compression_ratio: float = 0.70
    range_ratio_lo: float = 0.8
    range_ratio_hi: float = 1.3
    ema_slope_period: int = 50
    ema_slope_lookback: int = 10
    slope_threshold: float = 0.05  # |slope_norm| > 0.05, predefined hypothesis
    # routing (predefined): trend -> A_PRIMARY rules, compression -> B_PRIMARY
    # rules, range -> NO_TRADE (conservative; no mean-reversion invented).
    range_action: str = "no_trade"
    trend_params: StrategyAParams = field(default_factory=lambda: A_PRIMARY)
    compression_params: StrategyBParams = field(default_factory=lambda: B_PRIMARY)


C_PRIMARY = StrategyCParams()  # gates use ONLY this one

# Effective degrees of freedom tracked for Strategy C (free parameters incl.
# routed sub-strategy params). Counted programmatically in the report; the
# declared list is: adx_period, adx_trend, adx_range_max, atr_period,
# atr_sma_period, compression_ratio, range_ratio_lo, range_ratio_hi,
# ema_slope_period, ema_slope_lookback, slope_threshold, range_action
# + all StrategyAParams fields + all StrategyBParams fields routed to.
C_DEGREES_OF_FREEDOM_DECLARED = 12 + len(asdict(A_PRIMARY)) + len(asdict(B_PRIMARY))


# ---------------------------------------------------------------------------
# sensitivity grids (declared now; sensitivity runs on Development data only)
# ---------------------------------------------------------------------------
SENSITIVITY_GRIDS = {
    # strategy A: one-at-a-time neighbours (baseline marked with *)
    "A": {
        "pullback_max_atr": [0.6, 0.8, 1.0, 1.2, 1.4],
        "stop_buffer_atr": [0.1, 0.2, 0.3, 0.4, 0.5],
        "rr_target": [1.5, 2.0, 2.5],
        "h1_swing_n": [2, 3, 4],
        "trigger_swing_n": [1, 2, 3],
        "heatmap": ("pullback_max_atr", "stop_buffer_atr"),  # 5x5 = 25 combos
    },
    "B": {
        "compression_ratio": [0.50, 0.60, 0.70, 0.80, 0.90],
        "compression_min_bars": [4, 5, 6, 8, 10],
        "donchian_period": [10, 15, 20, 30, 40],
        "body_min_atr": [0.30, 0.40, 0.50, 0.60, 0.75],
        "stop_buffer_atr": [0.10, 0.15, 0.20, 0.30, 0.40],
        "rr_target": [1.5, 2.0, 2.5],
        "heatmap": ("compression_ratio", "donchian_period"),  # 5x5 = 25 combos
    },
    "C": {
        "adx_trend": [20.0, 22.0, 25.0, 28.0, 30.0],
        "slope_threshold": [0.02, 0.035, 0.05, 0.075, 0.10],
        "compression_ratio": [0.60, 0.65, 0.70, 0.75, 0.80],
        "ema_slope_lookback": [5, 8, 10, 15, 20],
        "heatmap": ("adx_trend", "slope_threshold"),  # 5x5 = 25 combos
    },
}

# Gate-2 robustness rule (predefined, favours plateaus over spikes):
# pass requires ALL of:
GATE2_MIN_POSITIVE_EXPECTANCY_FRAC = 0.40  # >=40% of OAT neighbours keep exp>0
GATE2_MIN_PF_GT1_FRAC = 0.40               # >=40% of OAT neighbours keep PF>1
GATE2_MAX_ISOLATION_RATIO = 2.0  # best PF <= 2x median neighbour PF (no lone spike)
GATE2_MIN_CONNECTED_REGION = 3   # heatmap: largest connected PF>1 region >= 3 cells


# ---------------------------------------------------------------------------
# gates (practical screening criteria from the brief)
# ---------------------------------------------------------------------------
GATE1_MIN_PF = 1.20
GATE1_MIN_TRADES = 50
GATE1_MAXDD_MULT_BASELINE = 1.5  # MaxDD <= 1.5 x Baseline MaxDD (same segment)

# walk-forward: frozen params on non-overlapping 2-month OOS windows in WF zone
WF_OOS_MONTHS = 2
WF_LOW_SAMPLE_TRADES = 30
WF_MIN_VALID_WINDOWS = 2  # need >=2 valid (>=30 trades) windows to pass Gate 3
GATE3_MIN_WFE = 0.50
GATE3_MIN_WIN_WINDOW_RATIO = 0.60
GATE3_MAX_PROFIT_CONCENTRATION = 0.50
# WFE definition (predefined): median OOS-window expectancy(R) / Development
# expectancy(R). Windows with <30 trades are LOW_SAMPLE: reported separately,
# excluded from WFE/win-ratio numerators but COUNTED in the ledger.

# Gate 4 (OOS holdout)
GATE4_MIN_PF = 1.0
GATE4_MAX_EXPECTANCY_DEGRADATION = 0.50  # vs WF median OOS expectancy
# MaxDD tolerance for Gate 4: same predefined rule as Gate 1 (1.5x OOS baseline)

# Gate 5 (statistical validation)
DSR_THRESHOLD = 0.95  # predefined screening criterion, NOT a profit probability
MC_SIMS = 5000
BOOTSTRAP_SIMS = 5000
PERMUTATION_SIMS = 5000
# Monte-Carlo fragility rule (predefined): fail if P95 MaxDD exceeds 3x the
# observed OOS MaxDD, or if P(terminal equity < 50% of start) > 5%.
MC_P95_DD_MULT_FAIL = 3.0
MC_RUIN_PROB_FAIL = 0.05
MC_RUIN_EQUITY_FRAC = 0.50
# Gate-5 operationalisations (predefined): bootstrap "supports positive
# expectancy" <=> 95% CI lower bound for expectancy > 0; delay "does not
# destroy the result" <=> PF at 2s proxy >= 1.0 AND expectancy(2s) > 0.
GATE5_REQUIRE_BOOTSTRAP_CI_ABOVE_ZERO = True
GATE5_DELAY_MIN_PF = 1.0

# execution-delay proxy (M15 close data cannot model sub-candle fills; this is
# documented as a PROXY, not a precision latency model):
#   0s -> no change; 1s -> +0.25x spread adverse; 2s -> +0.5x spread adverse.
DELAY_PROXY_SPREAD_FRAC = {0: 0.0, 1: 0.25, 2: 0.50}

# cost stress (predefined)
STRESS_SPREAD_MULTS = [1.5, 2.0]
STRESS_SLIPPAGE_POINTS = (1, 5)  # uniform adverse 1..5 points per side (point=0.01)
STRESS_TARGETS = {
    "spread_1.5": 1.0,
    "spread_2.0": 0.9,
    "slippage": 1.0,
    "combined": 0.9,
}


# ---------------------------------------------------------------------------
# canonical hash (freeze receipt input)
# ---------------------------------------------------------------------------
def frozen_config_dict() -> dict:
    """Canonical JSON-able dict of everything frozen before execution."""
    return {
        "research_version": RESEARCH_VERSION,
        "seeds": {
            "master": MASTER_SEED,
            "data": SEED_DATA,
            "montecarlo": SEED_MONTECARLO,
            "bootstrap": SEED_BOOTSTRAP,
            "permutation": SEED_PERMUTATION,
            "slippage": SEED_SLIPPAGE,
            "delay": SEED_DELAY,
        },
        "data": {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "synth_start": SYNTH_START,
            "synth_end": SYNTH_END,
            "synth_start_price": SYNTH_START_PRICE,
            "dev": DEV_FRACTION,
            "wf": WF_FRACTION,
            "oos": OOS_FRACTION,
            "warmup_bars": WARMUP_BARS,
        },
        "capital": {"initial_balance": INITIAL_BALANCE, "risk_per_trade": RISK_PER_TRADE},
        "A_primary": asdict(A_PRIMARY),
        "B_primary": asdict(B_PRIMARY),
        "C_primary": {
            **{k: v for k, v in asdict(C_PRIMARY).items()
               if k not in ("trend_params", "compression_params")},
            "trend_params": asdict(A_PRIMARY),
            "compression_params": asdict(B_PRIMARY),
        },
        "diagnostics": {
            "A_RR15": asdict(A_RR15),
            "A_RR25": asdict(A_RR25),
            "A_DIAG_NO_MGMT": asdict(A_DIAG_NO_MGMT),
            "A_DIAG_RETEST": asdict(A_DIAG_RETEST),
            "B_RR15": asdict(B_RR15),
            "B_RR25": asdict(B_RR25),
            "B_DIAG_TRAIL": asdict(B_DIAG_TRAIL),
            "B_DIAG_SESSION": asdict(B_DIAG_SESSION),
        },
        "sensitivity_grids": SENSITIVITY_GRIDS,
        "gates": {
            "gate1": {"min_pf": GATE1_MIN_PF, "min_trades": GATE1_MIN_TRADES,
                      "maxdd_mult": GATE1_MAXDD_MULT_BASELINE},
            "gate2": {"min_pos_exp_frac": GATE2_MIN_POSITIVE_EXPECTANCY_FRAC,
                      "min_pf_gt1_frac": GATE2_MIN_PF_GT1_FRAC,
                      "max_isolation": GATE2_MAX_ISOLATION_RATIO,
                      "min_connected": GATE2_MIN_CONNECTED_REGION},
            "gate3": {"min_wfe": GATE3_MIN_WFE,
                      "min_win_ratio": GATE3_MIN_WIN_WINDOW_RATIO,
                      "max_concentration": GATE3_MAX_PROFIT_CONCENTRATION,
                      "wf_oos_months": WF_OOS_MONTHS,
                      "low_sample": WF_LOW_SAMPLE_TRADES,
                      "min_valid_windows": WF_MIN_VALID_WINDOWS},
            "gate4": {"min_pf": GATE4_MIN_PF,
                      "max_degradation": GATE4_MAX_EXPECTANCY_DEGRADATION},
            "gate5": {"dsr": DSR_THRESHOLD, "mc_sims": MC_SIMS,
                      "bootstrap_sims": BOOTSTRAP_SIMS,
                      "perm_sims": PERMUTATION_SIMS},
            "stress": STRESS_TARGETS,
        },
    }


def config_hash() -> str:
    blob = json.dumps(frozen_config_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
