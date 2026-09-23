"""Short-horizon mean-reversion scalping engine (opt-in, separate strategy).

This module is intentionally NOT a variant of :mod:`gold_trader.strategy.signals`.
The trend-following baseline (EMA alignment + RSI + candle pattern) failed its
isolated-period validation on real gold data, so re-using that decision logic
on a faster timeframe would repeat the same mistake. This engine instead trades
the *exhaustion of a short excursion*: price pushes outside a tight Bollinger
band with an oversold/overbought fast RSI, and we act only when the very next
closed candle reverses back inside the band.

Design rules (all enforced here, all unit-tested):

* **Causal only** -- the value at index ``i`` uses rows ``0..i``; nothing ever
  reads a future row.
* **Confirmed reversal, never a touch** -- a candle that merely pokes below the
  lower band (or merely shows RSI < 20) produces NO_TRADE. The signal candle
  must close back inside the band, against the direction of the excursion.
* **Cost gate is mandatory** -- the computed take-profit distance must be at
  least ``min_profit_to_spread_ratio`` times the (spread + safety margin).
  When it is not, the answer is NO_TRADE *regardless of the signal*. This is
  the single most important rule for scalping: a target that does not clear
  the spread several times over is a guaranteed slow loss.
* **Volatility window** -- too little volatility and the price never covers the
  spread; too much and stop distance/slippage dominate. Both ends are rejected
  with a causal percentile rank of ATR.
* **Tiny SL/TP, no management** -- positions are closed entirely at SL or TP.
  Break-even / partial close / trailing are deliberately not applied here
  (see :mod:`gold_trader.trade_management.scalping_policy`).

Nothing in this module claims profitability; it is a mechanical, testable rule
set that the backtest layer is allowed to prove wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..config import Config
from ..models import Signal, TradeSignal
from .indicators import atr as atr_indicator
from .indicators import rsi as rsi_indicator

REQUIRED_COLUMNS = ("time", "open", "high", "low", "close")

#: Marker written into the broker position comment so the management layer can
#: recognise a scalp and leave it alone (see trade_management.scalping_policy).
SCALP_COMMENT_MARKER = "SCALP"


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalpingIndicators:
    """Precomputed, causal indicator series for one OHLC frame."""

    close: pd.Series
    basis: pd.Series
    upper: pd.Series
    lower: pd.Series
    rsi: pd.Series
    atr: pd.Series
    #: Causal percentile rank (0..1) of ``atr`` inside its recent window.
    atr_rank: pd.Series


def bollinger_bands(
    close: pd.Series, period: int, n_std: float
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Simple moving-average bands: ``basis +/- n_std * rolling stdev``.

    ``min_periods=period`` guarantees NaN before the window is full, so no
    partial (and therefore no future) window is ever used.
    """
    if period < 2:
        raise ValueError("bollinger period must be >= 2")
    if n_std <= 0:
        raise ValueError("bollinger n_std must be > 0")
    basis = close.rolling(window=int(period), min_periods=int(period)).mean()
    std = close.rolling(window=int(period), min_periods=int(period)).std(ddof=0)
    upper = basis + n_std * std
    lower = basis - n_std * std
    return basis, upper, lower


def percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
    """Fraction of the trailing ``lookback`` values (inclusive) <= value at i.

    Purely causal: the window ends at ``i``. Values before the window is full
    are NaN, which the strategy treats as "not ready" (no trade).
    """
    if lookback < 2:
        raise ValueError("percentile lookback must be >= 2")
    window = int(lookback)
    out = series.rolling(window=window, min_periods=window).apply(
        lambda arr: float((arr <= arr[-1]).mean()), raw=True
    )
    return out


def compute_scalping_indicators(df: pd.DataFrame, cfg: Config) -> ScalpingIndicators:
    """Compute every series the scalping engine reads (all causal)."""
    _validate_columns(df)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    basis, upper, lower = bollinger_bands(close, cfg.scalp_bb_period, cfg.scalp_bb_std)
    atr_series = atr_indicator(high, low, close, cfg.scalp_atr_period)
    return ScalpingIndicators(
        close=close,
        basis=basis,
        upper=upper,
        lower=lower,
        rsi=rsi_indicator(close, cfg.scalp_rsi_period),
        atr=atr_series,
        atr_rank=percentile_rank(atr_series, cfg.scalp_vol_lookback),
    )


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"scalping candle frame is missing columns: {missing}")


def _clean(value: Any) -> Optional[float]:
    """Scalar -> float with NaN mapped to None (None stays None)."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


# ---------------------------------------------------------------------------
# configuration sanity
# ---------------------------------------------------------------------------

def validate_scalping_config(cfg: Config) -> None:
    """Raise :class:`ValueError` when scalping settings are self-contradictory.

    Called before any live or simulated scalp decision. A ratio below 1.0
    would mean "target smaller than the cost of doing the trade", which is a
    mathematically guaranteed loss, so it is refused instead of simulated.
    """
    if cfg.min_profit_to_spread_ratio < 1.0:
        raise ValueError(
            "min_profit_to_spread_ratio must be >= 1.0 (a target that does not "
            f"cover the trade cost is a guaranteed loss; got "
            f"{cfg.min_profit_to_spread_ratio})"
        )
    if cfg.scalp_spread_safety_margin < 0:
        raise ValueError("scalp_spread_safety_margin must be >= 0")
    if cfg.scalp_expected_spread <= 0:
        raise ValueError("scalp_expected_spread must be > 0")
    if cfg.scalp_sl_atr_multiple <= 0 or cfg.scalp_tp_atr_multiple <= 0:
        raise ValueError("scalp SL/TP ATR multiples must be > 0")
    if not (0 <= cfg.scalp_vol_min_percentile < cfg.scalp_vol_max_percentile <= 1):
        raise ValueError(
            "scalp volatility percentiles must satisfy "
            "0 <= min < max <= 1 (got "
            f"{cfg.scalp_vol_min_percentile}, {cfg.scalp_vol_max_percentile})"
        )
    if cfg.scalp_max_trades_per_day < 1:
        raise ValueError("scalp_max_trades_per_day must be >= 1")
    if cfg.scalping_risk_per_trade <= 0:
        raise ValueError("scalping_risk_per_trade must be > 0")
    if cfg.scalping_timeframe.upper() not in ("M1", "M5", "M15"):
        raise ValueError(
            "scalping_timeframe must be M1, M5 or M15 (got "
            f"{cfg.scalping_timeframe!r}); M1 on candle-only history is not a "
            "reliable testing ground"
        )


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def effective_spread(cfg: Config, live_spread: Optional[float] = None) -> float:
    """Round-trip spread to plan against: the live quote when known.

    ``live_spread`` is ``ask - bid`` in price units. When it is unavailable the
    conservative configured estimate (``scalp_expected_spread``) is used -- the
    code never assumes a free execution.
    """
    value = _clean(live_spread)
    if value is None or value <= 0:
        return float(cfg.scalp_expected_spread)
    return value


def required_tp_distance(spread: float, cfg: Config) -> float:
    """Minimum acceptable TP distance for ``spread`` (price units).

    ``(spread + safety margin) x min_profit_to_spread_ratio``.
    """
    return (float(spread) + float(cfg.scalp_spread_safety_margin)) * float(
        cfg.min_profit_to_spread_ratio
    )


def cost_gate_result(
    tp_distance: float, spread: float, cfg: Config
) -> Tuple[bool, str]:
    """Return (passes, reason) for the mandatory profit-vs-cost gate."""
    required = required_tp_distance(spread, cfg)
    if not math.isfinite(tp_distance) or tp_distance <= 0:
        return False, "invalid take-profit distance"
    if tp_distance < required:
        return False, (
            f"TP distance {tp_distance:.5f} < required {required:.5f} = "
            f"(spread {spread:.5f} + margin {cfg.scalp_spread_safety_margin:.5f}) "
            f"x ratio {cfg.min_profit_to_spread_ratio:g}"
        )
    return True, f"TP distance {tp_distance:.5f} >= required {required:.5f}"


def volatility_gate_result(
    atr_value: Optional[float], atr_rank: Optional[float], cfg: Config
) -> Tuple[bool, str]:
    """Reject dead and explosive volatility; both destroy scalping.

    ``atr_rank`` is the causal percentile of the current ATR against its own
    recent distribution: below ``scalp_vol_min_percentile`` the bar range is
    too small to ever pay for the spread; above ``scalp_vol_max_percentile``
    the stop has to be so wide (or the slippage so large) that a scalp is no
    longer a scalp.
    """
    value = _clean(atr_value)
    rank = _clean(atr_rank)
    if value is None or rank is None:
        return False, "volatility history not ready (NaN ATR or rank)"
    if value <= 0:
        return False, f"ATR {value} is not positive"
    if rank < cfg.scalp_vol_min_percentile:
        return False, (
            f"volatility too low for scalping (ATR {value:.5f} at percentile "
            f"{rank:.2%} < {cfg.scalp_vol_min_percentile:.2%})"
        )
    if rank > cfg.scalp_vol_max_percentile:
        return False, (
            f"volatility too high for scalping (ATR {value:.5f} at percentile "
            f"{rank:.2%} > {cfg.scalp_vol_max_percentile:.2%})"
        )
    return True, f"volatility acceptable (ATR {value:.5f}, percentile {rank:.2%})"


def spread_gate_result(spread: float, cfg: Config) -> Tuple[bool, str]:
    """Reject entries while the quoted spread is wider than allowed."""
    if spread > cfg.scalp_max_spread:
        return False, (
            f"spread {spread:.5f} exceeds scalping maximum "
            f"{cfg.scalp_max_spread:.5f}"
        )
    return True, f"spread {spread:.5f} <= {cfg.scalp_max_spread:.5f}"


# ---------------------------------------------------------------------------
# SL / TP
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalpingSLTP:
    """Scalp stop/target with the provenance the report needs."""

    sl: float
    tp: float
    sl_distance: float
    tp_distance: float
    spread: float
    required_tp_distance: float
    source: str = "atr"


def build_scalping_sl_tp(
    direction: Signal,
    entry: float,
    atr_value: float,
    spread: float,
    cfg: Config,
    min_stop_distance: float = 0.0,
) -> Optional[ScalpingSLTP]:
    """Tiny ATR-based SL/TP that must clear the trade cost, or return None.

    ``entry`` is the expected fill *after* crossing the spread (for a buy the
    ask; for a sell the bid), so ``tp_distance`` is directly comparable to the
    round-trip cost. ``min_stop_distance`` is the broker's minimum allowed
    distance (``stops_level * point``); a scalp that cannot be placed without
    violating it is refused (``None``) instead of silently widened.
    """
    if direction not in (Signal.BUY, Signal.SELL):
        raise ValueError("direction must be Signal.BUY or Signal.SELL")
    entry_v = _clean(entry)
    atr_v = _clean(atr_value)
    if entry_v is None or entry_v <= 0:
        return None
    if atr_v is None or atr_v <= 0:
        return None

    sl_distance = atr_v * float(cfg.scalp_sl_atr_multiple)
    tp_distance = atr_v * float(cfg.scalp_tp_atr_multiple)
    required = required_tp_distance(effective_spread(cfg, spread), cfg)
    if tp_distance < required:
        return None

    floor = max(float(min_stop_distance), 0.0)
    if floor > 0 and (sl_distance < floor or tp_distance < floor):
        return None

    if direction is Signal.BUY:
        sl = entry_v - sl_distance
        tp = entry_v + tp_distance
        if sl <= 0:
            return None
    else:
        sl = entry_v + sl_distance
        tp = entry_v - tp_distance
        if tp <= 0:
            return None

    return ScalpingSLTP(
        sl=sl,
        tp=tp,
        sl_distance=sl_distance,
        tp_distance=tp_distance,
        spread=float(spread),
        required_tp_distance=required,
    )


# ---------------------------------------------------------------------------
# vectorised read-only view (same code path, ~10x faster over long frames)
# ---------------------------------------------------------------------------

class _ILocArray:
    """``.iloc``-shaped access over a numpy array."""

    __slots__ = ("_array",)

    def __init__(self, array) -> None:
        self._array = array

    def __getitem__(self, key: int):
        return self._array[key]


class _ColumnView:
    """Minimal ``Series`` stand-in: only what the strategy actually uses."""

    __slots__ = ("iloc",)

    def __init__(self, array) -> None:
        self.iloc = _ILocArray(array)


class _FrameView:
    """Minimal ``DataFrame`` stand-in (``df["col"].iloc[i]`` and ``len``)."""

    __slots__ = ("_columns", "_length")

    def __init__(self, columns: Dict[str, object], length: int) -> None:
        self._columns = columns
        self._length = length

    def __getitem__(self, name: str) -> _ColumnView:
        return self._columns[name]

    def __len__(self) -> int:
        return self._length


def fast_view(
    df: pd.DataFrame, ind: ScalpingIndicators
) -> Tuple[_FrameView, ScalpingIndicators]:
    """Return read-only numpy views of ``df``/``ind`` for the same decisions.

    The scalping decision reads exactly three things -- ``len(df)``,
    ``df["open"].iloc[i]`` and ``ind.<series>.iloc[i]``. Replacing pandas
    scalar access with numpy scalar access removes the per-bar indexing cost
    without duplicating a single line of decision logic, so the backtest and
    the live loop cannot drift apart (a test asserts the outputs are identical
    bar by bar).
    """
    columns = {
        name: _ColumnView(df[name].to_numpy(dtype=float))
        for name in REQUIRED_COLUMNS
        if name != "time"
    }
    view = _FrameView(columns, len(df))
    indicators = ScalpingIndicators(
        close=_ColumnView(ind.close.to_numpy(dtype=float)),
        basis=_ColumnView(ind.basis.to_numpy(dtype=float)),
        upper=_ColumnView(ind.upper.to_numpy(dtype=float)),
        lower=_ColumnView(ind.lower.to_numpy(dtype=float)),
        rsi=_ColumnView(ind.rsi.to_numpy(dtype=float)),
        atr=_ColumnView(ind.atr.to_numpy(dtype=float)),
        atr_rank=_ColumnView(ind.atr_rank.to_numpy(dtype=float)),
    )
    return view, indicators


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

def _outside_before(
    prev_close: float, prev_lower: float, prev_upper: float
) -> bool:
    """True when the previous close sat outside the band (either side)."""
    return prev_close <= prev_lower or prev_close >= prev_upper


def reversal_confirmed(
    index: int, df: pd.DataFrame, ind: ScalpingIndicators
) -> Optional[str]:
    """Classify the signal candle as a confirmed mean-reversion bar.

    Returns ``"BUY"``, ``"SELL"`` or ``None`` (no confirmed reversal). Uses
    rows ``index-1`` and ``index`` only.

    BUY  : previous close below the lower band, current close back inside it,
           current candle bullish, fast RSI rising.
    SELL : mirror image (previous close above the upper band, close back
           inside, bearish candle, RSI falling).
    """
    if index < 1 or index >= len(df):
        raise ValueError(f"index {index} out of range for {len(df)} candles")
    close = _clean(ind.close.iloc[index])
    open_v = _clean(df["open"].iloc[index])
    lower_prev = _clean(ind.lower.iloc[index - 1])
    upper_prev = _clean(ind.upper.iloc[index - 1])
    lower = _clean(ind.lower.iloc[index])
    upper = _clean(ind.upper.iloc[index])
    rsi_now = _clean(ind.rsi.iloc[index])
    rsi_prev = _clean(ind.rsi.iloc[index - 1])
    if None in (close, open_v, lower_prev, upper_prev, lower, upper, rsi_now, rsi_prev):
        return None

    if close <= lower_prev and close > lower and close > open_v and rsi_now > rsi_prev:
        return "BUY"
    if close >= upper_prev and close < upper and close < open_v and rsi_now < rsi_prev:
        return "SELL"
    return None


def excursion_extreme(index: int, ind: ScalpingIndicators) -> Optional[float]:
    """Signed band excursion of the previous close, in band widths.

    Positive = above the upper band, negative = below the lower band, ``0`` =
    inside. Band width is ``upper - lower`` at ``index-1``. Kept in the signal
    details so a report can show how far price actually stretched.
    """
    prev_close = _clean(ind.close.iloc[index - 1])
    lower = _clean(ind.lower.iloc[index - 1])
    upper = _clean(ind.upper.iloc[index - 1])
    if None in (prev_close, lower, upper) or upper <= lower:
        return None
    if prev_close > upper:
        return float((prev_close - upper) / (upper - lower))
    if prev_close < lower:
        return float((prev_close - lower) / (upper - lower))
    return 0.0


def evaluate_scalping_at(
    index: int,
    df: pd.DataFrame,
    ind: ScalpingIndicators,
    cfg: Config,
    spread: Optional[float] = None,
    trades_today: int = 0,
) -> TradeSignal:
    """Full scalping decision at closed candle ``index`` (rows 0..index only).

    Order of evaluation is deliberate: structural reasons first (history,
    daily cap, volatility, quoted spread), then the entry pattern, and only
    then the cost gate -- because a signal that cannot pay for the spread is
    NO_TRADE even when the pattern is perfect.
    """
    details: Dict[str, Any] = {"index": index}
    if index < 1 or index >= len(df):
        raise ValueError(f"index {index} out of range for {len(df)} candles")

    close = _clean(ind.close.iloc[index])
    atr_value = _clean(ind.atr.iloc[index])
    atr_rank = _clean(ind.atr_rank.iloc[index])
    rsi_value = _clean(ind.rsi.iloc[index])
    rsi_prev = _clean(ind.rsi.iloc[index - 1])
    lower_prev = _clean(ind.lower.iloc[index - 1])
    upper_prev = _clean(ind.upper.iloc[index - 1])
    details.update(
        {
            "close": close,
            "rsi": rsi_value,
            "rsi_prev": rsi_prev,
            "atr": atr_value,
            "atr_rank": atr_rank,
            "excursion": excursion_extreme(index, ind),
        }
    )
    if None in (close, atr_value, rsi_value):
        details["reject_code"] = "indicators_not_ready"
        return TradeSignal(
            Signal.NO_TRADE, "indicators not ready (close/ATR/RSI NaN)", details
        )

    if trades_today >= cfg.scalp_max_trades_per_day:
        details["reject_code"] = "daily_cap"
        return TradeSignal(
            Signal.NO_TRADE,
            f"daily scalp cap reached ({trades_today} >= "
            f"{cfg.scalp_max_trades_per_day})",
            details,
        )

    vol_ok, vol_reason = volatility_gate_result(atr_value, atr_rank, cfg)
    if not vol_ok:
        details["reject_code"] = "volatility"
        return TradeSignal(Signal.NO_TRADE, vol_reason, details)

    spread_v = effective_spread(cfg, spread)
    details["spread"] = spread_v
    spread_ok, spread_reason = spread_gate_result(spread_v, cfg)
    if not spread_ok:
        details["reject_code"] = "wide_spread"
        return TradeSignal(Signal.NO_TRADE, spread_reason, details)

    side = reversal_confirmed(index, df, ind)
    if side is None:
        details["reject_code"] = "no_reversal"
        return TradeSignal(
            Signal.NO_TRADE,
            "no confirmed reversal candle (excursion alone is not a signal)",
            details,
        )
    extreme_ok = (
        side == "BUY" and rsi_prev is not None and rsi_prev <= cfg.scalp_rsi_oversold
    ) or (
        side == "SELL" and rsi_prev is not None and rsi_prev >= cfg.scalp_rsi_overbought
    )
    details["rsi_extreme_ok"] = bool(extreme_ok)
    if not extreme_ok:
        details["reject_code"] = "rsi_extreme"
        return TradeSignal(
            Signal.NO_TRADE,
            f"reversal candle without an RSI extreme "
            f"(prev RSI {rsi_prev:.2f}, need <= {cfg.scalp_rsi_oversold:g} for buy "
            f"/ >= {cfg.scalp_rsi_overbought:g} for sell)",
            details,
        )
    details["band_extreme_ok"] = bool(
        (side == "BUY" and lower_prev is not None and close <= lower_prev)
        or (side == "SELL" and upper_prev is not None and close >= upper_prev)
    )

    # -- cost gate: applied to the *planned* target, before sizing --------
    is_buy = side == "BUY"
    entry_proxy = close  # live pricing re-checks this on the real tick
    sl_distance = atr_value * float(cfg.scalp_sl_atr_multiple)
    tp_distance = atr_value * float(cfg.scalp_tp_atr_multiple)
    required = required_tp_distance(spread_v, cfg)
    details.update(
        {
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "required_tp_distance": required,
            "entry_proxy": entry_proxy,
        }
    )
    gate_ok, gate_reason = cost_gate_result(tp_distance, spread_v, cfg)
    if not gate_ok:
        details["reject_code"] = "cost_gate"
        return TradeSignal(Signal.NO_TRADE, f"cost gate: {gate_reason}", details)

    reason = (
        f"mean-reversion {side}: prev close outside band, reversal bar back "
        f"inside, RSI {rsi_prev:.2f}->{rsi_value:.2f}, ATR {atr_value:.5f} "
        f"(pct {0.0 if atr_rank is None else atr_rank:.2%}), "
        f"TP {tp_distance:.5f} >= cost-required {required:.5f}"
    )
    return TradeSignal(Signal.BUY if is_buy else Signal.SELL, reason, details)


def generate_scalping_signal(
    df: pd.DataFrame,
    cfg: Config,
    spread: Optional[float] = None,
    trades_today: int = 0,
) -> TradeSignal:
    """Compute indicators and evaluate the last closed candle (live adapter)."""
    validate_scalping_config(cfg)
    if len(df) < 2:
        raise ValueError("at least 2 closed candles are required")
    if len(df) < cfg.scalp_min_candles_for_signal:
        return TradeSignal(
            Signal.NO_TRADE,
            f"insufficient candles for scalping ({len(df)} < "
            f"{cfg.scalp_min_candles_for_signal})",
            {"length": len(df)},
        )
    ind = compute_scalping_indicators(df, cfg)
    return evaluate_scalping_at(
        len(df) - 1, df, ind, cfg, spread=spread, trades_today=trades_today
    )
