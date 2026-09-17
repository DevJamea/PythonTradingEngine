"""Central, environment-overridable configuration.

Nothing sensitive (passwords, account logins, API keys) belongs in this
file: MT5 connection credentials come from environment variables (see
``.env.example``) or from an already-logged-in local MT5 terminal.

Every field of :class:`Config` can be overridden through environment
variables; code defaults are intentionally safe (``TRADING_ENABLED``
False, ``DRY_RUN`` True, trailing stop off).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is part of requirements
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv()


# ---------------------------------------------------------------------------
# small typed env readers
# ---------------------------------------------------------------------------

def _env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a string env var; empty values fall back to ``default``."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_tuple(key: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_partial_close_levels(
    key: str, default: Tuple[Tuple[float, float], ...]
) -> Tuple[Tuple[float, float], ...]:
    """Parse ``R:fraction`` pairs, e.g. ``1.0:0.5,2.0:0.3,3.0:1.0``."""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    levels: list[tuple[float, float]] = []
    for part in raw.split(","):
        if ":" not in part:
            continue
        left, right = part.split(":", 1)
        try:
            levels.append((float(left), float(right)))
        except ValueError:
            continue
    return tuple(levels)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """All tunables of the bot in one place (no scattered magic numbers)."""

    # -- symbol / market data -------------------------------------------
    symbol: Optional[str] = None
    gold_symbol_candidates: Tuple[str, ...] = (
        "XAUUSD", "GOLD", "XAUUSDm", "XAUUSD.a", "GOLDm", "GOLD.a",
        "XAUUSD.ecn", "GOLD.ecn", "XAUUSDc", "GOLDc",
    )
    timeframe: str = "M15"
    candles_count: int = 500
    loop_interval_seconds: float = 45.0
    magic_number: int = 123456789
    position_comment_prefix: str = "GB"

    # -- safety gates ----------------------------------------------------
    trading_enabled: bool = False
    dry_run: bool = True
    expected_account_mode: Optional[str] = None  # e.g. "HEDGING" or "NETTING"

    # -- risk -------------------------------------------------------------
    risk_per_trade: float = 0.005
    max_daily_loss: float = 500.0
    max_open_positions: int = 1
    max_pending_orders: int = 2
    max_spread: float = 0.50
    trading_start_time: str = "00:00"
    trading_end_time: str = "23:59"

    # -- stop loss / take profit ------------------------------------------
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.5
    use_levels_for_sl: bool = False
    use_levels_for_tp: bool = False
    level_buffer_atr: float = 0.10

    # -- strategy -----------------------------------------------------------
    ema_fast: int = 20
    ema_medium: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    trend_require_close: bool = True
    buy_patterns: Tuple[str, ...] = ("bullish_engulfing", "hammer")
    sell_patterns: Tuple[str, ...] = ("bearish_engulfing", "shooting_star")
    doji_body_fraction: float = 0.10
    level_lookback: int = 100
    level_pivot_window: int = 5
    min_candles_for_signal: int = 210

    # -- position management -------------------------------------------------
    break_even_enabled: bool = True
    break_even_r: float = 1.0
    break_even_buffer: float = 0.10
    partial_close_enabled: bool = True
    partial_close_levels: Tuple[Tuple[float, float], ...] = (
        (1.0, 0.50), (2.0, 0.30), (3.0, 1.00),
    )
    trailing_stop_enabled: bool = False
    trailing_atr_multiplier: float = 2.0

    # -- execution -------------------------------------------------------------
    max_deviation: int = 20

    # -- backtest ---------------------------------------------------------------
    backtest_initial_balance: float = 10_000.0
    backtest_spread_cost: float = 0.15
    backtest_candles: int = 3000

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from environment variables (see .env.example)."""
        cfg = cls()
        overrides: Dict[str, Any] = {
            "symbol": _env_str("SYMBOL"),
            "gold_symbol_candidates": _env_tuple(
                "GOLD_SYMBOL_CANDIDATES", cfg.gold_symbol_candidates
            ),
            "timeframe": (_env_str("TIMEFRAME", cfg.timeframe) or cfg.timeframe).upper(),
            "candles_count": _env_int("CANDLES_COUNT", cfg.candles_count),
            "loop_interval_seconds": _env_float(
                "LOOP_INTERVAL_SECONDS", cfg.loop_interval_seconds
            ),
            "magic_number": _env_int("MAGIC_NUMBER", cfg.magic_number),
            "position_comment_prefix": _env_str(
                "POSITION_COMMENT_PREFIX", cfg.position_comment_prefix
            ),
            "trading_enabled": _env_bool("TRADING_ENABLED", cfg.trading_enabled),
            "dry_run": _env_bool("DRY_RUN", cfg.dry_run),
            "expected_account_mode": _env_str("EXPECTED_ACCOUNT_MODE"),
            "risk_per_trade": _env_float("RISK_PER_TRADE", cfg.risk_per_trade),
            "max_daily_loss": _env_float("MAX_DAILY_LOSS", cfg.max_daily_loss),
            "max_open_positions": _env_int("MAX_OPEN_POSITIONS", cfg.max_open_positions),
            "max_pending_orders": _env_int("MAX_PENDING_ORDERS", cfg.max_pending_orders),
            "max_spread": _env_float("MAX_SPREAD", cfg.max_spread),
            "trading_start_time": _env_str("TRADING_START_TIME", cfg.trading_start_time),
            "trading_end_time": _env_str("TRADING_END_TIME", cfg.trading_end_time),
            "atr_period": _env_int("ATR_PERIOD", cfg.atr_period),
            "atr_sl_multiplier": _env_float("ATR_SL_MULTIPLIER", cfg.atr_sl_multiplier),
            "atr_tp_multiplier": _env_float("ATR_TP_MULTIPLIER", cfg.atr_tp_multiplier),
            "use_levels_for_sl": _env_bool("USE_LEVELS_FOR_SL", cfg.use_levels_for_sl),
            "use_levels_for_tp": _env_bool("USE_LEVELS_FOR_TP", cfg.use_levels_for_tp),
            "level_buffer_atr": _env_float("LEVEL_BUFFER_ATR", cfg.level_buffer_atr),
            "ema_fast": _env_int("EMA_FAST", cfg.ema_fast),
            "ema_medium": _env_int("EMA_MEDIUM", cfg.ema_medium),
            "ema_slow": _env_int("EMA_SLOW", cfg.ema_slow),
            "rsi_period": _env_int("RSI_PERIOD", cfg.rsi_period),
            "rsi_overbought": _env_float("RSI_OVERBOUGHT", cfg.rsi_overbought),
            "rsi_oversold": _env_float("RSI_OVERSOLD", cfg.rsi_oversold),
            "trend_require_close": _env_bool("TREND_REQUIRE_CLOSE", cfg.trend_require_close),
            "buy_patterns": _env_tuple("BUY_PATTERNS", cfg.buy_patterns),
            "sell_patterns": _env_tuple("SELL_PATTERNS", cfg.sell_patterns),
            "doji_body_fraction": _env_float("DOJI_BODY_FRACTION", cfg.doji_body_fraction),
            "level_lookback": _env_int("LEVEL_LOOKBACK", cfg.level_lookback),
            "level_pivot_window": _env_int("LEVEL_PIVOT_WINDOW", cfg.level_pivot_window),
            "min_candles_for_signal": _env_int(
                "MIN_CANDLES_FOR_SIGNAL", cfg.min_candles_for_signal
            ),
            "break_even_enabled": _env_bool("BREAK_EVEN_ENABLED", cfg.break_even_enabled),
            "break_even_r": _env_float("BREAK_EVEN_R", cfg.break_even_r),
            "break_even_buffer": _env_float("BREAK_EVEN_BUFFER", cfg.break_even_buffer),
            "partial_close_enabled": _env_bool(
                "PARTIAL_CLOSE_ENABLED", cfg.partial_close_enabled
            ),
            "partial_close_levels": _env_partial_close_levels(
                "PARTIAL_CLOSE_LEVELS", cfg.partial_close_levels
            ),
            "trailing_stop_enabled": _env_bool(
                "TRAILING_STOP_ENABLED", cfg.trailing_stop_enabled
            ),
            "trailing_atr_multiplier": _env_float(
                "TRAILING_ATR_MULTIPLIER", cfg.trailing_atr_multiplier
            ),
            "max_deviation": _env_int("MAX_DEVIATION", cfg.max_deviation),
            "backtest_initial_balance": _env_float(
                "BACKTEST_INITIAL_BALANCE", cfg.backtest_initial_balance
            ),
            "backtest_spread_cost": _env_float(
                "BACKTEST_SPREAD_COST", cfg.backtest_spread_cost
            ),
            "backtest_candles": _env_int("BACKTEST_CANDLES", cfg.backtest_candles),
            "log_level": _env_str("LOG_LEVEL", cfg.log_level),
        }
        return replace(cfg, **overrides)
