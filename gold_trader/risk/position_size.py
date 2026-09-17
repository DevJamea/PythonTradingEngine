"""Position sizing based on a risk amount, never on a fixed lot.

The volume is derived so that hitting the stop loss loses approximately
``balance * risk_percent``. Broker properties (minimum/maximum volume,
volume step, tick value/size or contract size) are always respected.
If a safe size cannot be computed, the result volume is None -> NO TRADE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..models import SymbolSpec


@dataclass(frozen=True)
class PositionSizingResult:
    """Outcome of position sizing. ``volume`` is None when unsafe."""

    volume: Optional[float]
    risk_amount: float
    raw_volume: float
    reason: str


def round_volume_to_step(volume: float, step: float) -> float:
    """Round volume DOWN to the broker volume step.

    Rounding down guarantees the size never exceeds the risk target.
    A tiny epsilon guards against float dust (e.g. 0.1 + 0.2 artifacts).
    """
    if step <= 0:
        raise ValueError("volume step must be positive")
    steps = math.floor(volume / step + 1e-9)
    return round(steps * step, 10)


def profit_for_volume(
    price_distance: float, volume: float, spec: SymbolSpec
) -> Optional[float]:
    """Absolute P/L (currency) if ``volume`` moves ``price_distance`` units.

    Prefers the broker's exact tick value/size (``trade_tick_value`` per
    ``trade_tick_size``); falls back to ``contract_size`` (1 lot moving
    the full distance yields contract_size currency units). Returns None
    when neither is available.
    """
    if price_distance < 0:
        raise ValueError("price_distance must be >= 0")
    if spec.trade_tick_size > 0 and spec.trade_tick_value > 0:
        per_lot = price_distance / spec.trade_tick_size * spec.trade_tick_value
    elif spec.contract_size > 0:
        per_lot = price_distance * spec.contract_size
    else:
        return None
    return per_lot * volume


def calculate_position_size(
    balance: float,
    risk_percent: float,
    entry: float,
    stop_loss: float,
    spec: SymbolSpec,
) -> PositionSizingResult:
    """Compute the trade volume for a given risk amount and stop distance.

    Returns ``volume=None`` (=> NO TRADE) whenever a safe size cannot be
    determined (invalid inputs, below broker minimum, unknown loss per lot).
    Volumes above the broker maximum are clamped down (actual risk then
    falls below the target, which is the safe direction).
    """
    try:
        balance_f = float(balance)
        risk_f = float(risk_percent)
        entry_f = float(entry)
        sl_f = float(stop_loss)
    except (TypeError, ValueError) as exc:
        return PositionSizingResult(None, 0.0, 0.0, f"invalid numeric input: {exc}")

    if not all(
        math.isfinite(v) for v in (balance_f, risk_f, entry_f, sl_f)
    ) or balance_f <= 0 or risk_f <= 0 or entry_f <= 0 or sl_f <= 0:
        return PositionSizingResult(
            None, 0.0, 0.0, "balance, risk_percent, entry and stop loss must be positive"
        )
    if entry_f == sl_f:
        return PositionSizingResult(None, 0.0, 0.0, "stop loss equals entry")
    if spec.volume_step <= 0:
        return PositionSizingResult(
            None, 0.0, 0.0, f"invalid broker volume step {spec.volume_step}"
        )

    distance = abs(entry_f - sl_f)
    per_lot = profit_for_volume(distance, 1.0, spec)
    if per_lot is None or per_lot <= 0:
        return PositionSizingResult(
            None,
            0.0,
            0.0,
            "cannot compute loss per lot from symbol properties "
            "(need trade_tick_value/size or contract_size)",
        )

    risk_amount = balance_f * risk_f
    raw_volume = risk_amount / per_lot
    volume = round_volume_to_step(raw_volume, spec.volume_step)

    if volume < spec.volume_min:
        return PositionSizingResult(
            None,
            risk_amount,
            raw_volume,
            f"required volume {volume:.4f} is below broker minimum "
            f"{spec.volume_min} (stop distance {distance:.5f} is too small "
            f"for this risk level)",
        )

    if volume > spec.volume_max:
        volume = spec.volume_max
        reason = (
            f"clamped to broker maximum {spec.volume_max} "
            "(actual risk will be below the target)"
        )
    else:
        reason = "ok"
    return PositionSizingResult(volume, risk_amount, raw_volume, reason)
