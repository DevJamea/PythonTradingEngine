"""ATR trailing stop.

Disabled by default (``TRAILING_STOP_ENABLED = False``) because an
untested feature should not manage live stops. When enabled, the SL is
moved behind the current price by ``trailing_atr_multiplier * ATR`` and
is only applied when it improves the current SL by at least the broker
minimum stop distance (idempotent across cycles).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..config import Config
from ..models import ManagementAction, PositionInfo, SymbolSpec
from ..utils.validators import round_price, validate_sl_modification


def compute_trailing_sl(
    position: PositionInfo,
    price: float,
    atr_value: Optional[float],
    cfg: Config,
    spec: SymbolSpec,
) -> Optional[float]:
    """Candidate trailing SL for one position, or None when not better."""
    if not cfg.trailing_stop_enabled:
        return None
    if atr_value is None or atr_value <= 0:
        return None
    distance = atr_value * cfg.trailing_atr_multiplier
    min_improvement = spec.min_stop_distance()

    if position.is_buy:
        candidate = price - distance
        improved = position.sl <= 0 or candidate > position.sl + min_improvement
    else:
        candidate = price + distance
        improved = position.sl <= 0 or candidate < position.sl - min_improvement
    if not improved:
        return None
    return round_price(candidate, spec.digits)


def manage_trailing_stop(
    positions: Sequence[PositionInfo],
    bid: float,
    ask: float,
    atr_value: Optional[float],
    cfg: Config,
    spec: SymbolSpec,
) -> List[ManagementAction]:
    """Return trailing-stop move actions for the given bot positions."""
    if not cfg.trailing_stop_enabled:
        return []
    actions: list[ManagementAction] = []
    for position in positions:
        price = bid if position.is_buy else ask
        candidate = compute_trailing_sl(position, price, atr_value, cfg, spec)
        if candidate is None:
            continue
        problems = validate_sl_modification(position, candidate, bid, ask, spec)
        if problems:
            continue
        actions.append(
            ManagementAction(
                kind="move_sl",
                ticket=position.ticket,
                new_sl=candidate,
                new_tp=position.tp,
                description=(
                    f"trailing stop #{position.ticket}: SL "
                    f"{position.sl:.5f} -> {candidate:.5f}"
                ),
            )
        )
    return actions
