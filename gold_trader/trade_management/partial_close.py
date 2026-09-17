"""Stateless partial closing driven by configurable R-multiple levels.

Levels are ``(R multiple, fraction of the ORIGINAL volume to close)``
pairs, e.g. ``((1.0, 0.5), (2.0, 0.3), (3.0, 1.0))`` means: close 50% of
the original volume at +1R, another 30% at +2R, and the remainder at +3R.
Nothing is hard-coded: levels come from ``Config.partial_close_levels``.

State is recomputed from the original volume (stored in the position
comment) and the current price, so a single action "catches up" when the
price jumps several levels within one candle -- and a close is never sent
twice. Before sending, the computed close volume is checked against the
broker ``volume_min``/``volume_step``; invalid orders are logged and
skipped, never sent.
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from ..config import Config
from ..models import ManagementAction, PositionInfo, SymbolSpec
from ..risk.position_size import round_volume_to_step
from .break_even import parse_position_comment

logger = logging.getLogger("gold_trader.management.partial_close")


def planned_remaining_volume(
    entry: float,
    is_buy: bool,
    initial_volume: float,
    current_price: float,
    levels: Sequence[Tuple[float, float]],
    r_distance: float,
) -> float:
    """Remaining volume that should stay open at the current price.

    A level counts as reached when the price has moved at least
    ``r_multiple * r_distance`` in the trade's favour. Fractions are
    summed (capped at 1.0) and applied to the original volume.
    """
    if r_distance <= 0 or initial_volume <= 0:
        return initial_volume
    total_fraction = 0.0
    for r_multiple, fraction in levels:
        if r_multiple <= 0 or fraction <= 0:
            continue
        if is_buy:
            reached = current_price >= entry + r_multiple * r_distance
        else:
            reached = current_price <= entry - r_multiple * r_distance
        if reached:
            total_fraction += fraction
    total_fraction = min(total_fraction, 1.0)
    return initial_volume * (1.0 - total_fraction)


def manage_partial_close(
    positions: Sequence[PositionInfo],
    bid: float,
    ask: float,
    cfg: Config,
    spec: SymbolSpec,
) -> List[ManagementAction]:
    """Return the partial/full close actions required right now."""
    if not cfg.partial_close_enabled or not cfg.partial_close_levels:
        return []
    actions: list[ManagementAction] = []
    has_final_level = any(
        fraction >= 1.0 for _, fraction in cfg.partial_close_levels
    )

    for position in positions:
        parsed = parse_position_comment(position.comment)
        if parsed is None:
            continue  # manual or unmanaged position -> never touch it
        initial_sl = parsed["initial_sl"]
        initial_volume = parsed["initial_volume"]
        r_distance = abs(position.price_open - initial_sl)
        if r_distance <= 0 or initial_volume <= 0:
            continue

        price = bid if position.is_buy else ask
        target = planned_remaining_volume(
            position.price_open,
            position.is_buy,
            initial_volume,
            price,
            cfg.partial_close_levels,
            r_distance,
        )
        close_volume = position.volume - target
        if close_volume <= 0:
            continue

        close_volume = round_volume_to_step(close_volume, spec.volume_step)
        if close_volume < spec.volume_min:
            if has_final_level and position.volume >= spec.volume_min:
                # closing the tiny remainder below the broker minimum is
                # impossible as a partial close -> close what is left
                close_volume = round_volume_to_step(position.volume, spec.volume_step)
                kind = "full_close"
                description = (
                    f"final level reached #{position.ticket}: closing remaining "
                    f"{close_volume:.2f} (computed partial was below minimum "
                    f"{spec.volume_min})"
                )
            else:
                logger.info(
                    "partial close skipped for #%s: computed close %.4f is below "
                    "broker minimum %.4f (no invalid order will be sent)",
                    position.ticket,
                    close_volume,
                    spec.volume_min,
                )
                continue
        else:
            kind = (
                "full_close"
                if close_volume >= position.volume - spec.volume_step / 2
                else "partial_close"
            )
            description = (
                f"{kind} #{position.ticket}: close {close_volume:.2f} of "
                f"{position.volume:.2f} (target remaining {target:.4f})"
            )
        actions.append(
            ManagementAction(
                kind=kind,
                ticket=position.ticket,
                close_volume=close_volume,
                description=description,
            )
        )
    return actions
