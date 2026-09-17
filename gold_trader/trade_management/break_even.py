"""Break-even management: move the SL to entry +/- buffer once the price
moves ``break_even_r`` R in the bot's favour (R = initial SL distance).

State survives restarts because it is encoded in the position comment at
open time:  ``<prefix>|sl=<initial_sl>|vol=<initial_volume>``
(the prefix defaults to ``GB``).

Idempotency: the move is applied only when the new SL is strictly better
than the current one (by at least the broker minimum stop distance), so
repeated cycles never send the same modification twice. Positions whose
comment carries no state marker (e.g. manual trades) are never touched.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from ..config import Config
from ..models import ManagementAction, PositionInfo, SymbolSpec

_COMMENT_RE = re.compile(r"sl=(?P<sl>\d+(?:\.\d+)?)\|vol=(?P<vol>\d+(?:\.\d+)?)")


def format_position_comment(
    prefix: str, initial_sl: float, initial_volume: float, digits: int
) -> str:
    """Build the bot comment that stores the initial SL and volume."""
    return f"{prefix}|sl={initial_sl:.{digits}f}|vol={initial_volume:.2f}"


def parse_position_comment(comment: str) -> Optional[Dict[str, float]]:
    """Extract the stored state; None when the comment has no marker."""
    match = _COMMENT_RE.search(comment or "")
    if not match:
        return None
    return {
        "initial_sl": float(match.group("sl")),
        "initial_volume": float(match.group("vol")),
    }


def breakeven_target(position: PositionInfo, buffer: float) -> float:
    """Target SL at break-even: entry + buffer (buy) / entry - buffer (sell)."""
    if position.is_buy:
        return position.price_open + buffer
    return position.price_open - buffer


def _mark_price(position: PositionInfo, bid: float, ask: float) -> float:
    """Mark a buy position at bid, a sell position at ask."""
    return bid if position.is_buy else ask


def manage_break_even(
    positions: Sequence[PositionInfo],
    bid: float,
    ask: float,
    cfg: Config,
    spec: SymbolSpec,
) -> List[ManagementAction]:
    """Return the break-even moves required for the given bot positions."""
    if not cfg.break_even_enabled:
        return []
    actions: list[ManagementAction] = []
    min_improvement = spec.min_stop_distance()

    for position in positions:
        parsed = parse_position_comment(position.comment)
        if parsed is None:
            continue  # not a bot-managed position -> never touch it
        initial_sl = parsed["initial_sl"]
        r_distance = abs(position.price_open - initial_sl)
        if r_distance <= 0:
            continue

        price = _mark_price(position, bid, ask)
        if position.is_buy:
            if price < position.price_open + cfg.break_even_r * r_distance:
                continue
        else:
            if price > position.price_open - cfg.break_even_r * r_distance:
                continue

        new_sl = breakeven_target(position, cfg.break_even_buffer)
        if position.sl <= 0:
            improved = True  # no current SL -> any valid SL is an improvement
        elif position.is_buy:
            improved = new_sl > position.sl + min_improvement
        else:
            improved = new_sl < position.sl - min_improvement
        if not improved:
            continue  # already at/past break-even -> idempotent skip

        actions.append(
            ManagementAction(
                kind="move_sl",
                ticket=position.ticket,
                new_sl=new_sl,
                new_tp=position.tp,
                description=(
                    f"break-even #{position.ticket}: SL "
                    f"{position.sl:.5f} -> {new_sl:.5f}"
                ),
            )
        )
    return actions
