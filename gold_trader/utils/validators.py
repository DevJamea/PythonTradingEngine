"""Small pure validation helpers shared by the risk and order layers."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..models import PositionInfo, SymbolSpec


def ensure_finite_positive(name: str, value: float) -> float:
    """Return ``value`` as float, raising ValueError if not finite/positive."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return number


def validate_sl_tp(
    entry: float,
    sl: float,
    tp: float,
    is_buy: bool,
    point: float,
    stops_level: int,
) -> List[str]:
    """Validate SL/TP placement and broker minimum stop distance.

    Returns a list of human-readable problems (empty list == valid).
    """
    problems: list[str] = []
    try:
        entry_f, sl_f, tp_f = float(entry), float(sl), float(tp)
    except (TypeError, ValueError):
        return ["entry/SL/TP must be numbers"]

    if not math.isfinite(entry_f) or entry_f <= 0:
        problems.append(f"invalid entry price {entry!r}")
    if not math.isfinite(sl_f) or sl_f <= 0:
        problems.append(f"invalid stop loss {sl!r}")
    if not math.isfinite(tp_f) or tp_f <= 0:
        problems.append(f"invalid take profit {tp!r}")
    if problems:
        return problems

    if is_buy:
        if not sl_f < entry_f:
            problems.append("BUY stop loss must be below entry")
        if not entry_f < tp_f:
            problems.append("BUY take profit must be above entry")
    else:
        if not sl_f > entry_f:
            problems.append("SELL stop loss must be above entry")
        if not tp_f < entry_f:
            problems.append("SELL take profit must be below entry")

    min_distance = stops_level * point
    if min_distance > 0:
        if abs(entry_f - sl_f) < min_distance:
            problems.append(
                f"stop loss too close to entry (distance {abs(entry_f - sl_f):.5f} "
                f"< broker minimum {min_distance:.5f})"
            )
        if abs(entry_f - tp_f) < min_distance:
            problems.append(
                f"take profit too close to entry (distance {abs(entry_f - tp_f):.5f} "
                f"< broker minimum {min_distance:.5f})"
            )
    return problems


def round_price(value: float, digits: int) -> float:
    """Round a price to the broker's digit count."""
    return round(float(value), int(digits))


def validate_sl_modification(
    position: PositionInfo,
    new_sl: float,
    bid: float,
    ask: float,
    spec: SymbolSpec,
) -> List[str]:
    """Validate a proposed SL modification against market prices and broker rules.

    Constraints:
    - SL direction relative to current market quote (BUY evaluates to Bid, SELL to Ask)
    - Broker minimum stops distance (stops_level * point)
    - Broker freeze distance (freeze_level * point) for both proposed SL and existing SL
    - Protection invariant: BUY SL >= current SL; SELL SL <= current SL
    - Digit constraints: SL must not exceed spec.digits precision
    """
    problems: list[str] = []
    try:
        new_sl_f = float(new_sl)
        bid_f = float(bid)
        ask_f = float(ask)
    except (TypeError, ValueError):
        return ["SL and market prices must be numbers"]

    if not math.isfinite(new_sl_f) or new_sl_f <= 0:
        problems.append(f"invalid stop loss {new_sl!r}")
    if not math.isfinite(bid_f) or bid_f <= 0 or not math.isfinite(ask_f) or ask_f <= 0:
        problems.append("invalid market Bid/Ask quotes")
    if ask_f < bid_f:
        problems.append(f"invalid market spread (ask {ask_f} < bid {bid_f})")
    if problems:
        return problems

    # Symbol digits constraint
    rounded_sl = round(new_sl_f, spec.digits)
    if abs(new_sl_f - rounded_sl) > 1e-7:
        problems.append(
            f"new SL {new_sl} exceeds symbol digits constraint ({spec.digits})"
        )

    stops_dist = max(spec.stops_level, 0) * spec.point
    freeze_dist = max(spec.freeze_level, 0) * spec.point

    if position.is_buy:
        # For BUY: SL evaluated relative to Bid
        if new_sl_f >= bid_f:
            problems.append(f"BUY stop loss {new_sl_f:.5f} must be below Bid {bid_f:.5f}")
        else:
            dist_to_bid = bid_f - new_sl_f
            if stops_dist > 0 and dist_to_bid < stops_dist - 1e-9:
                problems.append(
                    f"stop loss too close to Bid (distance {dist_to_bid:.5f} "
                    f"< broker stops_level distance {stops_dist:.5f})"
                )
            if freeze_dist > 0 and dist_to_bid < freeze_dist - 1e-9:
                problems.append(
                    f"stop loss within freeze level of Bid (distance {dist_to_bid:.5f} "
                    f"< broker freeze distance {freeze_dist:.5f})"
                )
        if freeze_dist > 0 and position.sl > 0:
            if abs(bid_f - position.sl) < freeze_dist - 1e-9:
                problems.append(
                    "current stop loss is within freeze level of Bid (order is frozen)"
                )
        if position.sl > 0 and new_sl_f < position.sl - 1e-9:
            problems.append(
                f"new SL {new_sl_f:.5f} reduces protection below current SL {position.sl:.5f}"
            )
    else:
        # For SELL: SL evaluated relative to Ask
        if new_sl_f <= ask_f:
            problems.append(f"SELL stop loss {new_sl_f:.5f} must be above Ask {ask_f:.5f}")
        else:
            dist_to_ask = new_sl_f - ask_f
            if stops_dist > 0 and dist_to_ask < stops_dist - 1e-9:
                problems.append(
                    f"stop loss too close to Ask (distance {dist_to_ask:.5f} "
                    f"< broker stops_level distance {stops_dist:.5f})"
                )
            if freeze_dist > 0 and dist_to_ask < freeze_dist - 1e-9:
                problems.append(
                    f"stop loss within freeze level of Ask (distance {dist_to_ask:.5f} "
                    f"< broker freeze distance {freeze_dist:.5f})"
                )
        if freeze_dist > 0 and position.sl > 0:
            if abs(position.sl - ask_f) < freeze_dist - 1e-9:
                problems.append(
                    "current stop loss is within freeze level of Ask (order is frozen)"
                )
        if position.sl > 0 and new_sl_f > position.sl + 1e-9:
            problems.append(
                f"new SL {new_sl_f:.5f} reduces protection below current SL {position.sl:.5f}"
            )

    return problems
