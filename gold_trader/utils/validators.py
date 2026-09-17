"""Small pure validation helpers shared by the risk and order layers."""
from __future__ import annotations

import math
from typing import List


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
