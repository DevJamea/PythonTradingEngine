"""Explicit, documented management exception for scalp positions.

A scalp holds a target measured in tens of cents on gold. Moving the stop to
break-even, closing half of it at +1R or trailing it by 2 ATR are swing-trade
mechanics: on this horizon break-even turns winners into scratches that still
paid the spread, a partial close halves a target that was already barely bigger
than the cost, and a trailing stop of any ATR multiple is wider than the whole
trade. Management here does not protect money, it manufactures cost.

So this module is the single place that says "not for scalps", instead of
scattering ``if scalping`` through the management functions:

* :func:`is_scalp_position` -- a position is a scalp when the bot's own
  comment carries :data:`SCALP_COMMENT_MARKER` (set at entry by the live loop).
* :func:`filter_management_actions` -- drops break-even / partial-close /
  trailing actions aimed at those positions while
  ``Config.scalping_disable_management`` is True (the default).
* :func:`format_scalp_comment` -- comment builder for entries (keeps the
  ``sl=..|vol=..`` state the other managers parse, plus the marker).

The exits of a scalp are therefore always: SL, TP, or end-of-data/manual.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..config import Config
from ..models import ManagementAction, PositionInfo, SymbolSpec
from ..strategy.scalping import SCALP_COMMENT_MARKER
from .break_even import format_position_comment

#: The only management kinds this layer may ever produce for a scalp.
#: ``full_close`` is allowed on purpose: closing a scalp by hand (or by the
#: daily-loss halt / shutdown path) is protective, not optimisation.
SCALP_ALLOWED_KINDS: tuple[str, ...] = ("full_close", "delete_order")


def format_scalp_comment(
    prefix: str,
    initial_sl: float,
    initial_volume: float,
    digits: int,
    volume_step: Optional[float] = None,
) -> str:
    """``<prefix>|<SCALP>|sl=..|vol=..`` -- the marker the exception looks for."""
    body = format_position_comment(prefix, initial_sl, initial_volume, digits, volume_step)
    return f"{body}|{SCALP_COMMENT_MARKER}"


def is_scalp_position(
    position: PositionInfo, cfg: Optional[Config] = None
) -> bool:
    """True when this position was opened by the scalping engine.

    Detection is by explicit marker only -- never by "the config currently
    says scalping", because the bot can switch engines while a position from
    the previous engine is still open, and that position must keep the
    management it was opened with.
    """
    if position is None:
        return False
    return SCALP_COMMENT_MARKER in (position.comment or "")


def management_allowed(position: PositionInfo, cfg: Config) -> bool:
    """Whether the optimising managers (break-even/partial/trailing) may act."""
    if cfg is None or not getattr(cfg, "scalping_disable_management", False):
        return True
    return not is_scalp_position(position, cfg)


def filter_management_actions(
    actions: Sequence[ManagementAction],
    positions: Sequence[PositionInfo],
    cfg: Config,
) -> List[ManagementAction]:
    """Drop optimising actions that target scalp positions.

    Unknown tickets are left alone (a management action for a position this
    layer cannot see is not this layer's decision to make).
    """
    if not getattr(cfg, "scalping_disable_management", False):
        return list(actions)
    scalp_tickets = {
        p.ticket for p in positions if is_scalp_position(p, cfg)
    }
    if not scalp_tickets:
        return list(actions)
    kept: List[ManagementAction] = []
    for action in actions:
        if action.ticket in scalp_tickets and action.kind not in SCALP_ALLOWED_KINDS:
            continue
        kept.append(action)
    return kept


def scalp_exits_only(
    positions: Sequence[PositionInfo],
    bid: float,
    ask: float,
    cfg: Config,
    spec: SymbolSpec,
) -> List[ManagementAction]:
    """The complete "management policy" for scalps: nothing.

    Kept as a named function so the behaviour is testable and visible in the
    API instead of being an absence: a scalp is closed only by its own SL or
    TP, so this always returns an empty list (and proves it does not silently
    inherit the swing managers).
    """
    return []
