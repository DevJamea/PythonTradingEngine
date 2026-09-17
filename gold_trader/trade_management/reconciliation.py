"""Management action reconciliation.

Ensures contradictory sequential SL modifications are eliminated and the
safety invariant holds: once a stronger SL has been calculated during a
management cycle, a later action must never overwrite it with a weaker SL.
The final SL applied during a cycle is the most protective valid SL.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from ..models import ManagementAction, PositionInfo


def resolve_management_actions(
    actions: Sequence[ManagementAction],
    positions: Sequence[PositionInfo],
) -> List[ManagementAction]:
    """Combine and resolve management actions for a cycle.

    For each position with proposed `move_sl` actions:
    - For BUY: the strongest (highest) valid SL is selected.
    - For SELL: the strongest (lowest) valid SL is selected.
    - An action is rejected if the proposed SL weakens the current position SL.
    Non-SL actions (partial closes, full closes, pending deletions) are preserved.
    """
    if not actions:
        return []

    pos_by_ticket: Dict[int, PositionInfo] = {p.ticket: p for p in positions}
    non_sl_actions: List[ManagementAction] = []
    sl_actions_by_ticket: Dict[int, List[ManagementAction]] = {}

    for action in actions:
        if action.kind == "move_sl":
            sl_actions_by_ticket.setdefault(action.ticket, []).append(action)
        else:
            non_sl_actions.append(action)

    resolved_sl_actions: List[ManagementAction] = []

    for ticket, sl_actions in sl_actions_by_ticket.items():
        position = pos_by_ticket.get(ticket)
        if position is None:
            # Unknown position, pass first action conservatively
            resolved_sl_actions.append(sl_actions[0])
            continue

        valid_proposals = [a for a in sl_actions if a.new_sl is not None]
        if not valid_proposals:
            continue

        if position.is_buy:
            # For BUY: highest SL gives strongest protection
            best_action = max(valid_proposals, key=lambda a: float(a.new_sl))
            # Must not weaken current SL
            if position.sl > 0 and float(best_action.new_sl) < position.sl:
                continue
        else:
            # For SELL: lowest SL gives strongest protection
            best_action = min(valid_proposals, key=lambda a: float(a.new_sl))
            # Must not weaken current SL
            if position.sl > 0 and float(best_action.new_sl) > position.sl:
                continue

        resolved_sl_actions.append(best_action)

    return resolved_sl_actions + non_sl_actions
