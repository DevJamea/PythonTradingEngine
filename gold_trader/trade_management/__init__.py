"""Trade management layer: break-even, partial close, trailing stop,
pending order hygiene. All decision functions are pure (positions +
prices + config in, ManagementActions out) so they can be unit-tested
without a terminal."""
from .break_even import (
    breakeven_target,
    format_position_comment,
    manage_break_even,
    parse_position_comment,
)
from .partial_close import manage_partial_close, planned_remaining_volume
from .pending_orders import expired_orders, find_duplicate, plan_cleanup
from .reconciliation import resolve_management_actions
from .trailing_stop import compute_trailing_sl, manage_trailing_stop

__all__ = [
    "breakeven_target",
    "format_position_comment",
    "manage_break_even",
    "parse_position_comment",
    "manage_partial_close",
    "planned_remaining_volume",
    "expired_orders",
    "find_duplicate",
    "plan_cleanup",
    "compute_trailing_sl",
    "manage_trailing_stop",
    "resolve_management_actions",
]
