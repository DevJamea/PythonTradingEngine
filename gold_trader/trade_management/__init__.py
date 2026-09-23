"""Trade management layer: break-even, partial close, trailing stop,
pending order hygiene, and the explicit no-management exception for scalps.

All decision functions are pure (positions +
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
from .scalping_policy import (
    SCALP_ALLOWED_KINDS,
    filter_management_actions,
    format_scalp_comment,
    is_scalp_position,
    management_allowed,
    scalp_exits_only,
)
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
    "SCALP_ALLOWED_KINDS",
    "filter_management_actions",
    "format_scalp_comment",
    "is_scalp_position",
    "management_allowed",
    "scalp_exits_only",
]
