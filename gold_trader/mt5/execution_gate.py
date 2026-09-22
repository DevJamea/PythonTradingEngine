"""Explicit permission to call ``mt5.order_send``.

The MT5 layer does not read ``.env`` or :class:`~gold_trader.config.Config`.
The application (or a test) installs an :class:`ExecutionPermission` built
from the switches it already decided. Until that happens the default is
deny: ``trading_enabled=False``, ``dry_run=True``.

A live send is allowed only when trading is enabled AND dry-run is off.
Either flag alone blocks every path that goes through ``send_request``
(market, pending, SL/TP, close, delete). A caller-supplied permission can
only narrow the installed one — it cannot reopen a closed gate.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class ExecutionPermission:
    """Whether ``order_send`` may be reached.

    Construct this from an already-loaded config (or an explicit test
    opt-in). Do not build it by reading the environment inside the MT5 layer.
    """

    trading_enabled: bool = False
    dry_run: bool = True

    @property
    def allows_order_send(self) -> bool:
        """True only for the single live combination (enabled and not dry-run)."""
        return bool(self.trading_enabled) and not bool(self.dry_run)

    def block_reason(self) -> Optional[str]:
        """Human-readable reason when a live send must not happen, else None."""
        if self.allows_order_send:
            return None
        reasons = []
        if not self.trading_enabled:
            reasons.append("TRADING_ENABLED=false")
        if self.dry_run:
            reasons.append("DRY_RUN=true")
        return ", ".join(reasons) if reasons else "execution not permitted"

    def intersect(self, other: "ExecutionPermission") -> "ExecutionPermission":
        """The more restrictive of the two permissions.

        Used so an explicit per-call permission can tighten the installed
        gate but cannot bypass ``TRADING_ENABLED`` / ``DRY_RUN``.
        """
        return ExecutionPermission(
            trading_enabled=bool(self.trading_enabled) and bool(other.trading_enabled),
            dry_run=bool(self.dry_run) or bool(other.dry_run),
        )


# Default deny. A fresh process, and any context that has not been given an
# explicit permission, cannot reach order_send.
_PERMISSION: ContextVar[ExecutionPermission] = ContextVar(
    "gold_trader_execution_permission",
    default=ExecutionPermission(),
)


def get_execution_permission() -> ExecutionPermission:
    """Permission currently installed in this context."""
    return _PERMISSION.get()


def install_execution_permission(permission: ExecutionPermission) -> None:
    """Install ``permission`` for the rest of this context.

    Does not read the environment and does not restore the previous value.
    :class:`~gold_trader.main.TradingBot` calls this from its ``Config``
    before any send. Tests that need a temporary permission should use
    :func:`execution_permission` instead.
    """
    _PERMISSION.set(permission)


def resolve_execution_permission(
    override: Optional[ExecutionPermission] = None,
) -> ExecutionPermission:
    """Installed permission, optionally narrowed by ``override``.

    ``override`` cannot widen a closed gate. Passing an allow-all permission
    while the installed gate is deny still returns deny.
    """
    installed = get_execution_permission()
    if override is None:
        return installed
    return installed.intersect(override)


@contextmanager
def execution_permission(permission: ExecutionPermission) -> Iterator[ExecutionPermission]:
    """Temporarily install ``permission``, restoring the previous value after."""
    token: Token[ExecutionPermission] = _PERMISSION.set(permission)
    try:
        yield permission
    finally:
        _PERMISSION.reset(token)
