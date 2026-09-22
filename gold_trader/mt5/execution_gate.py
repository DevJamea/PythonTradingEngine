"""Explicit permission to call ``mt5.order_send``.

The MT5 layer does not read ``.env`` or :class:`~gold_trader.config.Config`.
The application installs the configuration switches it already decided
(``TRADING_ENABLED`` / ``DRY_RUN``). Until that happens the default is deny.

Demo safety is not a switch and not a field a caller can set.
``account_is_demo=True`` is recorded only after this module reads
``account_info`` and proves ``ACCOUNT_TRADE_MODE_DEMO``. REAL, contest,
unknown, and a failed read all stay closed. A caller-supplied permission
may only narrow the configuration switches — it cannot reopen Demo safety,
and it cannot turn a REAL or unknown account into an execution allow.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator, Optional


def _is_mode_int(value: Any) -> bool:
    """True for a real broker constant, never for bool or MagicMock."""
    return isinstance(value, int) and not isinstance(value, bool)


def _read_trade_mode(info: Any) -> Any:
    """Trade mode from a terminal account object, or None if it is not there."""
    if info is None:
        return None
    if isinstance(info, dict):
        return info.get("trade_mode")
    as_dict = getattr(info, "_asdict", None)
    if callable(as_dict):
        try:
            data = as_dict()
        except Exception:
            data = None
        if isinstance(data, dict):
            return data.get("trade_mode")
    value = getattr(info, "trade_mode", None)
    if _is_mode_int(value):
        return value
    return None


def _read_account_is_demo() -> bool:
    """Fresh terminal read. Does not change the latch. False on any failure.

    Only ``trade_mode == ACCOUNT_TRADE_MODE_DEMO`` proves Demo. There is no
    fallback ``0``: a missing constant, a MagicMock constant, a missing
    ``trade_mode``, REAL, contest, and a failed ``account_info()`` all deny.
    """
    try:
        from .connection import require_mt5

        mt5_api = require_mt5()
        info = mt5_api.account_info()
        trade_mode = _read_trade_mode(info)
        demo_const = getattr(mt5_api, "ACCOUNT_TRADE_MODE_DEMO", None)
        real_const = getattr(mt5_api, "ACCOUNT_TRADE_MODE_REAL", None)
    except Exception:
        return False
    if not _is_mode_int(demo_const) or not _is_mode_int(trade_mode):
        return False
    if _is_mode_int(real_const) and trade_mode == real_const:
        return False
    return trade_mode == demo_const


# Proof the bot obtained by reading account_info. Callers cannot set this True.
_VERIFIED_DEMO: ContextVar[bool] = ContextVar(
    "gold_trader_verified_demo_account",
    default=False,
)


def verified_account_is_demo() -> bool:
    """True only after a successful Demo proof that has not been cleared."""
    return bool(_VERIFIED_DEMO.get())


def clear_verified_account_safety() -> None:
    """Drop any previous Demo proof. Execution stays closed until a new one."""
    _VERIFIED_DEMO.set(False)


def refresh_verified_account_safety() -> bool:
    """Read ``account_info`` and latch Demo only when that read proves it.

    This is the only way the latch becomes True. A caller cannot pass True.
    Failure and every non-demo result latch False.
    """
    proved = _read_account_is_demo()
    _VERIFIED_DEMO.set(proved)
    return proved


def demote_verified_demo_if_stale() -> None:
    """Clear a Demo latch the current terminal no longer proves.

    Used on the send path so a stale proof cannot reach ``order_check`` or
    ``order_send``. Never sets the latch True: a closed latch stays closed
    until :func:`refresh_verified_account_safety`.
    """
    if not _read_account_is_demo():
        clear_verified_account_safety()


@dataclass(frozen=True)
class ExecutionPermission:
    """Configuration switches, plus a mirror of the verified Demo latch.

    ``account_is_demo`` on a caller-built instance is not proof and does not
    grant ``order_send``. :func:`allows_order_send` reads the latch set by
    :func:`refresh_verified_account_safety`, never this constructor argument.
    Objects returned by :func:`get_execution_permission` copy the latch into
    the field so observers see the proof, not a declaration.
    """

    trading_enabled: bool = False
    dry_run: bool = True
    account_is_demo: bool = False

    @property
    def allows_order_send(self) -> bool:
        """True only for live switches and a broker-verified Demo latch."""
        return (
            bool(self.trading_enabled)
            and not bool(self.dry_run)
            and verified_account_is_demo()
        )

    def block_reason(self) -> Optional[str]:
        """Human-readable reason when a live send must not happen, else None."""
        if self.allows_order_send:
            return None
        reasons = []
        if not self.trading_enabled:
            reasons.append("TRADING_ENABLED=false")
        if self.dry_run:
            reasons.append("DRY_RUN=true")
        if not verified_account_is_demo():
            reasons.append("blocked by demo safety")
        return ", ".join(reasons) if reasons else "execution not permitted"

    def intersect(self, other: "ExecutionPermission") -> "ExecutionPermission":
        """The more restrictive configuration switches.

        An override cannot widen ``TRADING_ENABLED`` / ``DRY_RUN`` and cannot
        change Demo safety in either direction. The Demo field on the result
        is the verified latch, never ``other.account_is_demo``.
        """
        return _project(
            ExecutionPermission(
                trading_enabled=bool(self.trading_enabled) and bool(other.trading_enabled),
                dry_run=bool(self.dry_run) or bool(other.dry_run),
            )
        )


def _project(permission: ExecutionPermission) -> ExecutionPermission:
    """Switches from ``permission``. Demo field from the latch, never the caller."""
    return ExecutionPermission(
        trading_enabled=bool(permission.trading_enabled),
        dry_run=bool(permission.dry_run),
        account_is_demo=verified_account_is_demo(),
    )


# Default deny. A fresh process, and any context that has not been given an
# explicit permission, cannot reach order_send.
_PERMISSION: ContextVar[ExecutionPermission] = ContextVar(
    "gold_trader_execution_permission",
    default=ExecutionPermission(),
)


def get_execution_permission() -> ExecutionPermission:
    """Permission currently installed in this context.

    ``account_is_demo`` is the verified latch, not whatever a caller stored.
    """
    return _project(_PERMISSION.get())


def install_execution_permission(permission: ExecutionPermission) -> None:
    """Install configuration switches for the rest of this context.

    Does not read the environment, does not read the account, and does not
    restore the previous value. ``permission.account_is_demo`` is ignored:
    installing ``account_is_demo=True`` cannot turn a REAL or unknown account
    into an execution allow. :class:`~gold_trader.main.TradingBot` calls
    :func:`refresh_verified_account_safety` itself before installing.
    """
    _PERMISSION.set(_project(permission))


def resolve_execution_permission(
    override: Optional[ExecutionPermission] = None,
) -> ExecutionPermission:
    """Installed switches, optionally narrowed by ``override``.

    ``override`` cannot widen a closed gate and cannot supply Demo proof.
    ``account_is_demo=True`` on ``override`` is discarded.
    """
    installed = _PERMISSION.get()
    if override is not None:
        installed = installed.intersect(override)
    return _project(installed)


@contextmanager
def execution_permission(permission: ExecutionPermission) -> Iterator[ExecutionPermission]:
    """Temporarily install configuration switches, then restore the previous ones.

    Does not verify the account and does not accept ``account_is_demo=True``
    as proof. The previous Demo latch is left untouched (it is not opened).
    """
    stored = _project(permission)
    token: Token[ExecutionPermission] = _PERMISSION.set(stored)
    try:
        yield stored
    finally:
        _PERMISSION.reset(token)
