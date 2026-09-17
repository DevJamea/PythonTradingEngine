"""MT5 terminal connection: initialize / shutdown / status.

The MetaTrader5 package is Windows-only. The import is guarded so that
this module (and therefore the whole project) remains importable on other
platforms for unit tests and data analysis.

Credentials (optional) come from environment variables via
``TradingBot`` -- never from code, and never logged.
"""
from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, Dict, Optional

try:
    import MetaTrader5 as _mt5
    MT5_AVAILABLE: bool = True
    MT5_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # non-Windows platform
    _mt5 = None  # type: ignore[assignment]
    MT5_AVAILABLE = False
    MT5_IMPORT_ERROR = exc

from ..models import AccountMode

logger = logging.getLogger("gold_trader.mt5.connection")


class MT5Error(Exception):
    """Base error for the MT5 integration layer."""


class MT5ConnectionError(MT5Error):
    """The terminal could not be reached or initialized."""


class MT5DataError(MT5Error):
    """Market/account data could not be retrieved."""


def require_mt5() -> ModuleType:
    """Return the MetaTrader5 module, raising MT5Error when unavailable."""
    if not MT5_AVAILABLE:
        raise MT5Error(
            "MetaTrader5 package is not available on this platform "
            f"({MT5_IMPORT_ERROR}). MT5 terminal automation requires Windows."
        )
    assert _mt5 is not None
    return _mt5


class MT5Connection:
    """Owns the lifecycle of one MT5 terminal connection."""

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        terminal_path: Optional[str] = None,
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._terminal_path = terminal_path
        self._initialized = False

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        """Connect to the MT5 terminal (already logged in or via env vars)."""
        mt5_api = require_mt5()
        kwargs: Dict[str, Any] = {}
        if self._terminal_path:
            kwargs["path"] = self._terminal_path
        if self._login:
            kwargs["login"] = int(self._login)
            kwargs["password"] = self._password or ""
            kwargs["server"] = self._server or ""
        if not mt5_api.initialize(**kwargs):
            raise MT5ConnectionError(
                f"mt5.initialize() failed: {self._last_error()} "
                "(is the MT5 terminal running and logged in?)"
            )
        self._initialized = True
        info = self.account_info()
        login_tail = str(info.get("login", ""))[-4:]
        logger.info(
            "Connected to MT5 account ****%s on %r (currency %s)",
            login_tail,
            info.get("server"),
            info.get("currency"),
        )

    def shutdown(self) -> None:
        """Graceful shutdown; safe to call multiple times."""
        if self._initialized and MT5_AVAILABLE and _mt5 is not None:
            try:
                _mt5.shutdown()
            except Exception as exc:  # shutdown must not mask the caller's error
                logger.error("error during MT5 shutdown: %s", exc)
        self._initialized = False

    def __enter__(self) -> "MT5Connection":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    # -- status / info -------------------------------------------------------

    def is_connected(self) -> bool:
        """True when the terminal answers."""
        if not MT5_AVAILABLE or not self._initialized:
            return False
        return _mt5 is not None and _mt5.terminal_info() is not None

    @staticmethod
    def _last_error() -> str:
        if MT5_AVAILABLE and _mt5 is not None:
            return str(_mt5.last_error())
        return "unknown error"

    def _require_ready(self) -> ModuleType:
        mt5_api = require_mt5()
        if not self._initialized:
            raise MT5ConnectionError("MT5 not initialized - call initialize() first")
        return mt5_api

    def account_info(self) -> Dict[str, Any]:
        """Account details (balance, equity, mode, ...)."""
        mt5_api = self._require_ready()
        info = mt5_api.account_info()
        if info is None:
            raise MT5ConnectionError(
                f"account_info() failed: {self._last_error()}"
            )
        return info._asdict() if hasattr(info, "_asdict") else dict(info)

    def terminal_info(self) -> Dict[str, Any]:
        """Terminal details (company, connected, trade_allowed, ...)."""
        mt5_api = self._require_ready()
        info = mt5_api.terminal_info()
        if info is None:
            raise MT5ConnectionError(
                f"terminal_info() failed: {self._last_error()}"
            )
        return info._asdict() if hasattr(info, "_asdict") else dict(info)

    def account_mode(self) -> AccountMode:
        """NETTING or HEDGING -- never assume which one the account uses."""
        info = self.account_info()
        mode = info.get("margin_mode")
        if mode == 0:
            return AccountMode.HEDGING
        if mode == 1:
            return AccountMode.NETTING
        raise MT5Error(f"unknown account margin mode: {mode!r}")
