"""Shared test fixtures.

Live ``order_send`` is deny-by-default for every test. Execution-layer tests
that need to inspect a real request opt in explicitly via
``execution_permission``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import gold_trader.mt5.connection as mt5_conn
from gold_trader.mt5.execution_gate import (
    ExecutionPermission,
    clear_verified_account_safety,
    execution_permission,
)


@pytest.fixture(autouse=True)
def _deny_live_execution_by_default():
    """Restore the safe default even if a test installs a live permission.

    Also drops any Demo proof. A previous test's verified latch must not
    leak, and a test cannot opt in by writing ``account_is_demo=True``.
    """
    clear_verified_account_safety()
    with execution_permission(ExecutionPermission()):
        yield
    clear_verified_account_safety()


@pytest.fixture
def mt5_stub():
    """MT5 module stub that counts ``order_send`` without permitting it.

    Permission stays at the safe default. A regression that reaches
    ``order_send`` increments ``order_send.call_count``.
    """
    mock = MagicMock()
    mock.TRADE_ACTION_DEAL = 1
    mock.TRADE_ACTION_PENDING = 5
    mock.TRADE_ACTION_SLTP = 6
    mock.TRADE_ACTION_REMOVE = 8
    mock.ORDER_TYPE_BUY = 0
    mock.ORDER_TYPE_SELL = 1
    mock.ORDER_TYPE_BUY_LIMIT = 2
    mock.ORDER_TYPE_SELL_LIMIT = 3
    mock.ORDER_TYPE_BUY_STOP = 4
    mock.ORDER_TYPE_SELL_STOP = 5
    mock.ORDER_TIME_GTC = 0
    mock.ORDER_FILLING_FOK = 0
    mock.ORDER_FILLING_IOC = 1
    mock.ORDER_FILLING_RETURN = 2
    mock.TRADE_RETCODE_DONE = 10009
    mock.TRADE_RETCODE_PLACED = 10008
    mock.TRADE_RETCODE_DONE_PARTIAL = 10010
    mock.TRADE_RETCODE_INVALID_STOPS = 10016
    mock.TRADE_RETCODE_INVALID_PRICE = 10015
    mock.SYMBOL_TRADE_MODE_FULL = 4
    mock.ACCOUNT_TRADE_MODE_DEMO = 0
    mock.ACCOUNT_TRADE_MODE_REAL = 2
    mock.order_check.return_value = MagicMock(retcode=0, comment="Check OK")
    mock.order_send.return_value = MagicMock(retcode=10009, comment="Done", order=12345)
    mock.last_error.return_value = (0, "Success")
    mock.history_deals_get.return_value = []

    with patch.object(mt5_conn, "MT5_AVAILABLE", True), patch.object(mt5_conn, "_mt5", mock):
        yield mock
