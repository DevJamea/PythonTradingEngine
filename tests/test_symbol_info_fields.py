"""Regression tests for official MetaTrader5 ``symbol_info()`` field names.

Root cause of the first real-run failure (Windows + MetaTrader 5 Demo):
``gold_trader/mt5/symbols.py`` read ``info.stops_level`` and
``info.freeze_level`` from ``MetaTrader5.symbol_info()`` / ``symbols_get()``,
but the official MetaQuotes objects expose those distances ONLY as:

* ``trade_stops_level``   (SYMBOL_TRADE_STOPS_LEVEL)
* ``trade_freeze_level``  (SYMBOL_TRADE_FREEZE_LEVEL)

The short names do not exist on real terminals, so ``_try_symbol()``
raised ``AttributeError`` while building ``SymbolSpec``.

The doubles below mirror the real API shape: official field names only,
with ``__slots__`` so the short names cannot exist -- exactly like the
real SymbolInfo object. If the code ever regresses to a short/incorrect
MT5 field name, these tests fail immediately. No real MT5 terminal or
Windows machine is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import gold_trader.mt5.connection as mt5_conn
from gold_trader.mt5.symbols import find_gold_symbol


class OfficialSymbolInfo:
    """Shape of a real ``MetaTrader5`` SymbolInfo: official names only.

    ``__slots__`` forbids the short ``stops_level``/``freeze_level``
    attributes entirely -- reading them raises ``AttributeError``, just
    like on a real terminal.
    """

    __slots__ = (
        "name",
        "visible",
        "trade_mode",
        "point",
        "digits",
        "volume_min",
        "volume_max",
        "volume_step",
        "trade_stops_level",
        "trade_freeze_level",
        "trade_contract_size",
        "trade_tick_size",
        "trade_tick_value",
        "filling_mode",
    )

    def __init__(
        self,
        name="XAUUSD",
        visible=True,
        trade_mode=4,
        # Deliberately distinctive values (NOT default_gold_spec()'s) so a
        # hard-coded constant instead of broker data fails the round-trip.
        point=0.001,
        digits=3,
        volume_min=0.1,
        volume_max=55.0,
        volume_step=0.1,
        trade_stops_level=37,
        trade_freeze_level=19,
        trade_contract_size=5.0,
        trade_tick_size=0.001,
        trade_tick_value=2.5,
        filling_mode=1,
    ):
        self.name = name
        self.visible = visible
        self.trade_mode = trade_mode
        self.point = point
        self.digits = digits
        self.volume_min = volume_min
        self.volume_max = volume_max
        self.volume_step = volume_step
        self.trade_stops_level = trade_stops_level
        self.trade_freeze_level = trade_freeze_level
        self.trade_contract_size = trade_contract_size
        self.trade_tick_size = trade_tick_size
        self.trade_tick_value = trade_tick_value
        self.filling_mode = filling_mode


class LegacySymbolInfo:
    """Historical project test-double shape (short ``stops_level`` names).

    Mirrors ``MockSymbolInfo`` in tests/test_mt5_execution.py; kept to
    prove the project's existing fallback/mock objects still work.
    """

    def __init__(self, name="XAUUSD"):
        self.name = name
        self.visible = True
        self.trade_mode = 4
        self.point = 0.01
        self.digits = 2
        self.volume_min = 0.01
        self.volume_max = 10.0
        self.volume_step = 0.01
        self.stops_level = 20
        self.freeze_level = 10
        self.trade_contract_size = 100.0
        self.trade_tick_size = 0.01
        self.trade_tick_value = 1.0
        self.filling_mode = 3


class _NoStopFreezeLevels:
    """Official shape but with the level fields missing entirely."""

    __slots__ = (
        "name",
        "visible",
        "trade_mode",
        "point",
        "digits",
        "volume_min",
        "volume_max",
        "volume_step",
        "trade_contract_size",
        "trade_tick_size",
        "trade_tick_value",
        "filling_mode",
    )

    def __init__(self):
        self.name = "XAUUSD"
        self.visible = True
        self.trade_mode = 4
        self.point = 0.001
        self.digits = 3
        self.volume_min = 0.1
        self.volume_max = 55.0
        self.volume_step = 0.1
        self.trade_contract_size = 5.0
        self.trade_tick_size = 0.001
        self.trade_tick_value = 2.5
        self.filling_mode = 1


@pytest.fixture
def mock_mt5():
    """Mocked MT5 module, same patching pattern as test_mt5_execution.py."""
    mock = MagicMock()
    mock.SYMBOL_TRADE_MODE_FULL = 4
    mock.last_error.return_value = (0, "Success")
    with patch.object(mt5_conn, "MT5_AVAILABLE", True), patch.object(
        mt5_conn, "_mt5", mock
    ):
        yield mock


def _discover(mock_mt5, sym):
    mock_mt5.symbols_get.return_value = [sym]
    mock_mt5.symbol_info.return_value = sym
    return find_gold_symbol(preferred="XAUUSD")


# ===========================================================================
# REGRESSION: official trade_stops_level / trade_freeze_level must be read
# ===========================================================================


def test_spec_reads_official_trade_stops_and_freeze_level(mock_mt5):
    """``_try_symbol`` must read ``trade_stops_level``/``trade_freeze_level``.

    Before the fix this raised ``AttributeError: ... has no attribute
    'stops_level'`` at ``info.stops_level`` -- exactly the Windows + MT5
    Demo first-run failure. The double exposes ONLY the official names.
    """
    spec = _discover(mock_mt5, OfficialSymbolInfo())
    assert spec.stops_level == 37
    assert spec.freeze_level == 19
    assert spec.min_stop_distance() == pytest.approx(37 * 0.001)


def test_reselect_path_uses_official_field_names(mock_mt5):
    """Hidden-symbol path: ``symbol_select`` + re-read must use official names."""
    hidden = OfficialSymbolInfo(visible=False)
    shown = OfficialSymbolInfo(visible=True, trade_stops_level=37,
                               trade_freeze_level=19)
    mock_mt5.symbols_get.return_value = [hidden]
    mock_mt5.symbol_select.return_value = True
    mock_mt5.symbol_info.return_value = shown
    spec = find_gold_symbol(preferred="XAUUSD")
    mock_mt5.symbol_select.assert_called_once_with("XAUUSD", True)
    assert spec.stops_level == 37
    assert spec.freeze_level == 19


def test_symbol_spec_full_path_reads_official_fields(mock_mt5):
    """Every SymbolSpec value must round-trip from broker data, no constants.

    The double has only official MetaTrader5 ``symbol_info()`` names; any
    wrong MT5 field name anywhere in the SymbolSpec build path raises
    ``AttributeError`` here, and any hard-coded value fails the equality
    checks (the values are deliberately unlike default_gold_spec()).
    """
    spec = _discover(mock_mt5, OfficialSymbolInfo())
    assert spec.name == "XAUUSD"
    assert spec.point == pytest.approx(0.001)
    assert spec.digits == 3
    assert spec.volume_min == pytest.approx(0.1)
    assert spec.volume_max == pytest.approx(55.0)
    assert spec.volume_step == pytest.approx(0.1)
    assert spec.stops_level == 37
    assert spec.freeze_level == 19
    assert spec.visible is True
    assert spec.trade_mode == 4
    assert spec.contract_size == pytest.approx(5.0)      # trade_contract_size
    assert spec.trade_tick_size == pytest.approx(0.001)  # trade_tick_size
    assert spec.trade_tick_value == pytest.approx(2.5)   # trade_tick_value
    assert spec.filling_mode == 1                        # filling_mode


# ===========================================================================
# Compatibility: existing project fallback/mock doubles keep working
# ===========================================================================


def test_legacy_short_name_double_still_supported(mock_mt5):
    """Project doubles using ``stops_level``/``freeze_level`` still work.

    Keeps the existing MockSymbolInfo-style fallback objects (see
    tests/test_mt5_execution.py) compatible with the fixed reader.
    """
    spec = _discover(mock_mt5, LegacySymbolInfo())
    assert spec.stops_level == 20
    assert spec.freeze_level == 10
    assert spec.contract_size == pytest.approx(100.0)


def test_missing_level_fields_raise_instead_of_silent_defaults(mock_mt5):
    """Neither official nor legacy names -> loud error, never a hard-coded 0.

    Broker data is required; substituting a constant default would hide
    the failure and corrupt stop/freeze distance validation.
    """
    with pytest.raises(AttributeError, match="trade_stops_level"):
        _discover(mock_mt5, _NoStopFreezeLevels())
