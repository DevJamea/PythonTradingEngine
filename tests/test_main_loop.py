from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gold_trader.main import TradingBot
from tests._helpers import build_df, make_cfg


class FakeLoopBot(TradingBot):
    def __init__(self, *, interval: float = 4.0, interrupt_in_cycle: bool = False) -> None:
        self.cfg = make_cfg(loop_interval_seconds=interval)
        self.log = MagicMock()
        self.error_log = MagicMock()
        self.started = 0
        self.cycles = 0
        self.shutdowns = 0
        self.interrupt_in_cycle = interrupt_in_cycle

    def start(self) -> None:
        self.started += 1

    def cycle(self) -> None:
        self.cycles += 1
        if self.interrupt_in_cycle:
            raise KeyboardInterrupt

    def shutdown(self) -> None:
        self.shutdowns += 1


def test_run_forever_continues_after_normal_cycle_and_waits_interval():
    bot = FakeLoopBot(interval=4.0)
    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 2:
            raise KeyboardInterrupt

    with (
        patch("gold_trader.main.time.monotonic", side_effect=[0.0, 0.2, 10.0, 10.2]),
        patch("gold_trader.main.time.sleep", side_effect=fake_sleep),
    ):
        bot.run_forever()

    assert bot.started == 1
    assert bot.cycles == 2
    assert bot.shutdowns == 1
    assert sleep_calls == pytest.approx([3.8, 3.8])


def test_run_once_executes_exactly_one_cycle_and_shuts_down():
    bot = FakeLoopBot(interval=4.0)

    assert bot.run_once() == 0

    assert bot.started == 1
    assert bot.cycles == 1
    assert bot.shutdowns == 1


def test_run_forever_ctrl_c_during_cycle_terminates_cleanly_without_sleep():
    bot = FakeLoopBot(interval=4.0, interrupt_in_cycle=True)

    with patch("gold_trader.main.time.sleep") as sleep:
        bot.run_forever()

    assert bot.started == 1
    assert bot.cycles == 1
    assert bot.shutdowns == 1
    sleep.assert_not_called()


def test_cycle_logs_repeated_closed_candle_as_visible_completed_cycle():
    """A normal duplicate-candle cycle must return, not silently look stopped."""
    bot = object.__new__(TradingBot)
    bot.cfg = make_cfg()
    bot.log = MagicMock()
    bot.spec = SimpleNamespace(name="XAUUSD")
    bot.conn = SimpleNamespace(is_connected=lambda: True)
    bot._last_candle_time = None
    bot._install_execution_gate = MagicMock()
    bot._build_market_state = MagicMock(return_value=object())
    bot._load_data = MagicMock(return_value=build_df([(1, 2, 0, 1), (2, 3, 1, 2)]))
    bot._latest_atr = MagicMock(return_value=None)
    bot._evaluate_entry = MagicMock()

    with (
        patch("gold_trader.main.mt5_positions.get_positions", return_value=[]),
        patch("gold_trader.main.mt5_orders.get_pending_orders", return_value=[]),
        patch("gold_trader.main.market_data.get_tick", return_value=SimpleNamespace(bid=2000.0, ask=2000.5)),
        patch("gold_trader.main.manage_break_even", return_value=[]),
        patch("gold_trader.main.manage_partial_close", return_value=[]),
        patch("gold_trader.main.manage_trailing_stop", return_value=[]),
        patch("gold_trader.main.plan_cleanup", return_value=[]),
    ):
        bot.cycle()
        bot.cycle()

    assert bot._evaluate_entry.call_count == 1
    bot.log.info.assert_any_call(
        "cycle complete: no new closed candle yet (last=%s)",
        bot._last_candle_time,
    )
