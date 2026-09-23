"""Wiring tests: the scalping engine exists, but nothing turns it on.

Covers Config plumbing (env overrides + safe defaults) and the live-loop
integration in ``gold_trader/main.py`` -- including that the original trend
engine is untouched when the flag is off, and that the scalping path can never
reach ``order_send`` on its own.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gold_trader.config import Config
from gold_trader.main import TradingBot
from gold_trader.main import main as cli_main
from gold_trader.models import MarketState, OrderType, SymbolSpec
from gold_trader.mt5.market_data import TickData
from tests._helpers import make_cfg, make_scalp_frame, scalp_cfg

SPEC = SymbolSpec(
    name="XAUUSD",
    point=0.01,
    digits=2,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    stops_level=20,
    freeze_level=10,
    visible=True,
    trade_mode=4,
    contract_size=100.0,
    trade_tick_size=0.01,
    trade_tick_value=1.0,
    filling_mode=3,
)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_everything_new_is_off_by_default(self):
        cfg = Config()
        assert cfg.scalping_enabled is False
        assert cfg.active_strategy == "signals"
        assert cfg.scalping_disable_management is True
        assert cfg.backtest_commission_per_lot == 0.0
        assert cfg.backtest_slippage_per_side == 0.0
        assert cfg.backtest_spread_stress_multiplier == 1.0
        assert cfg.backtest_spread_cost == pytest.approx(0.15)

    def test_scalping_has_its_own_smaller_risk(self):
        cfg = Config()
        assert cfg.scalping_risk_per_trade == pytest.approx(0.0015)
        assert cfg.scalping_risk_per_trade < cfg.risk_per_trade
        assert cfg.min_profit_to_spread_ratio == pytest.approx(3.0)
        assert cfg.scalping_timeframe == "M5"
        assert cfg.scalp_max_trades_per_day == 6

    def test_old_strategy_settings_are_untouched(self):
        cfg = Config()
        assert cfg.timeframe == "M15"
        assert cfg.rsi_period == 14
        assert cfg.ema_fast == 20
        assert cfg.risk_per_trade == pytest.approx(0.005)
        assert cfg.break_even_enabled is True
        assert cfg.partial_close_enabled is True


class TestConfigEnv:
    def test_scalping_switches_are_env_overridable(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_STRATEGY", "scalping")
        monkeypatch.setenv("SCALPING_ENABLED", "true")
        monkeypatch.setenv("MIN_PROFIT_TO_SPREAD_RATIO", "4.5")
        monkeypatch.setenv("SCALPING_RISK_PER_TRADE", "0.001")
        monkeypatch.setenv("SCALP_MAX_TRADES_PER_DAY", "3")
        monkeypatch.setenv("SCALPING_TIMEFRAME", "m1")
        cfg = Config.from_env()
        assert cfg.active_strategy == "scalping"
        assert cfg.scalping_enabled is True
        assert cfg.min_profit_to_spread_ratio == pytest.approx(4.5)
        assert cfg.scalping_risk_per_trade == pytest.approx(0.001)
        assert cfg.scalp_max_trades_per_day == 3
        assert cfg.scalping_timeframe == "M1"

    def test_short_scalp_env_aliases_are_honoured(self, monkeypatch):
        """``SCALP_TIMEFRAME``/``SCALP_*_ATR_MULT`` are accepted spellings."""
        for key in (
            "SCALPING_TIMEFRAME", "SCALP_TIMEFRAME",
            "SCALP_SL_ATR_MULTIPLE", "SCALP_SL_ATR_MULT",
            "SCALP_TP_ATR_MULTIPLE", "SCALP_TP_ATR_MULT",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("SCALP_TIMEFRAME", "m1")
        monkeypatch.setenv("SCALP_SL_ATR_MULT", "1.75")
        monkeypatch.setenv("SCALP_TP_ATR_MULT", "2.25")
        cfg = Config.from_env()
        assert cfg.scalping_timeframe == "M1"
        assert cfg.scalp_sl_atr_multiple == pytest.approx(1.75)
        assert cfg.scalp_tp_atr_multiple == pytest.approx(2.25)

    def test_long_env_name_wins_over_its_alias(self, monkeypatch):
        monkeypatch.setenv("SCALP_TIMEFRAME", "m1")
        monkeypatch.setenv("SCALPING_TIMEFRAME", "M30")
        monkeypatch.setenv("SCALP_TP_ATR_MULT", "2.25")
        monkeypatch.setenv("SCALP_TP_ATR_MULTIPLE", "1.0")
        cfg = Config.from_env()
        assert cfg.scalping_timeframe == "M30"
        assert cfg.scalp_tp_atr_multiple == pytest.approx(1.0)

    def test_backtest_cost_knobs_are_env_overridable(self, monkeypatch):
        monkeypatch.setenv("BACKTEST_COMMISSION_PER_LOT", "3.5")
        monkeypatch.setenv("BACKTEST_SLIPPAGE_PER_SIDE", "0.05")
        monkeypatch.setenv("BACKTEST_SPREAD_STRESS_MULTIPLIER", "1.5")
        cfg = Config.from_env()
        assert cfg.backtest_commission_per_lot == pytest.approx(3.5)
        assert cfg.backtest_slippage_per_side == pytest.approx(0.05)
        assert cfg.backtest_spread_stress_multiplier == pytest.approx(1.5)

    def test_absent_env_keeps_the_safe_defaults(self, monkeypatch):
        for key in ("SCALPING_ENABLED", "ACTIVE_STRATEGY", "MIN_PROFIT_TO_SPREAD_RATIO"):
            monkeypatch.delenv(key, raising=False)
        cfg = Config.from_env()
        assert cfg.scalping_enabled is False
        assert cfg.active_strategy == "signals"

    def test_safety_switches_are_still_env_only_and_off(self, monkeypatch):
        monkeypatch.delenv("TRADING_ENABLED", raising=False)
        monkeypatch.delenv("DRY_RUN", raising=False)
        cfg = Config.from_env()
        assert cfg.trading_enabled is False
        assert cfg.dry_run is True


# ---------------------------------------------------------------------------
# bot wiring
# ---------------------------------------------------------------------------

def _bot(cfg: Config) -> TradingBot:
    bot = object.__new__(TradingBot)
    bot.cfg = cfg
    bot.conn = SimpleNamespace(is_connected=lambda: True)
    bot.log = MagicMock()
    bot.trades_log = MagicMock()
    bot.error_log = MagicMock()
    bot.spec = SPEC
    bot._last_candle_time = None
    bot._recent_signals = []
    bot._install_execution_gate = MagicMock()
    bot._install_switches = MagicMock()
    return bot


class TestBotStrategySelection:
    def test_scalping_needs_both_the_flag_and_the_master_switch(self):
        assert _bot(make_cfg())._scalping_active is False
        assert _bot(make_cfg(active_strategy="scalping"))._scalping_active is False
        assert _bot(make_cfg(scalping_enabled=True))._scalping_active is False
        assert (
            _scalp_bot()._scalping_active
            is True
        )

    def test_case_insensitive_selection(self):
        assert (
            _bot(make_cfg(active_strategy="SCALPING", scalping_enabled=True))._scalping_active
            is True
        )

    def test_timeframe_follows_the_selected_engine(self):
        assert _bot(make_cfg())._effective_timeframe == "M15"
        assert (
            _bot(
                make_cfg(
                    active_strategy="scalping",
                    scalping_enabled=True,
                    scalping_timeframe="M5",
                )
            )._effective_timeframe
            == "M5"
        )

    def test_minimum_history_scales_with_the_engine(self):
        assert _bot(make_cfg())._min_candles() == make_cfg().min_candles_for_signal
        bot = _scalp_bot()
        assert bot._min_candles() == bot.cfg.scalp_min_candles_for_signal

    def test_load_data_requests_the_scalping_timeframe(self):
        bot = _bot(make_cfg(active_strategy="scalping", scalping_enabled=True, candles_count=10_000))
        frame = make_scalp_frame(n_flat=400)
        with patch("gold_trader.main.market_data.get_candles", return_value=frame) as get_candles, patch(
            "gold_trader.main.market_data.drop_unclosed_candle", return_value=frame
        ):
            result = bot._load_data()
        assert result is not None
        assert get_candles.call_args[0][1] == "M5"

    def test_load_data_refuses_a_frame_shorter_than_the_engine_needs(self):
        bot = _scalp_bot()
        short = make_scalp_frame(n_flat=10)
        with patch("gold_trader.main.market_data.get_candles", return_value=short), patch(
            "gold_trader.main.market_data.drop_unclosed_candle", return_value=short
        ):
            assert bot._load_data() is None


def _scalp_bot(**overrides) -> TradingBot:
    """Bot wired for the scalping engine with fixture-friendly gates."""
    return _bot(scalp_cfg(**overrides))


class TestScalpingEntryPath:
    def _state(self, spread: float = 0.05) -> MarketState:
        return MarketState(
            connected=True,
            symbol_valid=True,
            market_open=True,
            server_trading_allowed=True,
            open_position_count=0,
            pending_order_count=0,
            daily_pnl=0.0,
            account_balance=10_000.0,
            account_equity=10_000.0,
            now=__import__("datetime").datetime(2026, 1, 5, 12, 0, tzinfo=__import__("datetime").timezone.utc),
            terminal_trade_allowed=True,
            account_trade_allowed=True,
            expert_trade_allowed=True,
            spread=spread,
            daily_pnl_known=True,
        )

    def test_cycle_dispatches_to_the_scalping_path_and_skips_the_old_one(self):
        bot = _scalp_bot()
        frame = make_scalp_frame()
        state = self._state()
        bot._build_market_state = MagicMock(return_value=state)
        bot._evaluate_entry = MagicMock()
        bot._evaluate_scalping_entry = MagicMock()
        bot._scalp_entries_today = MagicMock(return_value=0)
        bot._latest_atr = MagicMock(return_value=0.5)
        tick = TickData(bid=2000.0, ask=2000.05, last=2000.0, time=state.now)
        with (
            patch("gold_trader.main.mt5_positions.get_positions", return_value=[]),
            patch("gold_trader.main.mt5_orders.get_pending_orders", return_value=[]),
            patch("gold_trader.main.market_data.get_tick", return_value=tick),
            patch("gold_trader.main.manage_break_even", return_value=[]),
            patch("gold_trader.main.manage_partial_close", return_value=[]),
            patch("gold_trader.main.manage_trailing_stop", return_value=[]),
            patch("gold_trader.main.plan_cleanup", return_value=[]),
            patch.object(TradingBot, "_load_data", return_value=frame),
        ):
            bot.cycle()
        assert bot._evaluate_scalping_entry.call_count == 1
        assert bot._evaluate_entry.call_count == 0

    def test_dry_run_logs_the_would_execute_scalp_and_sends_nothing(self):
        bot = _scalp_bot()
        frame = make_scalp_frame()
        state = self._state()
        bot._scalp_entries_today = MagicMock(return_value=0)
        tick = TickData(bid=2000.0, ask=2000.05, last=2000.0, time=state.now)
        with patch("gold_trader.main.mt5_orders.send_plan") as send_plan:
            bot._evaluate_scalping_entry(frame, state, tick, [], [])
        send_plan.assert_not_called()
        message = bot.trades_log.info.call_args[0][0]
        assert "DRY RUN (SCALP)" in message
        logged = bot.trades_log.info.call_args[0]
        assert OrderType.BUY.value in logged
        assert logged[3] == pytest.approx(tick.ask)  # entry priced at the ask

    def test_the_plan_carries_the_scalp_marker_and_the_small_risk(self):
        bot = _scalp_bot()
        frame = make_scalp_frame()
        state = self._state()
        bot._scalp_entries_today = MagicMock(return_value=0)
        tick = TickData(bid=2000.0, ask=2000.05, last=2000.0, time=state.now)
        captured = {}

        def fake_check_trade(plan, market_state, cfg, spec, positions=(), pendings=()):
            captured["plan"] = plan
            captured["risk"] = cfg.scalping_risk_per_trade
            captured["equity_risk"] = market_state.account_equity * cfg.scalping_risk_per_trade
            from gold_trader.models import RiskDecision

            return RiskDecision(allowed=False, would_trade=False, reasons=["TRADING_ENABLED=false"], gate_failures=[])

        with patch("gold_trader.main.check_trade", side_effect=fake_check_trade):
            bot._evaluate_scalping_entry(frame, state, tick, [], [])
        plan = captured["plan"]
        assert "SCALP" in plan.comment
        assert captured["risk"] == pytest.approx(0.0015)
        assert captured["equity_risk"] == pytest.approx(15.0)
        assert plan.tp > plan.entry > plan.sl
        # the plan must clear the mandatory cost gate at the real quoted spread
        assert plan.tp - plan.entry >= 3.0 * (tick.spread + bot.cfg.scalp_spread_safety_margin) - 1e-9

    def test_a_wide_live_spread_never_reaches_the_gate(self):
        bot = _scalp_bot(scalp_max_spread=0.20)
        frame = make_scalp_frame()
        state = self._state()
        bot._scalp_entries_today = MagicMock(return_value=0)
        tick = TickData(bid=2000.0, ask=2000.30, last=2000.0, time=state.now)
        with patch("gold_trader.main.check_trade") as check_trade:
            bot._evaluate_scalping_entry(frame, state, tick, [], [])
        check_trade.assert_not_called()

    def test_daily_cap_unknown_blocks_the_entry(self):
        bot = _scalp_bot()
        state = self._state()
        bot._scalp_entries_today = MagicMock(return_value=None)
        tick = TickData(bid=2000.0, ask=2000.05, last=2000.0, time=state.now)
        with patch("gold_trader.main.check_trade") as check_trade:
            bot._evaluate_scalping_entry(make_scalp_frame(), state, tick, [], [])
        check_trade.assert_not_called()
        assert any("daily entry count" in str(call) or "NO TRADE (scalping)" in str(call) for call in bot.log.warning.call_args_list)

    def test_missing_tick_is_a_clean_no_trade(self):
        bot = _scalp_bot()
        bot._scalp_entries_today = MagicMock(return_value=0)
        bot._evaluate_scalping_entry(make_scalp_frame(), self._state(), None, [], [])
        assert any("no tick" in str(call) for call in bot.log.warning.call_args_list)

    def test_no_signal_produces_no_plan(self):
        bot = _scalp_bot()
        bot._scalp_entries_today = MagicMock(return_value=0)
        from tests._helpers import build_df

        frame = build_df([(2000.0, 2000.05, 1999.95, 2000.0)] * 200, freq="5min")
        tick = TickData(bid=2000.0, ask=2000.05, last=2000.0, time=self._state().now)
        with patch("gold_trader.main.check_trade") as check_trade:
            bot._evaluate_scalping_entry(frame, self._state(), tick, [], [])
        check_trade.assert_not_called()

    def test_scalp_entries_today_is_fail_closed_on_history_error(self):
        bot = _scalp_bot()
        import gold_trader.mt5.connection as conn_module

        api = MagicMock()
        api.history_deals_get.return_value = None
        api.last_error.return_value = (1, "no history")
        with patch.object(conn_module, "MT5_AVAILABLE", True), patch.object(
            conn_module, "_mt5", api
        ), patch("gold_trader.main.require_mt5", return_value=api):
            assert bot._scalp_entries_today() is None

    def test_scalp_entries_today_counts_matching_deals(self):
        bot = _scalp_bot()
        deals = [SimpleNamespace(magic=123456789, entry=0, comment="GB|SCALP")]
        api = MagicMock()
        api.history_deals_get.return_value = deals
        with patch("gold_trader.main.require_mt5", return_value=api):
            assert bot._scalp_entries_today() == 1


class TestManagementFilterInCycle:
    def test_break_even_for_a_scalp_is_dropped_before_it_is_applied(self):
        bot = _bot(make_cfg(active_strategy="scalping", scalping_enabled=True, break_even_enabled=True))
        scalp = SimpleNamespace(
            ticket=11,
            symbol="XAUUSD",
            is_buy=True,
            volume=0.1,
            price_open=2000.0,
            sl=1999.0,
            tp=2001.0,
            profit=0.5,
            magic=123456789,
            comment="GB|sl=1999.00|vol=0.10|SCALP",
            open_time=None,
        )
        action = SimpleNamespace(kind="move_sl", ticket=11, description="break-even", new_sl=2000.1, new_tp=2001.0, close_volume=None)
        bot.conn = SimpleNamespace(is_connected=lambda: True)
        bot._build_market_state = MagicMock(return_value=self_state())
        bot._load_data = MagicMock(return_value=make_scalp_frame())
        bot._latest_atr = MagicMock(return_value=0.5)
        bot._evaluate_entry = MagicMock()
        bot._evaluate_scalping_entry = MagicMock()
        bot._apply_actions = MagicMock()
        tick = TickData(bid=2001.0, ask=2001.05, last=2001.0, time=self_state().now)
        with (
            patch("gold_trader.main.mt5_positions.get_positions", return_value=[scalp]),
            patch("gold_trader.main.mt5_orders.get_pending_orders", return_value=[]),
            patch("gold_trader.main.market_data.get_tick", return_value=tick),
            patch("gold_trader.main.manage_break_even", return_value=[action]),
            patch("gold_trader.main.manage_partial_close", return_value=[]),
            patch("gold_trader.main.manage_trailing_stop", return_value=[]),
            patch("gold_trader.main.plan_cleanup", return_value=[]),
        ):
            bot.cycle()
        applied = [call[0][0] for call in bot._apply_actions.call_args_list]
        assert applied == [] or all(a for a in applied)
        bot._apply_actions.assert_not_called()


def self_state() -> MarketState:
    import datetime as dt

    return MarketState(
        connected=True,
        symbol_valid=True,
        market_open=True,
        server_trading_allowed=True,
        open_position_count=1,
        pending_order_count=0,
        daily_pnl=0.0,
        account_balance=10_000.0,
        account_equity=10_000.0,
        now=dt.datetime(2026, 1, 5, 12, 0, tzinfo=dt.timezone.utc),
        terminal_trade_allowed=True,
        account_trade_allowed=True,
        expert_trade_allowed=True,
        spread=0.05,
        daily_pnl_known=True,
    )


class TestCliSurface:
    def test_backtest_scalping_flag_is_accepted(self):
        parser_args = ["--backtest-scalping"]
        with patch("gold_trader.main.MT5_AVAILABLE", True), patch("gold_trader.main.TradingBot") as bot_cls:
            bot_cls.return_value.run_backtest.return_value = 0
            bot_cls.return_value.check_connection.return_value = 0
            rc = cli_main(parser_args)
        assert rc == 0
        assert bot_cls.return_value.run_backtest.call_args.kwargs == {"scalping": True}

    def test_scalping_backtest_never_touches_the_execution_gate(self):
        """Research output must not be able to place an order, ever."""
        import gold_trader.backtest.scalping_engine as engine_module

        bot = _scalp_bot()
        frame = make_scalp_frame(n_flat=120)
        with patch.object(engine_module.ScalpingBacktestEngine, "run") as run_mock:
            run_mock.return_value = SimpleNamespace(
                metrics=SimpleNamespace(
                    total_trades=0,
                    wins=0,
                    losses=0,
                    win_rate=0.0,
                    gross_profit=0.0,
                    gross_loss=0.0,
                    net_profit=0.0,
                    max_drawdown=0.0,
                    max_drawdown_pct=0.0,
                    profit_factor=0.0,
                    average_win=0.0,
                    average_loss=0.0,
                    final_equity=10_000.0,
                ),
                trades=[],
                initial_balance=10_000.0,
                final_equity=10_000.0,
                total_spread_cost=0.0,
                total_commission_cost=0.0,
                veto_counts={},
                cost_model="fixed",
            )
            with patch("gold_trader.main.mt5_orders.send_plan") as send_plan:
                code = bot._run_scalping_backtest(frame, SPEC)
        assert code == 0
        send_plan.assert_not_called()
