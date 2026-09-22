"""Entry point for the gold trading bot.

Usage (from the repository root, inside the virtual environment):

    python -m gold_trader.main              # main loop (DRY_RUN by default)
    python -m gold_trader.main --once       # run a single cycle and exit
    python -m gold_trader.main --check      # verify connection + symbol
    python -m gold_trader.main --backtest   # backtest on MT5 history

Safety model:
* ``TRADING_ENABLED=false`` and ``DRY_RUN=true`` are the defaults;
* no real order is sent unless BOTH are changed manually in the
  environment (see .env.example);
* those switches are installed as an explicit execution permission and
  enforced again inside ``send_request``, so management actions (SL/TP,
  close, partial close, delete) cannot reach ``order_send`` either;
* Demo safety is a separate broker proof. A caller cannot declare
  ``account_is_demo=True``. After any reconnect the proof is dropped until
  the current account is verified again;
* if today's closed P/L cannot be read, new entries are blocked regardless
  of floating P/L (``daily_pnl_known=False``);
* the account must be a Demo account;
* a live run prints a loud warning when it is about to send real orders.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import pandas as pd

from .config import Config
from .models import (
    ManagementAction,
    MarketState,
    OrderType,
    PendingOrderInfo,
    PositionInfo,
    Signal,
    SymbolSpec,
    TradePlan,
)
from .mt5 import market_data, orders as mt5_orders
from .mt5 import positions as mt5_positions
from .mt5.execution_gate import (
    ExecutionPermission,
    clear_verified_account_safety,
    install_execution_permission,
    refresh_verified_account_safety,
)
from .mt5 import symbols as mt5_symbols
from .mt5._constants import const
from .mt5.connection import (
    MT5_AVAILABLE,
    MT5Connection,
    MT5ConnectionError,
    MT5Error,
    require_mt5,
)
from .risk.position_size import calculate_position_size
from .risk.risk_manager import check_trade
from .strategy import signals as strategy_signals
from .strategy.indicators import atr as atr_indicator
from .strategy.indicators import latest_valid
from .strategy.levels import build_sl_tp
from .trade_management.break_even import (
    format_position_comment,
    manage_break_even,
)
from .trade_management.partial_close import manage_partial_close
from .trade_management.pending_orders import plan_cleanup
from .trade_management.reconciliation import resolve_management_actions
from .trade_management.trailing_stop import manage_trailing_stop
from .utils.logger import get_errors_logger, get_logger, get_trades_logger, setup_logging
from .utils.time_utils import utcnow


class TradingBot:
    """Wires connection, data, strategy, risk and management together."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.log = get_logger("gold_trader.bot")
        self.trades_log = get_trades_logger()
        self.error_log = get_errors_logger()
        env_login = os.getenv("MT5_LOGIN", "").strip()
        self.conn = MT5Connection(
            login=int(env_login) if env_login else None,
            password=os.getenv("MT5_PASSWORD") or None,
            server=os.getenv("MT5_SERVER") or None,
            terminal_path=os.getenv("MT5_TERMINAL_PATH") or None,
        )
        self.spec: Optional[SymbolSpec] = None
        self._last_candle_time: Optional[pd.Timestamp] = None
        self._recent_signals: Deque[Tuple[pd.Timestamp, str]] = deque(maxlen=20)
        # Install before any cycle. orders.py does not read .env itself.
        self._install_execution_gate()

    def _install_execution_gate(self) -> None:
        """Refresh the broker proof, then install this bot's switches.

        Re-installed at the start of each cycle and before each management
        path so a previously widened permission cannot outlive this config.
        Demo status comes only from a fresh ``account_info`` read inside the
        gate — never from ``.env`` and never from a caller-supplied
        ``account_is_demo=True``. Unknown is not Demo.
        """
        self._account_is_demo()
        self._install_switches()

    def _install_switches(self) -> None:
        """Install configuration switches without reading or opening Demo."""
        install_execution_permission(
            ExecutionPermission(
                trading_enabled=self.cfg.trading_enabled,
                dry_run=self.cfg.dry_run,
            )
        )

    def _close_execution_until_verified(self) -> None:
        """Drop the Demo proof. Switches stay; no send is allowed until a new proof."""
        clear_verified_account_safety()
        self._install_switches()

    def _account_is_demo(self) -> bool:
        """True only when a fresh terminal read proves ``ACCOUNT_TRADE_MODE_DEMO``.

        This is the bot's explicit verification. It does not accept a caller
        declaration, and it does not read the environment. REAL, contest, a
        missing constant, and any failure latch False.
        """
        return refresh_verified_account_safety()

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Connect, discover the gold symbol, validate the account mode."""
        self._connect_and_discover()
        self._log_account_mode()
        self._enforce_demo_safety()
        self.log.info(
            "Bot ready. dry_run=%s trading_enabled=%s timeframe=%s interval=%.0fs",
            self.cfg.dry_run,
            self.cfg.trading_enabled,
            self.cfg.timeframe,
            self.cfg.loop_interval_seconds,
        )

    def _connect_and_discover(self) -> None:
        self.conn.initialize()
        self.spec = mt5_symbols.find_gold_symbol(
            preferred=self.cfg.symbol,
            candidates=self.cfg.gold_symbol_candidates,
        )
        mt5_api = require_mt5()
        try:
            mt5_api.symbol_select(self.spec.name, True)
        except Exception as exc:
            self.log.warning("symbol_select(%s) raised: %s", self.spec.name, exc)

    def shutdown(self) -> None:
        self.conn.shutdown()
        self.log.info("MT5 connection shut down cleanly")

    def _log_account_mode(self) -> None:
        try:
            mode = self.conn.account_mode()
        except MT5Error as exc:
            self.log.warning("could not determine account mode: %s", exc)
            return
        self.log.info("Account mode: %s", mode.value)
        if self.cfg.expected_account_mode and (
            mode.value != self.cfg.expected_account_mode.upper()
        ):
            raise MT5ConnectionError(
                f"Account mode {mode.value} != expected "
                f"{self.cfg.expected_account_mode} - aborting (NETTING and "
                "HEDGING positions must not be handled the same way)"
            )

    def _enforce_demo_safety(self) -> None:
        """Hard safety gate: REAL accounts must never receive live orders.

        If TRADING_ENABLED=true and DRY_RUN=false, the account MUST be
        a Demo account (ACCOUNT_TRADE_MODE_DEMO = 0). Otherwise abort.
        """
        if not (self.cfg.trading_enabled and not self.cfg.dry_run):
            return
        try:
            info = self.conn.account_info()
            trade_mode = info.get("trade_mode")
            # Official MT5: 0=DEMO, 1=CONTEST, 2=REAL
            # Some wrappers document 0=REAL,1=DEMO,2=CONTEST - we handle both
            # by reading the constant names if available.
            mt5_api = require_mt5()
            demo_const = getattr(mt5_api, "ACCOUNT_TRADE_MODE_DEMO", 0)
            real_const = getattr(mt5_api, "ACCOUNT_TRADE_MODE_REAL", 2)
            # If trade_mode equals REAL, block.
            if trade_mode == real_const:
                raise MT5ConnectionError(
                    f"CRITICAL SAFETY: Trading is enabled (TRADING_ENABLED=true, "
                    f"DRY_RUN=false) but the connected account is REAL "
                    f"(trade_mode={trade_mode}). Aborting to prevent live trading. "
                    f"Use a Demo account."
                )
            # If we have explicit DEMO constant, require DEMO when live trading
            if demo_const is not None and trade_mode != demo_const:
                # Allow CONTEST as well? For safety, only DEMO is allowed for live.
                # If it's CONTEST, still block unless explicitly allowed.
                # Here we block anything that is not DEMO when live trading is on.
                if trade_mode != demo_const:
                    # Check if it's contest - still block for safety
                    self.log.warning(
                        "Live trading enabled but account trade_mode=%s != DEMO (%s). "
                        "Blocking as safety measure.",
                        trade_mode,
                        demo_const,
                    )
                    raise MT5ConnectionError(
                        f"CRITICAL SAFETY: Live trading requires a Demo account, "
                        f"but trade_mode={trade_mode} != DEMO ({demo_const}). Aborting."
                    )
        except MT5Error:
            raise
        except Exception as exc:
            self.log.warning("could not verify account trade mode for safety: %s", exc)
            # Fail safe: if we cannot determine, block live trading
            if self.cfg.trading_enabled and not self.cfg.dry_run:
                raise MT5ConnectionError(
                    f"Could not verify Demo account status ({exc}) - blocking live trading"
                ) from exc

    # -- data helpers ------------------------------------------------------

    def _load_data(self) -> Optional[pd.DataFrame]:
        """Closed candles for the configured timeframe (None on failure)."""
        try:
            df = market_data.get_candles(
                self.spec.name, self.cfg.timeframe, self.cfg.candles_count
            )
            df = market_data.drop_unclosed_candle(df, self.cfg.timeframe, utcnow())
            return df if len(df) >= 2 else None
        except MT5Error as exc:
            self.log.error("market data failed: %s", exc)
            return None

    def _latest_atr(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < self.cfg.atr_period + 2:
            return None
        return latest_valid(
            atr_indicator(
                df["high"], df["low"], df["close"], self.cfg.atr_period
            )
        )

    def _daily_closed_pnl(self) -> Optional[float]:
        """Realized P/L of today's bot deals (profit + commission + swap).

        Returns None when history is unavailable so the caller can fail-safe.
        """
        mt5_api = require_mt5()
        try:
            day_start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            deals = mt5_api.history_deals_get(
                int(day_start.timestamp()), int(time.time()), group=self.spec.name
            )
            if deals is None:
                raise MT5Error(f"history_deals_get failed: {mt5_api.last_error()}")
            return sum(
                float(d.profit) + float(d.commission) + float(d.swap)
                for d in deals
                if int(d.magic) == self.cfg.magic_number
            )
        except MT5Error as exc:
            self.log.error(
                "daily P/L unavailable (%s) - will block new entries as fail-safe", exc
            )
            return None

    def _build_market_state(
        self, positions: List[PositionInfo], pendings: List[PendingOrderInfo]
    ) -> MarketState:
        account = self.conn.account_info()
        terminal = self.conn.terminal_info()
        spread = 0.0
        market_open = False
        try:
            tick = market_data.get_tick(self.spec.name)
            spread = tick.spread
            market_open = market_data.is_tick_usable(tick)
        except MT5Error as exc:
            self.log.warning("tick unavailable: %s", exc)
        closed_pnl = self._daily_closed_pnl()
        floating = sum(p.profit for p in positions)
        if closed_pnl is None:
            # Fail closed on an explicit flag. Do not invent a numeric
            # sentinel: adding floating P/L (e.g. +600) used to cancel
            # ``-max_daily_loss - 1`` and reopen the daily-loss gate.
            daily_pnl_known = False
            daily_pnl = 0.0
            self.log.warning(
                "daily closed P/L unknown (history unavailable) - new entries "
                "blocked regardless of floating P/L (floating=%.2f)",
                floating,
            )
        else:
            daily_pnl_known = True
            daily_pnl = closed_pnl + floating
        # SYMBOL_TRADE_MODE_FULL = 4 (official MT5)
        trade_mode_full = const("SYMBOL_TRADE_MODE_FULL", 4)
        terminal_allowed = bool(terminal.get("trade_allowed"))
        account_allowed = bool(account.get("trade_allowed"))
        expert_allowed = bool(account.get("trade_expert"))
        all_trading_allowed = terminal_allowed and account_allowed and expert_allowed
        return MarketState(
            connected=self.conn.is_connected(),
            symbol_valid=self.spec.visible and self.spec.trade_mode == trade_mode_full,
            market_open=market_open,
            server_trading_allowed=all_trading_allowed,
            terminal_trade_allowed=terminal_allowed,
            account_trade_allowed=account_allowed,
            expert_trade_allowed=expert_allowed,
            spread=spread,
            open_position_count=len(positions),
            pending_order_count=len(pendings),
            daily_pnl=daily_pnl,
            account_balance=float(account.get("balance", 0.0)),
            account_equity=float(account.get("equity", 0.0)),
            now=utcnow(),
            daily_pnl_known=daily_pnl_known,
        )

    # -- order result logging -----------------------------------------------

    def _log_order_result(self, description: str, result) -> None:
        if getattr(result, "blocked_by_safety", False):
            self.trades_log.info(
                "WOULD EXECUTE | %s | %s", description, result.summary()
            )
            return
        if result.success:
            self.trades_log.info("%s | %s", description, result.summary())
        else:
            self.error_log.error("%s FAILED | %s", description, result.summary())

    def _log_trade_executed(self, plan: TradePlan, result) -> None:
        if getattr(result, "blocked_by_safety", False):
            self.trades_log.info(
                "WOULD EXECUTE | %s %s | entry=%.5f | sl=%.5f | tp=%.5f | "
                "volume=%.2f | %s",
                plan.order_type.value,
                plan.symbol,
                plan.entry,
                plan.sl,
                plan.tp,
                plan.volume or 0.0,
                result.comment,
            )
            return
        self.trades_log.info(
            "EXECUTED | ts=%s | signal=%s | order_type=%s | symbol=%s | "
            "entry=%.5f | sl=%.5f | tp=%.5f | volume=%.2f | risk=%.2f | "
            "ticket=%s | retcode=%s | result=%s",
            utcnow().isoformat(),
            plan.order_type.value,
            plan.order_type.value,
            plan.symbol,
            plan.entry,
            plan.sl,
            plan.tp,
            plan.volume or 0.0,
            plan.risk_amount,
            result.order,
            result.retcode,
            "success" if result.success else "FAILED",
        )
        if not result.success:
            self.error_log.error("order failed | %s", result.summary())

    # -- management ------------------------------------------------------------

    def _apply_actions(
        self,
        actions: List[ManagementAction],
        positions: List[PositionInfo],
        pendings: List[PendingOrderInfo],
        tick=None,
    ) -> None:
        # Re-assert this bot's switches before any management send. The
        # choke point remains send_request; this stops a widened permission
        # from outliving the config that owns these actions.
        self._install_execution_gate()
        pos_by_ticket = {p.ticket: p for p in positions}
        for action in actions:
            try:
                if action.kind == "move_sl":
                    position = pos_by_ticket.get(action.ticket)
                    if position is None:
                        continue
                    target_sl = action.new_sl if action.new_sl is not None else position.sl
                    target_tp = action.new_tp if action.new_tp is not None else position.tp

                    # Safety invariant: never reduce protection
                    if position.is_buy and position.sl > 0 and target_sl < position.sl - 1e-9:
                        self.log.warning(
                            "rejected weaker SL for BUY #%s: %.5f < current %.5f",
                            position.ticket,
                            target_sl,
                            position.sl,
                        )
                        continue
                    if not position.is_buy and position.sl > 0 and target_sl > position.sl + 1e-9:
                        self.log.warning(
                            "rejected weaker SL for SELL #%s: %.5f > current %.5f",
                            position.ticket,
                            target_sl,
                            position.sl,
                        )
                        continue

                    result = mt5_positions.modify_position_sltp(
                        position,
                        target_sl,
                        target_tp,
                        self.spec,
                        tick=tick,
                    )
                    self._log_order_result(action.description, result)
                    if result.success:
                        pos_by_ticket[position.ticket] = PositionInfo(
                            ticket=position.ticket,
                            symbol=position.symbol,
                            is_buy=position.is_buy,
                            volume=position.volume,
                            price_open=position.price_open,
                            sl=target_sl,
                            tp=target_tp,
                            profit=position.profit,
                            magic=position.magic,
                            comment=position.comment,
                            open_time=position.open_time,
                        )
                elif action.kind in ("partial_close", "full_close"):
                    position = pos_by_ticket.get(action.ticket)
                    if position is None:
                        continue
                    result = mt5_positions.close_position(
                        position,
                        self.spec,
                        volume=action.close_volume,
                        magic=self.cfg.magic_number,
                        deviation=self.cfg.max_deviation,
                        comment=f"goldbot {action.kind}",
                        tick=tick,
                    )
                    self._log_order_result(action.description, result)
                elif action.kind == "delete_order":
                    result = mt5_orders.delete_order(action.ticket)
                    self._log_order_result(action.description, result)
            except Exception as exc:
                self.error_log.exception(
                    "management action failed for ticket %s: %s", action.ticket, exc
                )

    # -- main cycle ---------------------------------------------------------------

    def cycle(self) -> None:
        """One full loop iteration (data -> manage -> signal -> execute)."""
        self._install_execution_gate()
        if not self.conn.is_connected():
            raise MT5ConnectionError("not connected to MT5 terminal")

        positions = mt5_positions.get_positions(
            symbol=self.spec.name, magic=self.cfg.magic_number
        )
        pendings = mt5_orders.get_pending_orders(
            magic=self.cfg.magic_number, symbol=self.spec.name
        )
        state = self._build_market_state(positions, pendings)

        df = self._load_data()
        if df is None:
            self.log.warning("cycle skipped: no market data")
            return
        atr_value = self._latest_atr(df)

        tick = None
        try:
            tick = market_data.get_tick(self.spec.name)
        except MT5Error as exc:
            self.log.warning("tick unavailable: %s", exc)

        # 1) manage existing bot positions (break-even, partial close, trailing)
        actions: list[ManagementAction] = []
        if tick is not None:
            actions += manage_break_even(
                positions, tick.bid, tick.ask, self.cfg, self.spec
            )
            actions += manage_partial_close(
                positions, tick.bid, tick.ask, self.cfg, self.spec
            )
            actions += manage_trailing_stop(
                positions, tick.bid, tick.ask, atr_value, self.cfg, self.spec
            )
        if actions:
            actions = resolve_management_actions(actions, positions)
            self.log.info(
                "position management: %d action(s): %s",
                len(actions),
                "; ".join(a.description for a in actions),
            )
            self._apply_actions(actions, positions, pendings, tick=tick)

        # 2) pending order cleanup (expired orders)
        cleanup = plan_cleanup(pendings, utcnow())
        if cleanup:
            self._apply_actions(cleanup, positions, pendings, tick=tick)

        # 3) new entry -- only on a NEW closed candle (duplicate protection)
        last_time = df["time"].iloc[-1]
        if self._last_candle_time == last_time:
            self.log.info("cycle complete: no new closed candle yet (last=%s)", last_time)
            return
        self._last_candle_time = last_time
        self._evaluate_entry(df, state, tick, positions, pendings, atr_value)

    def _evaluate_entry(
        self,
        df: pd.DataFrame,
        state: MarketState,
        tick,
        positions: List[PositionInfo],
        pendings: List[PendingOrderInfo],
        atr_value: Optional[float],
    ) -> None:
        self._install_execution_gate()
        signal = strategy_signals.generate_signal(df, self.cfg)
        last_time = df["time"].iloc[-1]
        self._recent_signals.append(
            (last_time, f"{signal.signal.value}: {signal.reason}")
        )
        self.log.info(
            "Signal @%s: %s - %s (recent: %s)",
            last_time,
            signal.signal.value,
            signal.reason,
            [f"{t} {s}" for t, s in list(self._recent_signals)[-3:]],
        )
        if signal.signal is Signal.NO_TRADE:
            return
        if tick is None:
            self.log.warning("no tick - cannot price the entry")
            return
        if atr_value is None or atr_value <= 0:
            self.log.warning("NO TRADE: ATR not available")
            return

        is_buy = signal.signal is Signal.BUY
        entry = tick.ask if is_buy else tick.bid
        sltp = build_sl_tp(signal.signal, entry, atr_value, df, self.cfg)
        sizing = calculate_position_size(
            state.account_equity, self.cfg.risk_per_trade, entry, sltp.sl, self.spec
        )
        if sizing.volume is None:
            self.log.warning("NO TRADE: position sizing failed: %s", sizing.reason)
            return

        comment = format_position_comment(
            self.cfg.position_comment_prefix,
            sltp.sl,
            sizing.volume,
            self.spec.digits,
            volume_step=self.spec.volume_step,
        )
        plan = TradePlan(
            symbol=self.spec.name,
            order_type=OrderType.BUY if is_buy else OrderType.SELL,
            entry=entry,
            sl=sltp.sl,
            tp=sltp.tp,
            volume=sizing.volume,
            risk_amount=sizing.risk_amount,
            comment=comment,
        )
        decision = check_trade(
            plan, state, self.cfg, self.spec, positions=positions, pendings=pendings
        )
        if decision.allowed:
            result = mt5_orders.send_plan(
                plan, self.spec, tick, self.cfg.magic_number, self.cfg.max_deviation
            )
            self._log_trade_executed(plan, result)
        elif self.cfg.dry_run and decision.would_trade:
            self.trades_log.info(
                "DRY RUN\nSIGNAL: %s\nSYMBOL: %s\nENTRY: %.5f\nSL: %.5f\n"
                "TP: %.5f\nVOLUME: %.2f\nRISK: %.2f\nWOULD EXECUTE: %s",
                plan.order_type.value,
                plan.symbol,
                plan.entry,
                plan.sl,
                plan.tp,
                plan.volume,
                plan.risk_amount,
                plan.order_type.value,
            )
        else:
            self.log.warning(
                "TRADE BLOCKED (%s): %s",
                plan.order_type.value,
                "; ".join(decision.reasons) or "unknown reason",
            )

    # -- run modes ---------------------------------------------------------------

    def run_forever(self) -> None:
        """Main loop: cycle + bounded sleep (no busy loop), graceful stop."""
        self.start()
        try:
            while True:
                started = time.monotonic()
                try:
                    self.cycle()
                except MT5ConnectionError as exc:
                    self.error_log.error("connection lost: %s - reconnecting", exc)
                    self._reconnect()
                except MT5Error as exc:
                    self.error_log.error("MT5 error in cycle: %s", exc)
                except Exception:
                    self.error_log.exception("unhandled error in cycle")
                elapsed = time.monotonic() - started
                sleep_for = max(1.0, self.cfg.loop_interval_seconds - elapsed)
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            self.log.info("interrupt received - shutting down")
        finally:
            self.shutdown()

    def _reconnect(self) -> None:
        """Reconnect, then verify the account that this session actually opened.

        The previous Demo proof is dropped before shutdown and stays dropped
        through the retry sleep. A send in that window cannot reuse it, and
        cannot open a new proof: only the refresh after a successful session
        may do that. A failed reconnect does not refresh, so execution remains
        closed until a later successful verification (the next cycle, or a
        later reconnect that completes).
        """
        self._close_execution_until_verified()
        session_opened = False
        try:
            self.conn.shutdown()
            self.conn.initialize()
            self.spec = mt5_symbols.find_gold_symbol(
                preferred=self.spec.name if self.spec else self.cfg.symbol,
                candidates=self.cfg.gold_symbol_candidates,
            )
            time.sleep(5.0)
            session_opened = True
        except MT5Error as exc:
            self.error_log.error("reconnect failed: %s - retrying next cycle", exc)
        finally:
            if session_opened:
                self._install_execution_gate()
            else:
                self._close_execution_until_verified()

    def run_once(self) -> int:
        """Single cycle (useful for smoke tests on a Demo terminal)."""
        self.start()
        try:
            self.cycle()
            return 0
        finally:
            self.shutdown()

    def check_connection(self) -> int:
        """--check: verify terminal, account and gold symbol, then exit."""
        exit_code = 0
        try:
            self.conn.initialize()
            terminal = self.conn.terminal_info()
            account = self.conn.account_info()
            print(
                "Terminal: "
                f"{terminal.get('name')} | company={terminal.get('company')} | "
                f"connected={terminal.get('connected')} | "
                f"trade_allowed={terminal.get('trade_allowed')}"
            )
            print(
                f"Account: login=****{str(account.get('login', ''))[-4:]} "
                f"server={account.get('server')} currency={account.get('currency')} "
                f"balance={account.get('balance')} equity={account.get('equity')} "
                f"leverage={account.get('leverage')}"
            )
            print(f"Account mode: {self.conn.account_mode().value}")
            spec = mt5_symbols.find_gold_symbol(
                preferred=self.cfg.symbol,
                candidates=self.cfg.gold_symbol_candidates,
            )
            print(
                f"Symbol: {spec.name} visible={spec.visible} trade_mode={spec.trade_mode} "
                f"stops_level={spec.stops_level} freeze_level={spec.freeze_level} "
                f"volume={spec.volume_min}-{spec.volume_max} step={spec.volume_step}"
            )
            tick = market_data.get_tick(spec.name)
            print(f"Tick: bid={tick.bid} ask={tick.ask} spread={tick.spread:.5f}")
        except MT5Error as exc:
            print(f"CHECK FAILED: {exc}")
            exit_code = 1
        finally:
            self.conn.shutdown()
        return exit_code

    def run_backtest(self) -> int:
        """--backtest: fetch history from MT5 and run the backtest engine."""
        from .backtest.engine import BacktestEngine
        from .backtest.metrics import format_metrics

        exit_code = 0
        try:
            self.conn.initialize()
            spec = mt5_symbols.find_gold_symbol(
                preferred=self.cfg.symbol,
                candidates=self.cfg.gold_symbol_candidates,
            )
            print(
                f"Fetching {self.cfg.backtest_candles} x {self.cfg.timeframe} "
                f"candles for {spec.name} ..."
            )
            df = market_data.get_candles(
                spec.name, self.cfg.timeframe, self.cfg.backtest_candles
            )
            df = market_data.drop_unclosed_candle(df, self.cfg.timeframe, utcnow())
            engine = BacktestEngine(self.cfg)
            result = engine.run(df, spec=spec)
            print(format_metrics(result.metrics, result.initial_balance))
            if result.trades:
                print("\nLast 10 trades:")
                for trade in result.trades[-10:]:
                    print(
                        f"  {trade.entry_time} {trade.side} "
                        f"entry={trade.entry:.5f} exit={trade.exit_price:.5f} "
                        f"vol={trade.volume:.2f} pnl={trade.pnl:.2f} "
                        f"({trade.reason})"
                    )
        except (MT5Error, ValueError) as exc:
            print(f"BACKTEST FAILED: {exc}")
            exit_code = 1
        finally:
            self.conn.shutdown()
        return exit_code


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gold auto-trading bot (MetaTrader 5, Demo account only)"
    )
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--check", action="store_true", help="verify MT5 connection and symbol, then exit")
    parser.add_argument("--backtest", action="store_true", help="run a backtest on MT5 history, then exit")
    args = parser.parse_args(argv)

    cfg = Config.from_env()
    setup_logging(level=cfg.log_level)
    log = get_logger("gold_trader.main")

    if not MT5_AVAILABLE:
        print(
            "ERROR: the MetaTrader5 package is not available on this platform. "
            "MT5 terminal automation requires Windows. You can still run the "
            "unit tests (python -m pytest) on this machine.",
            file=sys.stderr,
        )
        return 2

    if cfg.trading_enabled and not cfg.dry_run:
        log.warning(
            "TRADING_ENABLED=true and DRY_RUN=false - REAL orders will be sent "
            "(Demo account required!)"
        )

    bot = TradingBot(cfg)
    if args.check:
        return bot.check_connection()
    if args.backtest:
        return bot.run_backtest()
    if args.once:
        return bot.run_once()
    bot.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
