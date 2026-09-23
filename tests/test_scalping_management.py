"""Tests for the scalping management exception and the daily-cost controls."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gold_trader.models import ManagementAction, PositionInfo, SymbolSpec
from gold_trader.risk.scalping_limits import (
    DEAL_ENTRY_IN,
    count_scalp_entries,
    scalp_entry_allowed,
)
from gold_trader.trade_management.scalping_policy import (
    SCALP_ALLOWED_KINDS,
    filter_management_actions,
    format_scalp_comment,
    is_scalp_position,
    management_allowed,
    scalp_exits_only,
)
from tests._helpers import make_cfg


def _position(ticket: int, comment: str, is_buy: bool = True) -> PositionInfo:
    return PositionInfo(
        ticket=ticket,
        symbol="XAUUSD",
        is_buy=is_buy,
        volume=0.1,
        price_open=2000.0,
        sl=1999.0,
        tp=2001.0,
        profit=0.0,
        magic=123456789,
        comment=comment,
    )


def _action(kind: str, ticket: int) -> ManagementAction:
    return ManagementAction(kind=kind, ticket=ticket, description=f"{kind} #{ticket}")


SCALP_SPEC = SymbolSpec(
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


class TestScalpComment:
    def test_marker_is_appended_and_state_is_still_parsable(self):
        from gold_trader.trade_management.break_even import parse_position_comment

        comment = format_scalp_comment("GB", 1999.123, 0.25, 2, volume_step=0.01)
        assert comment.startswith("GB|")
        assert comment.endswith("|SCALP")
        state = parse_position_comment(comment)
        assert state is not None
        assert state["initial_sl"] == pytest.approx(1999.12)
        assert state["initial_volume"] == pytest.approx(0.25)

    def test_volume_precision_is_preserved_for_fine_steps(self):
        comment = format_scalp_comment("GB", 1999.0, 0.015, 2, volume_step=0.001)
        assert "vol=0.015" in comment


class TestManagementException:
    def test_a_marked_position_is_a_scalp(self):
        assert is_scalp_position(_position(1, "GB|sl=1999.00|vol=0.10|SCALP")) is True
        assert is_scalp_position(_position(1, "GB|sl=1999.00|vol=0.10")) is False
        assert is_scalp_position(_position(1, "")) is False
        assert is_scalp_position(_position(1, "SCALP manual note")) is True

    def test_management_allowed_follows_the_switch(self):
        scalp = _position(1, "GB|sl=1999.00|vol=0.10|SCALP")
        swing = _position(2, "GB|sl=1999.00|vol=0.10")
        assert management_allowed(scalp, make_cfg(scalping_disable_management=True)) is False
        assert management_allowed(swing, make_cfg(scalping_disable_management=True)) is True
        # with the exception switched off, the scalp is managed like anything else
        assert management_allowed(scalp, make_cfg(scalping_disable_management=False)) is True

    def test_optimising_actions_are_dropped_for_scalps_only(self):
        cfg = make_cfg(scalping_disable_management=True)
        scalp = _position(1, "GB|sl=1999.00|vol=0.10|SCALP")
        swing = _position(2, "GB|sl=1999.00|vol=0.10")
        actions = [
            _action("move_sl", 1),
            _action("partial_close", 1),
            _action("move_sl", 2),
            _action("full_close", 1),
            _action("delete_order", 1),
            _action("move_sl", 99),  # unknown ticket -> not ours to decide
        ]
        kept = filter_management_actions(actions, [scalp, swing], cfg)
        kinds_by_ticket = {(a.kind, a.ticket) for a in kept}
        assert ("move_sl", 2) in kinds_by_ticket
        assert ("move_sl", 99) in kinds_by_ticket
        assert ("move_sl", 1) not in kinds_by_ticket
        assert ("partial_close", 1) not in kinds_by_ticket
        # protective closes for a scalp survive: only optimisation is blocked
        assert ("full_close", 1) in kinds_by_ticket
        assert ("delete_order", 1) in kinds_by_ticket

    def test_allowed_kinds_are_exactly_the_protective_ones(self):
        assert SCALP_ALLOWED_KINDS == ("full_close", "delete_order")

    def test_nothing_is_filtered_when_the_switch_is_off(self):
        cfg = make_cfg(scalping_disable_management=False)
        scalp = _position(1, "GB|sl=1999.00|vol=0.10|SCALP")
        actions = [_action("move_sl", 1), _action("partial_close", 1)]
        assert filter_management_actions(actions, [scalp], cfg) == actions

    def test_empty_position_list_is_a_no_op(self):
        cfg = make_cfg(scalping_disable_management=True)
        actions = [_action("move_sl", 7)]
        assert filter_management_actions(actions, [], cfg) == actions

    def test_scalp_policy_manages_nothing_by_construction(self):
        scalp = _position(1, "GB|sl=1999.00|vol=0.10|SCALP")
        assert scalp_exits_only([scalp], 2000.0, 2000.3, make_cfg(), SCALP_SPEC) == []
        assert scalp_exits_only([], 2000.0, 2000.3, make_cfg(), SCALP_SPEC) == []


class TestDailyCapCounter:
    def test_counts_only_the_bots_own_opening_scalps(self):
        deals = [
            SimpleNamespace(magic=123456789, entry=DEAL_ENTRY_IN, comment="GB|SCALP"),
            SimpleNamespace(magic=123456789, entry=DEAL_ENTRY_IN, comment="GB|SCALP"),
            SimpleNamespace(magic=123456789, entry=1, comment="GB|SCALP"),      # closing deal
            SimpleNamespace(magic=999, entry=DEAL_ENTRY_IN, comment="GB|SCALP"),  # manual bot
            SimpleNamespace(magic=123456789, entry=DEAL_ENTRY_IN, comment="GB"),  # not a scalp
        ]
        assert count_scalp_entries(deals, magic=123456789) == 2

    @pytest.mark.parametrize("deals", [None, [], ()])
    def test_no_deals_means_zero(self, deals):
        assert count_scalp_entries(deals, magic=1) == 0

    def test_malformed_records_are_skipped_not_counted(self):
        deals = [
            SimpleNamespace(magic="not-a-number", entry=DEAL_ENTRY_IN, comment="SCALP"),
            SimpleNamespace(magic=1, entry=None, comment="SCALP"),
            SimpleNamespace(magic=1, entry=DEAL_ENTRY_IN, comment=None),
            SimpleNamespace(magic=1, entry=DEAL_ENTRY_IN, comment="GB|SCALP"),
        ]
        assert count_scalp_entries(deals, magic=1) == 1

    def test_marker_and_entry_types_are_configurable(self):
        deals = [SimpleNamespace(magic=1, entry=1, comment="X|SCALP")]
        assert count_scalp_entries(deals, magic=1, entry_types=(1,)) == 1
        assert count_scalp_entries(deals, magic=1, marker="OTHER") == 0


class TestDailyCapGate:
    def test_unknown_count_blocks_entries(self):
        status = scalp_entry_allowed(None, limit=6)
        assert status.allowed is False
        assert "unknown" in status.reason

    def test_cap_reached_blocks_entries(self):
        status = scalp_entry_allowed(6, limit=6)
        assert status.allowed is False
        assert "daily scalp cap reached" in status.reason

    def test_below_the_cap_allows(self):
        status = scalp_entry_allowed(5, limit=6)
        assert status.allowed is True
        assert status.entries_today == 5
        assert status.limit == 6

    def test_a_broken_limit_is_refused_instead_of_trading_freely(self):
        status = scalp_entry_allowed(0, limit=0)
        assert status.allowed is False
        assert "must be >= 1" in status.reason

    def test_zero_trades_is_allowed_when_the_limit_is_sane(self):
        assert scalp_entry_allowed(0, limit=1).allowed is True
