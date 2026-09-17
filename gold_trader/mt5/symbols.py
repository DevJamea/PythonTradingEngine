"""Gold symbol discovery.

Never assumes the broker symbol is literally ``XAUUSD``: the configured
``symbol`` is tried first, then the candidate list in priority order,
then a deterministic fallback scan for names containing ``XAU`` or
``GOLD``. A candidate is only accepted when it is visible and in full
trade mode with sane volume constraints. The selected symbol (with its
trading properties) is printed and logged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ..models import SymbolSpec
from ._constants import const
from .connection import MT5DataError, MT5Error, require_mt5

logger = logging.getLogger("gold_trader.mt5.symbols")


class GoldSymbolNotFoundError(MT5Error):
    """No tradable gold symbol was found on the server."""

    def __init__(self, candidates: Sequence[str], scanned: List[str]) -> None:
        super().__init__(
            "no tradable gold symbol found. tried: "
            f"{', '.join(candidates) or '(none)'}; "
            f"broker names containing XAU/GOLD: {', '.join(scanned) or '(none)'}"
        )
        self.scanned = list(scanned)


def find_gold_symbol(
    preferred: Optional[str] = None,
    candidates: Sequence[str] = (),
) -> SymbolSpec:
    """Find the broker's tradable gold symbol and return its spec."""
    mt5_api = require_mt5()
    all_symbols = mt5_api.symbols_get()
    if all_symbols is None:
        raise MT5DataError(
            f"symbols_get() failed: {mt5_api.last_error()} "
            "(terminal not connected?)"
        )
    by_name: Dict[str, Any] = {s.name: s for s in all_symbols}

    scanned = sorted(
        name for name in by_name if "XAU" in name.upper() or "GOLD" in name.upper()
    )
    ordered: List[str] = []
    if preferred:
        ordered.append(preferred)
    for name in list(candidates) + scanned:
        if name not in ordered:
            ordered.append(name)

    for name in ordered:
        spec = _try_symbol(name, by_name)
        if spec is not None:
            _log_selected(spec)
            return spec
    raise GoldSymbolNotFoundError(
        [preferred or "", *candidates], scanned
    )


def _try_symbol(name: str, by_name: Dict[str, Any]) -> Optional[SymbolSpec]:
    """Validate one candidate; return its SymbolSpec or None."""
    mt5_api = require_mt5()
    info = by_name.get(name)
    if info is None:
        return None

    # Gold symbols are often hidden by default -- make it visible, then
    # re-read the (possibly updated) description.
    if not info.visible:
        try:
            mt5_api.symbol_select(name, True)
        except Exception as exc:
            logger.warning("symbol_select(%s) raised: %s", name, exc)
        info = mt5_api.symbol_info(name)
        if info is None:
            logger.info("candidate %s: description unavailable after select", name)
            return None

    if not info.visible:
        logger.info("candidate %s: not visible", name)
        return None
    trade_mode_full = const("SYMBOL_TRADE_MODE_FULL", 1)
    if info.trade_mode != trade_mode_full:
        logger.info(
            "candidate %s: trade mode %s is not FULL", name, info.trade_mode
        )
        return None
    if info.volume_min <= 0 or info.volume_max < info.volume_min:
        logger.info("candidate %s: invalid volume constraints", name)
        return None
    if info.volume_step <= 0:
        logger.info("candidate %s: invalid volume step", name)
        return None
    if info.point <= 0 or info.digits < 0:
        logger.info("candidate %s: invalid point/digits", name)
        return None

    return SymbolSpec(
        name=name,
        point=float(info.point),
        digits=int(info.digits),
        volume_min=float(info.volume_min),
        volume_max=float(info.volume_max),
        volume_step=float(info.volume_step),
        stops_level=int(info.stops_level),
        freeze_level=int(info.freeze_level),
        visible=bool(info.visible),
        trade_mode=int(info.trade_mode),
        contract_size=float(info.trade_contract_size),
        trade_tick_size=float(info.trade_tick_size),
        trade_tick_value=float(info.trade_tick_value),
        filling_mode=int(info.filling_mode),
    )


def _log_selected(spec: SymbolSpec) -> None:
    message = (
        f"Selected gold symbol: {spec.name} | point={spec.point} "
        f"digits={spec.digits} | volume={spec.volume_min}-{spec.volume_max} "
        f"step={spec.volume_step} | stops_level={spec.stops_level} "
        f"freeze_level={spec.freeze_level} | contract_size={spec.contract_size} "
        f"tick_size={spec.trade_tick_size} tick_value={spec.trade_tick_value}"
    )
    logger.info(message)
    print(message)
