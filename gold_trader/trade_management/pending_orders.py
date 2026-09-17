"""Pending order management: duplicate detection and expiry cleanup."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence

from ..models import ManagementAction, OrderType, PendingOrderInfo


def find_duplicate(
    pendings: Sequence[PendingOrderInfo],
    order_type: OrderType,
    price: float,
    point: float,
) -> Optional[PendingOrderInfo]:
    """Existing same-side order at (almost) the same price, if any."""
    tolerance = max(point, 1e-12)
    for order in pendings:
        if (
            order.order_type.is_buy_side == order_type.is_buy_side
            and abs(order.price - price) <= tolerance
        ):
            return order
    return None


def expired_orders(
    pendings: Sequence[PendingOrderInfo], now: datetime
) -> List[PendingOrderInfo]:
    """Pending orders whose ``time_expire`` has passed (GTC never expires)."""
    return [
        order
        for order in pendings
        if order.time_expire is not None and order.time_expire <= now
    ]


def plan_cleanup(
    pendings: Sequence[PendingOrderInfo], now: datetime
) -> List[ManagementAction]:
    """Build delete actions for all expired pending orders."""
    return [
        ManagementAction(
            kind="delete_order",
            ticket=order.ticket,
            description=f"delete expired pending order #{order.ticket}",
        )
        for order in expired_orders(pendings, now)
    ]
