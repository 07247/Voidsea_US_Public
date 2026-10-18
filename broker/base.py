"""Broker-neutral account, position, and order contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    equity: float
    buying_power: float
    account_id: str = "backtest"
    previous_close_equity: float | None = None


@dataclass(frozen=True)
class MarketQuote:
    bid: float
    ask: float
    timestamp: datetime


@dataclass(frozen=True)
class TradingSession:
    opens_at: datetime
    closes_at: datetime

    def __post_init__(self) -> None:
        if self.opens_at.tzinfo is None or self.closes_at.tzinfo is None:
            raise ValueError("Trading session times must be timezone-aware")
        if self.opens_at >= self.closes_at:
            raise ValueError("Trading session must close after it opens")

    def contains(self, timestamp: datetime) -> bool:
        return self.opens_at <= timestamp < self.closes_at


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float


@dataclass(frozen=True)
class OrderResult:
    symbol: str
    side: OrderSide
    qty: float
    status: str
    requested_price: float | None = None
    fill_price: float | None = None
    commission: float = 0.0
    timestamp: datetime | None = None
    order_id: str | None = None
    raw: Any = None
    filled_qty: float = 0.0
    client_order_id: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in {"filled", "canceled", "expired", "rejected"}


class Broker(ABC):
    """Minimal interface used by both the trading and backtest engines."""

    requires_monitoring = False

    def validate_execution(self) -> None:
        """Fail before journaling if execution is locally disabled."""

    def get_order(self, client_order_id: str) -> OrderResult | None:
        raise NotImplementedError("Broker must support order reconciliation")

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("Broker must support cancellation")

    def get_quote(self, symbol: str) -> MarketQuote:
        raise NotImplementedError("Broker must supply current quotes")

    def is_market_open(self) -> bool:
        return True

    def get_trading_session(self, timestamp: datetime) -> TradingSession | None:
        raise NotImplementedError("Broker must supply the exchange trading calendar")

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[PositionSnapshot]:
        raise NotImplementedError

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        target = symbol.upper()
        return next(
            (item for item in self.get_positions() if item.symbol == target), None
        )

    @abstractmethod
    def submit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        *,
        price: float | None = None,
        timestamp: datetime | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        raise NotImplementedError

    def buy(self, symbol: str, qty: float, **kwargs: Any) -> OrderResult:
        return self.submit_market_order(symbol, OrderSide.BUY, qty, **kwargs)

    def sell(self, symbol: str, qty: float, **kwargs: Any) -> OrderResult:
        return self.submit_market_order(symbol, OrderSide.SELL, qty, **kwargs)


__all__ = [
    "AccountSnapshot",
    "Broker",
    "OrderResult",
    "OrderSide",
    "PositionSnapshot",
]
