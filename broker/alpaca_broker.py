"""Alpaca adapter for paper and live trading plus historical market data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.common.exceptions import APIError
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import GetCalendarRequest, MarketOrderRequest

from broker.base import (
    AccountSnapshot,
    Broker,
    OrderResult,
    OrderSide,
    PositionSnapshot,
    MarketQuote,
    TradingSession,
)
from config import (
    ALLOW_LIVE_TRADING,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    DATA_FEED,
    TIMEFRAME_MINUTES,
    TRADING_MODE,
)


class _BoundedRequests:
    """Bound SDK socket waits and leave retries to the durable execution loop.

    alpaca-py 0.44.0 uses _one_request as its single transport hook. Keep the
    adapter contract tests when upgrading the pinned SDK.
    """
    def _one_request(self, method, url, opts, retry):
        return super()._one_request(method, url, {**opts, "timeout": (3, 5)}, 0)


class _TradingClient(_BoundedRequests, TradingClient):
    pass


class _DataClient(_BoundedRequests, StockHistoricalDataClient):
    pass


class AlpacaBroker(Broker):
    """Use the same interface against Alpaca paper or live endpoints."""

    requires_monitoring = True

    def __init__(
        self,
        *,
        mode: str = TRADING_MODE,
        api_key: str | None = ALPACA_API_KEY,
        secret_key: str | None = ALPACA_SECRET_KEY,
        data_feed: str = DATA_FEED,
        allow_live_trading: bool = ALLOW_LIVE_TRADING,
    ) -> None:
        mode = mode.strip().lower()
        if mode not in {"paper", "live"}:
            raise ValueError("mode must be 'paper' or 'live'")
        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")

        self.mode = mode
        self.allow_live_trading = allow_live_trading
        self.data_feed = DataFeed(data_feed.lower())
        self.trading_client = _TradingClient(api_key, secret_key, paper=mode == "paper")
        self.data_client = _DataClient(api_key, secret_key)
        self.quote_client = _DataClient(api_key, secret_key)
        self._session_cache = None

    def get_account(self) -> AccountSnapshot:
        account = self.trading_client.get_account()
        return AccountSnapshot(
            cash=float(account.cash),
            equity=float(account.equity),
            buying_power=float(account.buying_power),
            account_id=str(account.id),
            previous_close_equity=float(account.last_equity),
        )

    def is_market_open(self) -> bool:
        return bool(self.trading_client.get_clock().is_open)

    def get_trading_session(self, timestamp: datetime) -> TradingSession | None:
        if timestamp.tzinfo is None:
            raise ValueError("Calendar lookup requires a timezone-aware timestamp")
        ny = ZoneInfo("America/New_York")
        day = timestamp.astimezone(ny).date()
        if self._session_cache is not None and self._session_cache[0] == day:
            return self._session_cache[1]
        calendar = self.trading_client.get_calendar(GetCalendarRequest(start=day, end=day))
        if not calendar:
            session = None
        else:
            if len(calendar) != 1 or calendar[0].date != day:
                raise ValueError("Broker returned an unexpected trading calendar date")
            # alpaca-py Calendar combines the date and exchange-local time,
            # yielding naive datetimes. Attach New York, never UTC.
            def exchange_time(value):
                return value.replace(tzinfo=ny) if value.tzinfo is None else value.astimezone(ny)
            session = TradingSession(exchange_time(calendar[0].open), exchange_time(calendar[0].close))
        self._session_cache = (day, session)
        return session

    def get_quote(self, symbol: str) -> MarketQuote:
        symbol = symbol.upper()
        quotes = self.quote_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.data_feed)
        )
        quote = quotes[symbol]
        return MarketQuote(float(quote.bid_price), float(quote.ask_price), quote.timestamp)

    def get_order(self, client_order_id: str) -> OrderResult | None:
        try:
            order = self.trading_client.get_order_by_client_id(client_order_id)
        except APIError as exc:
            # Only a confirmed 404 means not found. Network/auth errors must
            # never be interpreted as permission to submit another order.
            if exc.status_code == 404:
                return None
            raise
        return self._order_result(order)

    def validate_execution(self) -> None:
        if self.mode == "live" and not self.allow_live_trading:
            raise PermissionError("Live execution requires ALLOW_LIVE_TRADING=true")

    def cancel_order(self, order_id: str) -> None:
        self.validate_execution()
        self.trading_client.cancel_order_by_id(order_id)

    def get_positions(self) -> list[PositionSnapshot]:
        return [
            PositionSnapshot(
                symbol=str(position.symbol).upper(),
                qty=float(position.qty),
                avg_entry_price=float(position.avg_entry_price),
                market_value=float(position.market_value),
            )
            for position in self.trading_client.get_all_positions()
        ]

    def get_bars(
        self,
        symbol: str,
        days: int = 5,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe_minutes: int = TIMEFRAME_MINUTES,
    ):
        if days <= 0:
            raise ValueError("days must be positive")
        end = end or datetime.now(timezone.utc)
        start = start or end - timedelta(days=days)
        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed=self.data_feed,
        )
        return self.data_client.get_stock_bars(request).df

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
        del price, timestamp  # Alpaca determines the actual market fill.
        side = OrderSide(side)
        if not isfinite(qty) or qty <= 0:
            raise ValueError("order quantity must be positive")
        if self.mode == "live" and not self.allow_live_trading:
            raise PermissionError(
                "Live order blocked. Set ALLOW_LIVE_TRADING=true after reviewing the order."
            )

        request = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=(
                AlpacaOrderSide.BUY if side is OrderSide.BUY else AlpacaOrderSide.SELL
            ),
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = self.trading_client.submit_order(order_data=request)
        return self._order_result(order)

    @staticmethod
    def _order_result(order) -> OrderResult:
        filled_price = getattr(order, "filled_avg_price", None)
        filled_at = getattr(order, "filled_at", None)
        status = getattr(order, "status", "accepted")
        status_value = getattr(status, "value", str(status))
        return OrderResult(
            symbol=str(order.symbol).upper(),
            side=OrderSide(str(getattr(order.side, "value", order.side)).upper()),
            qty=float(order.qty),
            status=str(status_value),
            fill_price=float(filled_price) if filled_price else None,
            timestamp=filled_at,
            order_id=str(getattr(order, "id", "")) or None,
            raw=order,
            filled_qty=float(getattr(order, "filled_qty", 0) or 0),
            client_order_id=getattr(order, "client_order_id", None),
        )


__all__ = ["AlpacaBroker"]
