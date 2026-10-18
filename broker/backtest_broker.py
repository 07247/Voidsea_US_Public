"""In-memory broker that simulates long-only market-order fills."""

from __future__ import annotations

from datetime import datetime

from broker.base import (
    AccountSnapshot,
    Broker,
    OrderResult,
    OrderSide,
    PositionSnapshot,
)


class BacktestBroker(Broker):
    def __init__(
        self,
        initial_cash: float = 10_000.0,
        *,
        commission_rate: float = 0.0,
        min_commission: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if min(commission_rate, min_commission, slippage_bps) < 0:
            raise ValueError("execution costs cannot be negative")

        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.commission_rate = float(commission_rate)
        self.min_commission = float(min_commission)
        self.slippage_bps = float(slippage_bps)
        self.last_prices: dict[str, float] = {}
        self._positions: dict[str, PositionSnapshot] = {}
        self._entry_times: dict[str, datetime] = {}
        self._entry_commissions: dict[str, float] = {}
        self.fills: list[dict[str, object]] = []
        self.closed_trades: list[dict[str, object]] = []
        self.total_commission = 0.0
        self.total_slippage_cost = 0.0
        self._orders: dict[str, OrderResult] = {}

    def get_order(self, client_order_id: str) -> OrderResult | None:
        return self._orders.get(client_order_id)

    def mark_to_market(self, symbol: str, price: float) -> None:
        if price <= 0:
            raise ValueError("market price must be positive")
        self.last_prices[symbol.upper()] = float(price)

    def get_account(self) -> AccountSnapshot:
        market_value = sum(
            position.qty * self.last_prices.get(symbol, position.avg_entry_price)
            for symbol, position in self._positions.items()
        )
        equity = self.cash + market_value
        return AccountSnapshot(cash=self.cash, equity=equity, buying_power=self.cash)

    def get_positions(self) -> list[PositionSnapshot]:
        return [
            PositionSnapshot(
                symbol=symbol,
                qty=position.qty,
                avg_entry_price=position.avg_entry_price,
                market_value=position.qty
                * self.last_prices.get(symbol, position.avg_entry_price),
            )
            for symbol, position in self._positions.items()
            if position.qty > 0
        ]

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
        if client_order_id and client_order_id in self._orders:
            return self._orders[client_order_id]
        symbol = symbol.upper()
        side = OrderSide(side)
        if qty <= 0:
            raise ValueError("order quantity must be positive")
        if price is None:
            price = self.last_prices.get(symbol)
        if price is None or price <= 0:
            raise ValueError("a positive fill reference price is required")

        requested_price = float(price)
        direction = 1.0 if side is OrderSide.BUY else -1.0
        fill_price = requested_price * (1 + direction * self.slippage_bps / 10_000)

        if side is OrderSide.BUY:
            qty = min(float(qty), self._maximum_affordable_qty(fill_price))
            if qty <= 0:
                raise ValueError("insufficient cash for order and commission")
            commission = self._commission(fill_price * qty)
            self.cash -= fill_price * qty + commission
            previous = self._positions.get(symbol)
            previous_qty = previous.qty if previous else 0.0
            previous_cost = previous_qty * previous.avg_entry_price if previous else 0.0
            total_qty = previous_qty + qty
            average_price = (previous_cost + fill_price * qty) / total_qty
            self._positions[symbol] = PositionSnapshot(
                symbol=symbol,
                qty=total_qty,
                avg_entry_price=average_price,
                market_value=total_qty * fill_price,
            )
            if previous is None:
                self._entry_times[symbol] = timestamp
                self._entry_commissions[symbol] = commission
            else:
                self._entry_commissions[symbol] += commission
        else:
            previous = self._positions.get(symbol)
            if previous is None or previous.qty <= 0:
                raise ValueError(f"no open {symbol} position to sell")
            qty = min(float(qty), previous.qty)
            commission = self._commission(fill_price * qty)
            self.cash += fill_price * qty - commission
            entry_commission = self._entry_commissions[symbol] * (qty / previous.qty)
            gross_pnl = (fill_price - previous.avg_entry_price) * qty
            net_pnl = gross_pnl - entry_commission - commission
            basis = previous.avg_entry_price * qty + entry_commission
            self.closed_trades.append(
                {
                    "symbol": symbol,
                    "entry_time": self._entry_times[symbol],
                    "entry_price": previous.avg_entry_price,
                    "exit_time": timestamp,
                    "exit_price": fill_price,
                    "quantity": qty,
                    "gross_pnl": gross_pnl,
                    "commission": entry_commission + commission,
                    "pnl": net_pnl,
                    "return_pct": (net_pnl / basis * 100) if basis else 0.0,
                }
            )
            remaining = previous.qty - qty
            if remaining <= 1e-12:
                del self._positions[symbol]
                del self._entry_times[symbol]
                del self._entry_commissions[symbol]
            else:
                self._positions[symbol] = PositionSnapshot(
                    symbol=symbol,
                    qty=remaining,
                    avg_entry_price=previous.avg_entry_price,
                    market_value=remaining * fill_price,
                )
                self._entry_commissions[symbol] -= entry_commission

        slippage_cost = abs(fill_price - requested_price) * qty
        self.total_commission += commission
        self.total_slippage_cost += slippage_cost
        self.last_prices[symbol] = fill_price
        self.fills.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side.value,
                "quantity": qty,
                "requested_price": requested_price,
                "fill_price": fill_price,
                "commission": commission,
                "slippage_cost": slippage_cost,
            }
        )
        result = OrderResult(
            symbol=symbol,
            side=side,
            qty=qty,
            status="filled",
            requested_price=requested_price,
            fill_price=fill_price,
            commission=commission,
            timestamp=timestamp,
            filled_qty=qty,
            client_order_id=client_order_id,
        )
        if client_order_id:
            self._orders[client_order_id] = result
        return result

    def _commission(self, notional: float) -> float:
        if notional <= 0:
            return 0.0
        return max(self.min_commission, notional * self.commission_rate)

    def _maximum_affordable_qty(self, price: float) -> float:
        if self.cash <= self.min_commission:
            return 0.0
        rate_qty = self.cash / (price * (1 + self.commission_rate))
        if self._commission(price * rate_qty) >= self.min_commission:
            return rate_qty
        return (self.cash - self.min_commission) / price


__all__ = ["BacktestBroker"]
