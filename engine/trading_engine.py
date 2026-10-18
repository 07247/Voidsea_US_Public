"""Broker-neutral strategy execution for live, paper, and backtest modes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import isfinite

import pandas as pd

from broker.backtest_broker import BacktestBroker
from broker.base import Broker, OrderResult, OrderSide
from config import (
    BAR_CLOSE_BUFFER_SECONDS,
    COOLDOWN_BARS,
    TIMEFRAME_MINUTES,
)
from market_data.bars import filter_completed_bars
from engine.trading_state import (
    TradingStateStore,
)

from performance.backtest_metrics import (
    calculate_backtest_metrics,
)

from risk.risk_manager import RiskManager
from strategy.strategy import Signal, Strategy


@dataclass(frozen=True)
class TradeDecision:
    signal: Signal
    reason: str
    quantity: float = 0.0
    order: OrderResult | None = None


class TradingEngine:
    def __init__(
            self,
            broker: Broker,
            strategy: Strategy | None = None,
            risk_manager: RiskManager | None = None,
            *,
            cooldown_bars: int = COOLDOWN_BARS,
            state_store: TradingStateStore | None = None,
            timeframe_minutes: int = TIMEFRAME_MINUTES,
            bar_close_buffer_seconds: int = BAR_CLOSE_BUFFER_SECONDS,
    ) -> None:
        if cooldown_bars < 0:
            raise ValueError("cooldown_bars cannot be negative")

        self.broker = broker
        self.strategy = strategy or Strategy()
        self.risk_manager = risk_manager or RiskManager()
        self.cooldown_bars = cooldown_bars

        # 回测模式继续使用内存里的 cooldown 计数。
        self._bars_since_trade = cooldown_bars

        # paper/live 模式可以使用持久化状态。
        self.state_store = state_store
        if timeframe_minutes <= 0:
            raise ValueError(
                "timeframe_minutes must be positive"
            )

        if bar_close_buffer_seconds < 0:
            raise ValueError(
                "bar_close_buffer_seconds cannot be negative"
            )

        self.timeframe_minutes = timeframe_minutes
        self.bar_close_buffer_seconds = (
            bar_close_buffer_seconds
        )

        from engine.live_execution import LiveExecution
        self._live_execution = LiveExecution(self)

    def run_latest(
            self,
            symbol: str,
            bars: pd.DataFrame,
            *,
            execute: bool = False,
            as_of: datetime | pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, TradeDecision]:
        """
        Evaluate the latest strategy signal.

        paper/live 模式下：
        - 恢复上一次交易状态
        - 防止同一根 K 线重复执行
        - 根据实际经过的 K 线数量计算 cooldown
        """

        completed = filter_completed_bars(
            bars, timeframe_minutes=self.timeframe_minutes, now=as_of,
            buffer_seconds=self.bar_close_buffer_seconds,
        )
        prepared = self.strategy.prepare_bars(completed)
        data = self._single_symbol(prepared, symbol)
        return prepared, self._live_execution.run(
            symbol, data=data, execute=execute, as_of=as_of,
        )

    def monitor(self, symbol: str, *, execute: bool = False,
                as_of: datetime | None = None) -> TradeDecision:
        """Reconcile orders and check risk without requiring a new bar."""
        return self._live_execution.run(symbol, execute=execute, as_of=as_of)

    def run_backtest(
        self,
        symbol: str,
        bars: pd.DataFrame,
    ) -> tuple[
        dict[str, float | int | str],
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Execute close-bar signals at the following bar's open price.
        """

        if not isinstance(self.broker, BacktestBroker):
            raise TypeError(
                "run_backtest requires a BacktestBroker"
            )

        prepared = self.strategy.prepare_bars(bars)

        data = self._single_symbol(
            prepared,
            symbol,
        )

        pending_signal = Signal.HOLD
        pending_atr: float | None = None
        entry_atr: float | None = None
        stop_price: float | None = None

        equity_records: list[dict[str, object]] = []

        for timestamp, row in data.iterrows():
            # 回测每经过一根 K 线，
            # cooldown 计数增加 1。
            self._bars_since_trade += 1

            open_price = float(row["open"])
            low_price = float(row["low"])
            close_price = float(row["close"])

            # ==================================================
            # 开盘价 Mark-to-Market
            # ==================================================

            self.broker.mark_to_market(
                symbol,
                open_price,
            )

            opening_account = self.broker.get_account()

            self.risk_manager.observe_equity(
                timestamp,
                opening_account.equity,
            )

            # ==================================================
            # 上一根 K 线产生的信号
            # 在当前 K 线开盘成交
            # ==================================================

            if pending_signal is not Signal.HOLD:
                decision = self._handle_signal(
                    symbol=symbol,
                    signal=pending_signal,
                    price=open_price,
                    atr=pending_atr,
                    stop_price=stop_price,
                    timestamp=timestamp,
                    execute=True,
                )

                if decision.order is not None:

                    if decision.signal is Signal.BUY:
                        entry_atr = pending_atr

                        entry_price = (
                                decision.order.fill_price
                                or open_price
                        )

                        stop_price = (
                            self.risk_manager.initial_stop_price(
                                entry_price=entry_price,
                                entry_atr=entry_atr,
                            )
                        )

                    elif decision.signal is Signal.SELL:
                        self.broker.closed_trades[-1][
                            "entry_atr"
                        ] = entry_atr

                        self.broker.closed_trades[-1][
                            "initial_stop_price"
                        ] = stop_price

                        self.broker.closed_trades[-1][
                            "exit_reason"
                        ] = decision.reason

                        entry_atr = None
                        stop_price = None

            # ==================================================
            # 盘中 ATR Stop
            #
            # 如果当前 K 线的最低价碰到 stop，
            # 认为止损订单在本根 K 线内触发。
            #
            # 如果开盘已经低于 stop，代表发生跳空，
            # 应按开盘价作为成交参考，而不是假装能在 stop 成交。
            # ==================================================

            position = self.broker.get_position(symbol)

            if (
                    position is not None
                    and stop_price is not None
            ):
                stop_fill_price = (
                    self._backtest_stop_fill_price(
                        open_price=open_price,
                        low_price=low_price,
                        stop_price=stop_price,
                    )
                )

                if stop_fill_price is not None:
                    stop_decision = self._handle_signal(
                        symbol=symbol,
                        signal=Signal.SELL,
                        price=stop_fill_price,
                        atr=None,
                        stop_price=stop_price,
                        timestamp=timestamp,
                        execute=True,
                    )

                    if stop_decision.order is not None:
                        self.broker.closed_trades[-1][
                            "entry_atr"
                        ] = entry_atr

                        self.broker.closed_trades[-1][
                            "initial_stop_price"
                        ] = stop_price

                        self.broker.closed_trades[-1][
                            "exit_reason"
                        ] = "atr_stop_loss"

                        entry_atr = None
                        stop_price = None

            self.broker.mark_to_market(
                symbol,
                close_price,
            )

            account = self.broker.get_account()

            self.risk_manager.observe_equity(
                timestamp,
                account.equity,
            )

            equity_records.append(
                {
                    "timestamp": timestamp,
                    "equity": account.equity,
                }
            )

            # ==================================================
            # 当前 K 线收盘后产生下一根要执行的信号
            # ==================================================

            current_signal = self._coerce_signal(
                row["signal"]
            )

            position = self.broker.get_position(symbol)

            atr = self._clean_number(
                row.get(self._atr_column)
            )

            # ==================================================
            # 风控退出
            # ==================================================

            if position is not None:
                risk_exit = self.risk_manager.exit_reason(
                    position,
                    account,
                    price=close_price,
                    stop_price=stop_price,
                )

                if risk_exit:
                    current_signal = Signal.SELL

            pending_signal = current_signal
            pending_atr = atr

        # ==================================================
        # 数据结束时强制平掉剩余仓位
        # ==================================================

        position = self.broker.get_position(symbol)

        if position is not None:
            timestamp = data.index[-1]

            last_close = float(
                data.iloc[-1]["close"]
            )

            self.broker.submit_market_order(
                symbol,
                OrderSide.SELL,
                position.qty,
                price=last_close,
                timestamp=timestamp,
            )

            self.broker.closed_trades[-1][
                "exit_reason"
            ] = "end_of_data"
            self.broker.closed_trades[-1][
                "entry_atr"
            ] = entry_atr

            self.broker.closed_trades[-1][
                "initial_stop_price"
            ] = stop_price

            self._bars_since_trade = 0

            if equity_records:
                equity_records[-1][
                    "equity"
                ] = self.broker.get_account().equity

        trades = pd.DataFrame(
            self.broker.closed_trades
        )

        equity = pd.DataFrame(
            equity_records
        ).set_index("timestamp")

        return (
            self._backtest_result(
                symbol,
                trades,
                equity,
            ),
            trades,
            equity,
        )

    @property
    def _atr_column(self) -> str:
        return (
            f"atr_{self.strategy.settings.atr_period}"
        )

    def _handle_signal(
        self,
        *,
        symbol: str,
        signal: Signal,
        price: float,
        atr: float | None,
        stop_price: float | None,
        timestamp: datetime,
        execute: bool,
    ) -> TradeDecision:
        account = self.broker.get_account()

        position = self.broker.get_position(
            symbol
        )

        # ==================================================
        # 已有持仓时检查风险退出
        # ==================================================

        if position is not None:
            risk_exit = self.risk_manager.exit_reason(
                position,
                account,
                price=price,
                stop_price=stop_price,
            )

            if risk_exit:
                signal = Signal.SELL
                reason = risk_exit

            else:
                reason = "strategy_signal"

        else:
            reason = "strategy_signal"

        # ==================================================
        # BUY
        # ==================================================

        if signal is Signal.BUY:
            if position is not None:
                return TradeDecision(
                    Signal.HOLD,
                    "already_in_position",
                )

            rejection = (
                self.risk_manager.entry_rejection_reason(
                    account
                )
            )

            if rejection:
                return TradeDecision(
                    Signal.HOLD,
                    rejection,
                )

            if (
                self._bars_since_trade
                < self.cooldown_bars
            ):
                return TradeDecision(
                    Signal.HOLD,
                    "cooldown",
                )

            quantity = (
                self.risk_manager.calculate_position_size(
                    account,
                    price=price,
                    atr=atr,
                )
            )

            if quantity <= 0:
                return TradeDecision(
                    Signal.HOLD,
                    "position_size_zero",
                )

            decision = TradeDecision(
                Signal.BUY,
                reason,
                quantity,
            )

            side = OrderSide.BUY

        # ==================================================
        # SELL
        # ==================================================

        elif signal is Signal.SELL:
            if position is None:
                return TradeDecision(
                    Signal.HOLD,
                    "no_position",
                )

            quantity = position.qty

            decision = TradeDecision(
                Signal.SELL,
                reason,
                quantity,
            )

            side = OrderSide.SELL

        # ==================================================
        # HOLD
        # ==================================================

        else:
            return TradeDecision(
                Signal.HOLD,
                "no_action",
            )

        # ==================================================
        # dry-run
        # ==================================================

        if not execute:
            return decision

        # ==================================================
        # 真正提交订单
        # ==================================================

        order = self.broker.submit_market_order(
            symbol,
            side,
            quantity,
            price=price,
            timestamp=timestamp,
        )

        self._bars_since_trade = 0

        return replace(
            decision,
            quantity=order.qty,
            order=order,
        )

    def _backtest_result(
            self,
            symbol: str,
            trades: pd.DataFrame,
            equity: pd.DataFrame,
    ) -> dict[str, float | int | str]:
        """
        Build the final backtest performance report.
        """

        return calculate_backtest_metrics(
            symbol=symbol,
            trades=trades,
            equity=equity,
            initial_cash=self.broker.initial_cash,
            final_equity=(
                self.broker.get_account().equity
            ),
            total_commission=(
                self.broker.total_commission
            ),
            total_slippage_cost=(
                self.broker.total_slippage_cost
            ),
        )

    @staticmethod
    def _single_symbol(
        data: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        if not isinstance(
            data.index,
            pd.MultiIndex,
        ):
            if data.empty:
                raise ValueError(
                    "No regular-session bars are available"
                )

            return data

        if "symbol" not in data.index.names:
            raise ValueError(
                "MultiIndex market data needs a symbol level"
            )

        symbols = {
            str(item).upper(): item
            for item in data.index.get_level_values(
                "symbol"
            ).unique()
        }

        key = symbols.get(
            symbol.upper()
        )

        if key is None:
            raise ValueError(
                f"{symbol.upper()} is not present in market data"
            )

        result = data.xs(
            key,
            level="symbol",
        ).copy()

        if result.empty:
            raise ValueError(
                f"No regular-session bars are available for {symbol.upper()}"
            )

        return result

    @staticmethod
    def _backtest_stop_fill_price(
            *,
            open_price: float,
            low_price: float,
            stop_price: float,
    ) -> float | None:
        """
        Determine the reference fill price for a long stop loss.

        Gap below stop:
            fill from the opening price.

        Intrabar touch:
            fill from the stop price.

        BacktestBroker will apply configured slippage
        to this reference price.
        """

        if stop_price <= 0:
            return None

        # 跳空低开，已经越过止损价。
        if open_price <= stop_price:
            return open_price

        # 正常盘中跌到止损价。
        if low_price <= stop_price:
            return stop_price

        return None

    @staticmethod
    def _coerce_signal(
        value: object,
    ) -> Signal:
        try:
            return Signal(
                str(value).upper()
            )

        except ValueError:
            return Signal.HOLD

    @staticmethod
    def _count_bars_since_trade(
        data: pd.DataFrame,
        last_trade_bar: datetime,
    ) -> int:
        """
        计算从上次交易以后实际经过了多少根 K 线。

        不使用现实时间，因此周末和非交易时间
        不会被错误计算进 cooldown。
        """

        index = pd.DatetimeIndex(
            data.index
        )

        if index.tz is None:
            raise ValueError(
                "Market data timestamp must be timezone-aware"
            )

        last_trade = pd.Timestamp(
            last_trade_bar
        )

        if last_trade.tzinfo is None:
            raise ValueError(
                "last_trade_bar must be timezone-aware"
            )

        # 统一转换到 UTC 后比较。
        index_utc = index.tz_convert(
            "UTC"
        )

        last_trade_utc = (
            last_trade.tz_convert("UTC")
        )

        return int(
            (
                index_utc
                > last_trade_utc
            ).sum()
        )

    @staticmethod
    def _to_datetime(
        value: object,
    ) -> datetime:
        """
        将 pandas Timestamp 转成普通 datetime，
        方便保存到 JSON。
        """

        timestamp = pd.Timestamp(
            value
        )

        if timestamp.tzinfo is None:
            raise ValueError(
                "bar timestamp must be timezone-aware"
            )

        return timestamp.to_pydatetime()

    @staticmethod
    def _clean_number(
        value: object,
    ) -> float | None:
        try:
            number = float(value)

        except (TypeError, ValueError):
            return None

        return (
            number
            if isfinite(number)
            else None
        )


__all__ = [
    "TradeDecision",
    "TradingEngine",
]
