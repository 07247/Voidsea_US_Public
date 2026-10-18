from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from broker.backtest_broker import BacktestBroker
from engine.trading_engine import TradingEngine
from engine.trading_state import (
    SymbolTradingState,
    TradingStateStore,
)
from risk.risk_manager import RiskManager
from strategy.strategy import Strategy


class PresetSignalStrategy(Strategy):

    def prepare_bars(
        self,
        bars,
        *,
        regular_session_only=True,
    ):
        del regular_session_only
        return bars.copy()


class TradingStateTests(unittest.TestCase):

    def test_state_survives_restart(self) -> None:

        with TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "trading_state.json"
            )

            store = TradingStateStore(
                mode="paper",
                path=path,
            )

            trade_time = datetime.fromisoformat(
                "2026-08-28T10:30:00-04:00"
            )

            store.save(
                "SPY",
                SymbolTradingState(
                    last_processed_bar=trade_time,
                    last_trade_bar=trade_time,
                    last_trade_time=trade_time,
                    entry_atr=2.5,
                    stop_price=95.0,
                )
            )

            # 模拟程序重启
            new_store = TradingStateStore(
                mode="paper",
                path=path,
            )

            state = new_store.load("SPY")

            self.assertEqual(
                state.last_trade_bar,
                trade_time,
            )

            self.assertEqual(
                state.entry_atr,
                2.5,
            )

            self.assertEqual(
                state.stop_price,
                95.0,
            )

    def test_duplicate_bar_is_blocked(self) -> None:

        with TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "state.json"
            )

            store = TradingStateStore(
                mode="paper",
                path=path,
            )

            index = pd.date_range(
                "2026-08-28 10:30",
                periods=1,
                freq="5min",
                tz="America/New_York",
            )

            bars = pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.0],
                    "volume": [1000],
                    "atr_14": [1.0],
                    "signal": ["BUY"],
                },
                index=index,
            )

            store.save(
                "SPY",
                SymbolTradingState(
                    last_processed_bar=(
                        index[-1].to_pydatetime()
                    )
                ),
            )

            engine = TradingEngine(
                BacktestBroker(10_000),
                PresetSignalStrategy(),
                RiskManager(),
                state_store=store,
            )

            _, decision = engine.run_latest(
                "SPY",
                bars,
                execute=True,
            )

            self.assertEqual(
                decision.reason,
                "bar_already_processed",
            )

    def test_cooldown_survives_restart(self) -> None:

        with TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "state.json"
            )

            store = TradingStateStore(
                mode="paper",
                path=path,
            )

            index = pd.date_range(
                "2026-08-28 10:30",
                periods=2,
                freq="5min",
                tz="America/New_York",
            )

            bars = pd.DataFrame(
                {
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.0, 101.0],
                    "volume": [1000, 1000],
                    "atr_14": [1.0, 1.0],
                    "signal": ["HOLD", "BUY"],
                },
                index=index,
            )

            store.save(
                "SPY",
                SymbolTradingState(
                    last_processed_bar=(
                        index[0].to_pydatetime()
                    ),
                    last_trade_bar=(
                        index[0].to_pydatetime()
                    ),
                ),
            )

            # 模拟程序重启
            engine = TradingEngine(
                BacktestBroker(10_000),
                PresetSignalStrategy(),
                RiskManager(),
                cooldown_bars=3,
                state_store=TradingStateStore(
                    mode="paper",
                    path=path,
                ),
            )

            _, decision = engine.run_latest(
                "SPY",
                bars,
                execute=True,
            )

            self.assertEqual(
                decision.signal.value,
                "HOLD",
            )

            self.assertEqual(
                decision.reason,
                "cooldown",
            )


if __name__ == "__main__":
    unittest.main()