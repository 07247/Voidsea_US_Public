from __future__ import annotations

import unittest

import pandas as pd

from broker.backtest_broker import BacktestBroker
from broker.base import OrderSide
from engine.trading_engine import TradingEngine
from risk.risk_manager import RiskConfig, RiskManager
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


class BrokerAndEngineTests(unittest.TestCase):

    def test_backtest_broker_charges_commission_and_slippage(
        self,
    ) -> None:
        broker = BacktestBroker(
            1_000,
            commission_rate=0.001,
            slippage_bps=10,
        )

        timestamp = pd.Timestamp(
            "2026-01-05 10:00",
            tz="America/New_York",
        )

        buy = broker.submit_market_order(
            "SPY",
            OrderSide.BUY,
            5,
            price=100,
            timestamp=timestamp,
        )

        sell = broker.submit_market_order(
            "SPY",
            OrderSide.SELL,
            5,
            price=110,
            timestamp=timestamp,
        )

        self.assertGreater(
            buy.fill_price,
            100,
        )

        self.assertLess(
            sell.fill_price,
            110,
        )

        self.assertGreater(
            broker.total_commission,
            0,
        )

        self.assertGreater(
            broker.total_slippage_cost,
            0,
        )

        self.assertEqual(
            len(broker.closed_trades),
            1,
        )

    def test_engine_executes_signal_on_next_open(
        self,
    ) -> None:
        index = pd.date_range(
            "2026-01-05 09:30",
            periods=4,
            freq="5min",
            tz="America/New_York",
        )

        bars = pd.DataFrame(
            {
                "open": [
                    100.0,
                    101.0,
                    105.0,
                    106.0,
                ],
                "high": [
                    101.0,
                    102.0,
                    106.0,
                    107.0,
                ],
                "low": [
                    99.0,
                    100.0,
                    104.0,
                    105.0,
                ],
                "close": [
                    100.5,
                    104.0,
                    105.5,
                    106.5,
                ],
                "volume": [
                    1_000,
                    1_000,
                    1_000,
                    1_000,
                ],
                "atr_14": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                "signal": [
                    "BUY",
                    "HOLD",
                    "SELL",
                    "HOLD",
                ],
            },
            index=index,
        )

        broker = BacktestBroker(
            10_000
        )

        risk = RiskManager(
            RiskConfig(
                risk_per_trade_pct=0.01,
                max_position_pct=0.25,
                max_daily_loss_pct=0.03,
                atr_stop_multiplier=2,
            )
        )

        engine = TradingEngine(
            broker,
            PresetSignalStrategy(),
            risk,
            cooldown_bars=0,
        )

        result, trades, _ = engine.run_backtest(
            "SPY",
            bars,
        )

        self.assertEqual(
            result["trade_count"],
            1,
        )

        self.assertEqual(
            trades.iloc[0]["entry_price"],
            101.0,
        )

        self.assertEqual(
            trades.iloc[0]["exit_price"],
            106.0,
        )

    def test_atr_stop_does_not_widen_after_entry(
        self,
    ) -> None:
        index = pd.date_range(
            "2026-01-05 09:30",
            periods=3,
            freq="5min",
            tz="America/New_York",
        )

        bars = pd.DataFrame(
            {
                "open": [
                    100.0,
                    100.0,
                    96.0,
                ],
                "high": [
                    101.0,
                    101.0,
                    97.0,
                ],
                "low": [
                    99.0,
                    96.0,
                    95.0,
                ],
                "close": [
                    100.0,
                    97.0,
                    96.0,
                ],
                "volume": [
                    1_000,
                    1_000,
                    1_000,
                ],
                "atr_14": [
                    1.0,
                    10.0,
                    10.0,
                ],
                "signal": [
                    "BUY",
                    "HOLD",
                    "HOLD",
                ],
            },
            index=index,
        )

        broker = BacktestBroker(
            10_000
        )

        engine = TradingEngine(
            broker,
            PresetSignalStrategy(),
            RiskManager(),
            cooldown_bars=0,
        )

        result, trades, _ = engine.run_backtest(
            "SPY",
            bars,
        )

        self.assertEqual(
            result["trade_count"],
            1,
        )

        self.assertEqual(
            trades.iloc[0]["entry_atr"],
            1.0,
        )

        self.assertEqual(
            trades.iloc[0]["initial_stop_price"],
            98.0,
        )

        self.assertEqual(
            trades.iloc[0]["exit_reason"],
            "atr_stop_loss",
        )

    def test_cooldown_blocks_immediate_reentry(
        self,
    ) -> None:
        index = pd.date_range(
            "2026-01-05 09:30",
            periods=5,
            freq="5min",
            tz="America/New_York",
        )

        bars = pd.DataFrame(
            {
                "open": [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                ],
                "high": [
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                ],
                "low": [
                    99.0,
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                ],
                "close": [
                    100.5,
                    101.5,
                    102.5,
                    103.5,
                    104.5,
                ],
                "volume": [
                    1_000,
                    1_000,
                    1_000,
                    1_000,
                    1_000,
                ],
                "atr_14": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                "signal": [
                    "BUY",
                    "SELL",
                    "BUY",
                    "HOLD",
                    "HOLD",
                ],
            },
            index=index,
        )

        broker = BacktestBroker(
            10_000
        )

        engine = TradingEngine(
            broker,
            PresetSignalStrategy(),
            RiskManager(),
            cooldown_bars=3,
        )

        result, _trades, _equity = (
            engine.run_backtest(
                "SPY",
                bars,
            )
        )

        self.assertEqual(
            result["trade_count"],
            1,
        )

        self.assertEqual(
            len(broker.fills),
            2,
        )

    def test_intrabar_stop_fills_at_stop_price(
            self,
    ) -> None:
        index = pd.date_range(
            "2026-01-05 09:30",
            periods=2,
            freq="5min",
            tz="America/New_York",
        )

        bars = pd.DataFrame(
            {
                "open": [
                    100.0,
                    100.0,
                ],
                "high": [
                    101.0,
                    101.0,
                ],
                "low": [
                    99.0,
                    97.0,
                ],
                "close": [
                    100.0,
                    100.0,
                ],
                "volume": [
                    1_000,
                    1_000,
                ],
                "atr_14": [
                    1.0,
                    1.0,
                ],
                "signal": [
                    "BUY",
                    "HOLD",
                ],
            },
            index=index,
        )

        broker = BacktestBroker(
            10_000
        )

        engine = TradingEngine(
            broker,
            PresetSignalStrategy(),
            RiskManager(),
            cooldown_bars=0,
        )

        result, trades, _ = engine.run_backtest(
            "SPY",
            bars,
        )

        self.assertEqual(
            result["trade_count"],
            1,
        )

        self.assertEqual(
            trades.iloc[0]["initial_stop_price"],
            98.0,
        )

        self.assertEqual(
            trades.iloc[0]["exit_price"],
            98.0,
        )

        self.assertEqual(
            trades.iloc[0]["exit_reason"],
            "atr_stop_loss",
        )

    def test_gap_below_stop_fills_at_open(
            self,
    ) -> None:
        index = pd.date_range(
            "2026-01-05 09:30",
            periods=3,
            freq="5min",
            tz="America/New_York",
        )

        bars = pd.DataFrame(
            {
                "open": [
                    100.0,
                    100.0,
                    95.0,
                ],
                "high": [
                    101.0,
                    101.0,
                    96.0,
                ],
                "low": [
                    99.0,
                    99.0,
                    94.0,
                ],
                "close": [
                    100.0,
                    100.0,
                    95.0,
                ],
                "volume": [
                    1_000,
                    1_000,
                    1_000,
                ],
                "atr_14": [
                    1.0,
                    1.0,
                    1.0,
                ],
                "signal": [
                    "BUY",
                    "HOLD",
                    "HOLD",
                ],
            },
            index=index,
        )

        broker = BacktestBroker(
            10_000
        )

        engine = TradingEngine(
            broker,
            PresetSignalStrategy(),
            RiskManager(),
            cooldown_bars=0,
        )

        _, trades, _ = engine.run_backtest(
            "SPY",
            bars,
        )

        self.assertEqual(
            trades.iloc[0]["initial_stop_price"],
            98.0,
        )

        # 98 的止损遇到 95 跳空，
        # 不允许回测假设能在 98 成交。
        self.assertEqual(
            trades.iloc[0]["exit_price"],
            95.0,
        )

        self.assertEqual(
            trades.iloc[0]["exit_reason"],
            "atr_stop_loss",
        )


if __name__ == "__main__":
    unittest.main()