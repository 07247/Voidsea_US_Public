from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strategy.strategy import Strategy, StrategyConfig


class StrategyTests(unittest.TestCase):
    def test_example_crossovers_and_warmup(self):
        close = [1, 1, 1, 3, 3, 1, 1]
        bars = pd.DataFrame(
            {"open": close, "high": [v + 0.1 for v in close],
             "low": [v - 0.1 for v in close], "close": close, "volume": 100},
            index=pd.date_range("2026-01-05 09:30", periods=7, freq="5min", tz="America/New_York"),
        )
        strategy = Strategy(StrategyConfig(sma_short=2, sma_long=3, atr_period=2))
        prepared = strategy.prepare_bars(bars)
        self.assertEqual(prepared["signal"].tolist(),
                         ["HOLD", "HOLD", "HOLD", "BUY", "HOLD", "SELL", "HOLD"])
        self.assertNotIn("buy_score", prepared.columns)

    def test_prepare_bars_adds_all_indicators_and_signal(self) -> None:
        index = pd.date_range("2026-01-05 14:30", periods=100, freq="5min", tz="UTC")
        close = np.linspace(100, 120, len(index)) + np.sin(np.arange(len(index)))
        bars = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.arange(len(index)) + 1_000,
            },
            index=index,
        )

        result = Strategy().prepare_bars(bars)

        expected = {"sma_10", "sma_30", "atr_14", "signal"}
        self.assertTrue(expected.issubset(result.columns))
        self.assertEqual(str(result.index.tz), "America/New_York")
        self.assertTrue(
            set(result["signal"].unique()).issubset({"BUY", "SELL", "HOLD"})
        )


if __name__ == "__main__":
    unittest.main()
