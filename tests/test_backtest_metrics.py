from __future__ import annotations

import unittest

import pandas as pd

from performance.backtest_metrics import (
    calculate_backtest_metrics,
)


class BacktestMetricsTests(unittest.TestCase):

    def test_extended_metrics(self) -> None:
        trades = pd.DataFrame(
            {
                "pnl": [
                    100.0,
                    -50.0,
                    -25.0,
                    200.0,
                ],
                "return_pct": [
                    1.0,
                    -0.5,
                    -0.25,
                    2.0,
                ],
                "entry_time": pd.to_datetime(
                    [
                        "2026-01-05 10:00",
                        "2026-01-05 11:00",
                        "2026-01-05 12:00",
                        "2026-01-05 13:00",
                    ],
                    utc=True,
                ),
                "exit_time": pd.to_datetime(
                    [
                        "2026-01-05 10:30",
                        "2026-01-05 12:00",
                        "2026-01-05 13:30",
                        "2026-01-05 15:00",
                    ],
                    utc=True,
                ),
            }
        )

        equity = pd.DataFrame(
            {
                "equity": [
                    10_000,
                    10_100,
                    10_050,
                    10_225,
                ]
            }
        )

        result = calculate_backtest_metrics(
            symbol="SPY",
            trades=trades,
            equity=equity,
            initial_cash=10_000,
            final_equity=10_225,
            total_commission=10,
            total_slippage_cost=5,
        )

        self.assertEqual(
            result["trade_count"],
            4,
        )

        self.assertEqual(
            result["winning_trades"],
            2,
        )

        self.assertEqual(
            result["losing_trades"],
            2,
        )

        self.assertAlmostEqual(
            result["win_rate_pct"],
            50.0,
        )

        self.assertAlmostEqual(
            result["gross_profit"],
            300.0,
        )

        self.assertAlmostEqual(
            result["gross_loss"],
            75.0,
        )

        self.assertAlmostEqual(
            result["profit_factor"],
            4.0,
        )

        self.assertAlmostEqual(
            result["average_win"],
            150.0,
        )

        self.assertAlmostEqual(
            result["average_loss"],
            -37.5,
        )

        self.assertAlmostEqual(
            result["payoff_ratio"],
            4.0,
        )

        self.assertAlmostEqual(
            result["average_win_return_pct"],
            1.5,
        )

        self.assertAlmostEqual(
            result["average_loss_return_pct"],
            -0.375,
        )

        self.assertAlmostEqual(
            result["average_holding_minutes"],
            75.0,
        )

        self.assertAlmostEqual(
            result["median_holding_minutes"],
            75.0,
        )

        self.assertAlmostEqual(
            result["max_holding_minutes"],
            120.0,
        )

        self.assertEqual(
            result["max_consecutive_losses"],
            2,
        )

        self.assertAlmostEqual(
            result["total_return_pct"],
            2.25,
        )

        self.assertAlmostEqual(
            result["total_commission"],
            10.0,
        )

        self.assertAlmostEqual(
            result["total_slippage_cost"],
            5.0,
        )

    def test_empty_trades(self) -> None:
        trades = pd.DataFrame(
            columns=[
                "pnl",
                "return_pct",
                "entry_time",
                "exit_time",
            ]
        )

        equity = pd.DataFrame(
            {
                "equity": [
                    10_000.0,
                ]
            }
        )

        result = calculate_backtest_metrics(
            symbol="SPY",
            trades=trades,
            equity=equity,
            initial_cash=10_000,
            final_equity=10_000,
            total_commission=0,
            total_slippage_cost=0,
        )

        self.assertEqual(
            result["trade_count"],
            0,
        )

        self.assertEqual(
            result["winning_trades"],
            0,
        )

        self.assertEqual(
            result["losing_trades"],
            0,
        )

        self.assertEqual(
            result["profit_factor"],
            0.0,
        )

        self.assertEqual(
            result["payoff_ratio"],
            0.0,
        )

        self.assertEqual(
            result["max_consecutive_losses"],
            0,
        )

        self.assertEqual(
            result["max_drawdown_pct"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()