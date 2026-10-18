from __future__ import annotations

import unittest

from broker.base import AccountSnapshot, PositionSnapshot
from risk.risk_manager import RiskConfig, RiskManager


class RiskManagerTests(unittest.TestCase):
    def test_position_size_uses_tightest_limit(self) -> None:
        manager = RiskManager(
            RiskConfig(
                risk_per_trade_pct=0.01,
                max_position_pct=0.25,
                max_daily_loss_pct=0.03,
                atr_stop_multiplier=2,
            )
        )
        account = AccountSnapshot(cash=10_000, equity=10_000, buying_power=10_000)
        quantity = manager.calculate_position_size(account, price=100, atr=2)
        self.assertEqual(quantity, 25)

    def test_atr_stop_loss(self) -> None:
        manager = RiskManager()

        account = AccountSnapshot(
            cash=0,
            equity=9_900,
            buying_power=0,
        )

        position = PositionSnapshot(
            "SPY",
            10,
            100,
            990,
        )

        stop_price = manager.initial_stop_price(
            entry_price=100,
            entry_atr=1,
        )

        self.assertEqual(
            stop_price,
            98,
        )

        reason = manager.exit_reason(
            position,
            account,
            price=97,
            stop_price=stop_price,
        )

        self.assertEqual(
            reason,
            "atr_stop_loss",
        )


if __name__ == "__main__":
    unittest.main()
