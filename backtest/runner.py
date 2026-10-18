"""Backtest runner backed by the unified TradingEngine."""

from __future__ import annotations

import pandas as pd

from broker.backtest_broker import BacktestBroker
from config import (
    COMMISSION_RATE,
    COOLDOWN_BARS,
    INITIAL_CASH,
    MIN_COMMISSION,
    SLIPPAGE_BPS,
    SYMBOL,
)
from engine.trading_engine import TradingEngine
from risk.risk_manager import RiskManager
from strategy.strategy import Strategy


def run_backtest(
    df: pd.DataFrame,
    initial_cash: float = INITIAL_CASH,
    *,
    symbol: str = SYMBOL,
    commission_rate: float = COMMISSION_RATE,
    min_commission: float = MIN_COMMISSION,
    slippage_bps: float = SLIPPAGE_BPS,
    cooldown_bars: int = COOLDOWN_BARS,
):
    broker = BacktestBroker(
        initial_cash,
        commission_rate=commission_rate,
        min_commission=min_commission,
        slippage_bps=slippage_bps,
    )
    engine = TradingEngine(
        broker,
        Strategy(),
        RiskManager(),
        cooldown_bars=cooldown_bars,
    )
    return engine.run_backtest(symbol, df)


__all__ = ["run_backtest"]
