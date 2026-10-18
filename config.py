"""Application configuration.

Every value that can materially change trading behaviour can be overridden in
``.env``. Percentages are stored as decimals (``0.01`` means 1%).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
TRADING_MODE = os.getenv("TRADING_MODE", "paper").strip().lower()
if TRADING_MODE not in {"paper", "live"}:
    raise ValueError("TRADING_MODE must be 'paper' or 'live'")

# A second, explicit switch prevents an accidental live order.
ALLOW_LIVE_TRADING = _env_bool("ALLOW_LIVE_TRADING", False)
PAPER = TRADING_MODE == "paper"  # Backwards-compatible alias.
DATA_FEED = os.getenv("DATA_FEED", "iex").strip().lower()

# Market data
SYMBOL = os.getenv("SYMBOL", "SPY").strip().upper()
TIMEFRAME_MINUTES = _env_int("TIMEFRAME_MINUTES", 5)
BAR_CLOSE_BUFFER_SECONDS = _env_int("BAR_CLOSE_BUFFER_SECONDS", 5)
HISTORICAL_DAYS = _env_int("HISTORICAL_DAYS", 365)

# Demonstration strategy only; these are not private strategy parameters.
SMA_SHORT = _env_int("SMA_SHORT", 10)
SMA_LONG = _env_int("SMA_LONG", 30)
ATR_PERIOD = _env_int("ATR_PERIOD", 14)

# Illustrative risk and execution assumptions, not calibrated for trading
INITIAL_CASH = _env_float("INITIAL_CASH", 10_000.0)
RISK_PER_TRADE_PCT = _env_float("RISK_PER_TRADE_PCT", 0.005)
MAX_POSITION_PCT = _env_float("MAX_POSITION_PCT", 0.10)
MAX_DAILY_LOSS_PCT = _env_float("MAX_DAILY_LOSS_PCT", 0.02)
ATR_STOP_MULTIPLIER = _env_float("ATR_STOP_MULTIPLIER", 2.0)
COMMISSION_RATE = _env_float("COMMISSION_RATE", 0.0005)
MIN_COMMISSION = _env_float("MIN_COMMISSION", 0.0)
SLIPPAGE_BPS = _env_float("SLIPPAGE_BPS", 2.0)
COOLDOWN_BARS = _env_int("COOLDOWN_BARS", 3)

