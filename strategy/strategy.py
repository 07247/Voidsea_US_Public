"""One strategy pipeline shared by live, paper, and backtest execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from config import ATR_PERIOD, SMA_LONG, SMA_SHORT
from strategy.indicators import add_atr, add_sma, add_sma_signal, filter_regular_session


class Signal(str, Enum):
    """Actions requested by the strategy."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class StrategyConfig:
    sma_short: int = SMA_SHORT
    sma_long: int = SMA_LONG
    atr_period: int = ATR_PERIOD

    def __post_init__(self) -> None:
        if self.sma_short <= 0 or self.sma_long <= self.sma_short:
            raise ValueError("SMA windows must satisfy 0 < short < long")
        if self.atr_period <= 0:
            raise ValueError("ATR period must be positive")


class Strategy:
    """Educational SMA crossover example; not the private trading strategy."""

    REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

    def __init__(self, settings: StrategyConfig | None = None) -> None:
        self.settings = settings or StrategyConfig()

    def prepare_bars(
        self,
        bars: pd.DataFrame,
        *,
        regular_session_only: bool = True,
    ) -> pd.DataFrame:
        """Return a sorted copy containing every indicator and signal column."""

        self._validate_market_data(bars)
        data = self._normalise_timezone(bars).sort_index()

        if regular_session_only:
            data = filter_regular_session(data)

        cfg = self.settings
        data = add_sma(data, short_window=cfg.sma_short, long_window=cfg.sma_long)
        data = add_atr(data, period=cfg.atr_period)
        return add_sma_signal(data, short_window=cfg.sma_short, long_window=cfg.sma_long)

    def latest_signal(self, prepared_bars: pd.DataFrame) -> Signal:
        if prepared_bars.empty or "signal" not in prepared_bars:
            return Signal.HOLD
        try:
            return Signal(str(prepared_bars.iloc[-1]["signal"]).upper())
        except ValueError:
            return Signal.HOLD

    @classmethod
    def _validate_market_data(cls, bars: pd.DataFrame) -> None:
        missing = cls.REQUIRED_COLUMNS.difference(bars.columns)
        if missing:
            raise ValueError(f"Missing market data columns: {sorted(missing)}")
        if bars.empty:
            raise ValueError("Market data is empty")

    @staticmethod
    def _normalise_timezone(bars: pd.DataFrame) -> pd.DataFrame:
        data = bars.copy()
        if isinstance(data.index, pd.MultiIndex):
            if "timestamp" not in data.index.names:
                raise ValueError("MultiIndex market data needs a timestamp level")
            timestamps = data.index.get_level_values("timestamp")
            if not isinstance(timestamps, pd.DatetimeIndex):
                raise TypeError("timestamp level must contain datetime values")
            if timestamps.tz is None:
                data = data.tz_localize("UTC", level="timestamp")
            return data.tz_convert("America/New_York", level="timestamp")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Market data index must be a DatetimeIndex")
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        return data.tz_convert("America/New_York")


__all__ = ["Signal", "Strategy", "StrategyConfig"]
