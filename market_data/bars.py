"""Utilities for validating and selecting completed market-data bars."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from broker.base import TradingSession

import pandas as pd

from config import (
    BAR_CLOSE_BUFFER_SECONDS,
    TIMEFRAME_MINUTES,
)


def filter_completed_bars(
    bars: pd.DataFrame,
    *,
    timeframe_minutes: int = TIMEFRAME_MINUTES,
    now: datetime | pd.Timestamp | None = None,
    buffer_seconds: int = BAR_CLOSE_BUFFER_SECONDS,
) -> pd.DataFrame:
    """
    只保留已经完全结束的 K 线。

    Alpaca bar 的 timestamp 表示这根 K 线的开始时间。

    例如：
        timestamp = 10:30
        timeframe = 5 minutes

    代表：
        10:30 -> 10:35

    只有当前时间超过 10:35，并且超过安全缓冲时间后，
    才允许使用这根 K 线。
    """

    if bars.empty:
        raise ValueError("Market data is empty")

    if timeframe_minutes <= 0:
        raise ValueError(
            "timeframe_minutes must be positive"
        )

    if buffer_seconds < 0:
        raise ValueError(
            "buffer_seconds cannot be negative"
        )

    # ==================================================
    # 获取 K 线时间
    # ==================================================

    if isinstance(bars.index, pd.MultiIndex):
        if "timestamp" not in bars.index.names:
            raise ValueError(
                "MultiIndex market data needs a timestamp level"
            )

        timestamps = bars.index.get_level_values(
            "timestamp"
        )

    else:
        timestamps = pd.DatetimeIndex(
            bars.index
        )

    # ==================================================
    # 时间必须带时区
    # ==================================================

    if timestamps.tz is None:
        raise ValueError(
            "Market data timestamp must be timezone-aware"
        )

    # ==================================================
    # 当前时间
    # ==================================================

    if now is None:
        current_time = pd.Timestamp(
            datetime.now(timezone.utc)
        )

    else:
        current_time = pd.Timestamp(now)

    if current_time.tzinfo is None:
        raise ValueError(
            "Current time must be timezone-aware"
        )

    # 统一转换成 UTC 比较
    current_utc = current_time.tz_convert(
        "UTC"
    )

    timestamps_utc = timestamps.tz_convert(
        "UTC"
    )

    # ==================================================
    # 计算每根 K 线真正结束的时间
    # ==================================================

    bar_end_times = (
        timestamps_utc
        + pd.Timedelta(
            minutes=timeframe_minutes
        )
    )

    # 给行情数据额外留一点缓冲
    safe_time = (
        current_utc
        - pd.Timedelta(
            seconds=buffer_seconds
        )
    )

    completed_mask = (
        bar_end_times <= safe_time
    )

    completed = bars.loc[
        completed_mask
    ].copy()

    if completed.empty:
        raise ValueError(
            "No completed market-data bars are available"
        )

    return completed


def signal_rejection_reason(
    timestamp: datetime | pd.Timestamp,
    *,
    now: datetime | pd.Timestamp,
    session: TradingSession | None,
    timeframe_minutes: int = TIMEFRAME_MINUTES,
    buffer_seconds: int = BAR_CLOSE_BUFFER_SECONDS,
) -> str | None:
    """Accept only a current-session completed bar until the next bar is ready.

    E.g. the 10:30 five-minute bar is eligible from 10:35:05 up to (but
    excluding) 10:40:05 with a five-second close buffer. Older signals never
    become eligible just because the market reopens or a quote is current.
    """
    if timeframe_minutes <= 0 or buffer_seconds < 0:
        raise ValueError("Invalid bar timeframe or close buffer")
    current, start = pd.Timestamp(now), pd.Timestamp(timestamp)
    if pd.isna(current) or current.tzinfo is None:
        raise ValueError("Current time must be timezone-aware and valid")
    if pd.isna(start) or start.tzinfo is None:
        return "invalid_signal_time"
    if session is None or not session.contains(current):
        return "market_closed"
    duration = pd.Timedelta(minutes=timeframe_minutes)
    end = start + duration
    if start < session.opens_at or end > session.closes_at:
        return "signal_outside_session"
    ready_at = end + pd.Timedelta(seconds=buffer_seconds)
    if current < ready_at:
        return "incomplete_signal"
    if current >= ready_at + duration:
        return "stale_signal"
    return None


__all__ = [
    "filter_completed_bars",
    "signal_rejection_reason",
]
