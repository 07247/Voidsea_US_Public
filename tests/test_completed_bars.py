from __future__ import annotations

import unittest

import pandas as pd

from market_data.bars import filter_completed_bars


class CompletedBarsTests(unittest.TestCase):

    def test_incomplete_bar_is_removed(self) -> None:
        index = pd.date_range(
            "2026-08-28 10:20",
            periods=3,
            freq="5min",
            tz="America/New_York",
        )

        bars = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000, 1100, 1200],
            },
            index=index,
        )

        # 当前时间 10:32
        #
        # 10:20 K线：10:25结束 -> 完成
        # 10:25 K线：10:30结束 -> 完成
        # 10:30 K线：10:35结束 -> 尚未完成
        now = pd.Timestamp(
            "2026-08-28 10:32",
            tz="America/New_York",
        )

        result = filter_completed_bars(
            bars,
            timeframe_minutes=5,
            now=now,
            buffer_seconds=0,
        )

        self.assertEqual(
            len(result),
            2,
        )

        self.assertEqual(
            result.index[-1],
            index[1],
        )

    def test_bar_waits_for_close_buffer(self) -> None:
        index = pd.DatetimeIndex(
            [
                pd.Timestamp(
                    "2026-08-28 10:30",
                    tz="America/New_York",
                )
            ]
        )

        bars = pd.DataFrame(
            {
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100.5],
                "volume": [1000],
            },
            index=index,
        )

        # 10:35:03
        #
        # K线理论上已经结束，
        # 但设置了5秒缓冲，所以还不能使用。
        now = pd.Timestamp(
            "2026-08-28 10:35:03",
            tz="America/New_York",
        )

        with self.assertRaises(ValueError):
            filter_completed_bars(
                bars,
                timeframe_minutes=5,
                now=now,
                buffer_seconds=5,
            )

    def test_bar_available_after_buffer(self) -> None:
        index = pd.DatetimeIndex(
            [
                pd.Timestamp(
                    "2026-08-28 10:30",
                    tz="America/New_York",
                )
            ]
        )

        bars = pd.DataFrame(
            {
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100.5],
                "volume": [1000],
            },
            index=index,
        )

        # 10:35:06
        now = pd.Timestamp(
            "2026-08-28 10:35:06",
            tz="America/New_York",
        )

        result = filter_completed_bars(
            bars,
            timeframe_minutes=5,
            now=now,
            buffer_seconds=5,
        )

        self.assertEqual(
            len(result),
            1,
        )


if __name__ == "__main__":
    unittest.main()