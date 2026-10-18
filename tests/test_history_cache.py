from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from backtest.run_backtest import main
from market_data.history import download_history, latest_history_file


class HistoryCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def cache(self, name):
        path = self.directory / name
        path.write_text("placeholder")
        return path

    def test_selects_end_date_not_start_date_or_filename_order(self):
        self.cache("SPY_5min_20250830_20260830.csv")
        newer = self.cache("SPY_5min_20250815_20260901.csv")
        self.assertEqual(latest_history_file("spy", self.directory), newer)

    def test_filters_symbol_and_configured_timeframe(self):
        chosen = self.cache("SPY_5min_20250801_20260801.csv")
        for name in ["SPY_1min_20250901_20260901.csv", "SPY_15min_20250901_20260901.csv",
                     "SPY_60min_20250901_20260901.csv", "AAPL_5min_20250901_20260901.csv"]:
            self.cache(name)
        self.assertEqual(latest_history_file("SPY", self.directory, timeframe_minutes=5), chosen)
        self.assertEqual(latest_history_file("SPY", self.directory, timeframe_minutes=15).name,
                         "SPY_15min_20250901_20260901.csv")

    def test_equal_end_dates_prefer_longest_coverage(self):
        self.cache("SPY_5min_20260101_20260901.csv")
        longer = self.cache("SPY_5min_20250901_20260901.csv")
        self.assertEqual(latest_history_file("SPY", self.directory), longer)

    def test_minimum_requested_window_is_respected(self):
        self.cache("SPY_5min_20260801_20260901.csv")
        long_enough = self.cache("SPY_5min_20250801_20260801.csv")
        self.assertEqual(latest_history_file("SPY", self.directory, min_days=365), long_enough)
        self.assertIsNone(latest_history_file("SPY", self.directory, min_days=400))

    def test_malformed_dates_reversed_windows_and_directories_are_ignored(self):
        for name in ["SPY_5min_20260230_20260901.csv", "SPY_5min_20260901_20260101.csv",
                     "SPY_5min_20260901_20260901.csv", "SPY_5min_bad_20260901.csv",
                     "SPY_5min_20250101_20290101.csv.bak"]:
            self.cache(name)
        (self.directory / "SPY_5min_20250101_20290901.csv").mkdir()
        valid = self.cache("SPY_5min_20250101_20260101.csv")
        self.assertEqual(latest_history_file("SPY", self.directory), valid)

    def test_missing_and_incompatible_caches_return_none(self):
        self.assertIsNone(latest_history_file("SPY", self.directory / "missing"))
        self.cache("SPY_1min_20250101_20260101.csv")
        self.assertIsNone(latest_history_file("SPY", self.directory))

    def test_symbols_are_matched_literally(self):
        self.cache("BRKxB_5min_20250101_20260101.csv")
        correct = self.cache("BRK.B_5min_20240101_20250101.csv")
        self.assertEqual(latest_history_file("brk.b", self.directory), correct)

    def test_download_request_and_filename_use_the_same_timeframe(self):
        broker = Mock()
        broker.get_bars.return_value = pd.DataFrame({"close": [100]})
        end = datetime(2026, 9, 1, tzinfo=timezone.utc)
        _, path = download_history(broker, "SPY", days=365, end=end,
                                   timeframe_minutes=15, output_dir=self.directory)
        broker.get_bars.assert_called_once_with("SPY", days=365, end=end, timeframe_minutes=15)
        self.assertEqual(path.name, "SPY_15min_20250901_20260901.csv")

    def test_cli_uses_configured_timeframe_and_requested_days_for_lookup(self):
        args = SimpleNamespace(symbol="SPY", data=None, refresh=False, days=365)
        with patch("backtest.run_backtest.parse_args", return_value=args), \
                patch("backtest.run_backtest.TIMEFRAME_MINUTES", 15), \
                patch("backtest.run_backtest.latest_history_file", return_value=Path("cache.csv")) as lookup, \
                patch("backtest.run_backtest.load_history", return_value=pd.DataFrame()), \
                patch("backtest.run_backtest.run_backtest", return_value=({}, pd.DataFrame(), pd.DataFrame())), \
                patch("backtest.run_backtest.save_backtest_csv"), patch("builtins.print"):
            main()
        lookup.assert_called_once_with("SPY", timeframe_minutes=15, min_days=365)

    def test_cli_downloads_when_no_compatible_cache_exists(self):
        args = SimpleNamespace(symbol="SPY", data=None, refresh=False, days=365)
        with patch("backtest.run_backtest.parse_args", return_value=args), \
                patch("backtest.run_backtest.TIMEFRAME_MINUTES", 15), \
                patch("backtest.run_backtest.latest_history_file", return_value=None), \
                patch("backtest.run_backtest.AlpacaBroker") as broker, \
                patch("backtest.run_backtest.download_history", return_value=(pd.DataFrame(), Path("new.csv"))) as download, \
                patch("backtest.run_backtest.run_backtest", return_value=({}, pd.DataFrame(), pd.DataFrame())), \
                patch("backtest.run_backtest.save_backtest_csv"), patch("builtins.print"):
            main()
        download.assert_called_once_with(broker.return_value, "SPY", days=365, timeframe_minutes=15)

    def test_explicit_data_bypasses_automatic_selection(self):
        args = SimpleNamespace(symbol="SPY", data=Path("chosen.csv"), refresh=False, days=365)
        with patch("backtest.run_backtest.parse_args", return_value=args), \
                patch("backtest.run_backtest.latest_history_file") as lookup, \
                patch("backtest.run_backtest.load_history", return_value=pd.DataFrame()) as load, \
                patch("backtest.run_backtest.run_backtest", return_value=({}, pd.DataFrame(), pd.DataFrame())), \
                patch("backtest.run_backtest.save_backtest_csv"), patch("builtins.print"):
            main()
        lookup.assert_not_called()
        load.assert_called_once_with(Path("chosen.csv"))
