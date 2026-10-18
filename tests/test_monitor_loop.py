from concurrent.futures import Future
import unittest
from unittest.mock import Mock, patch

from engine.trading_engine import TradeDecision
from main import run_monitor
from strategy.strategy import Signal


class MonitorLoopTests(unittest.TestCase):
    def test_slow_bar_download_does_not_block_risk_checks(self):
        engine, broker = Mock(), Mock()
        engine.monitor.return_value = TradeDecision(Signal.HOLD, "monitoring")
        pool = Mock()
        pool.submit.return_value = Future()  # never finishes
        with patch("main.ThreadPoolExecutor", return_value=pool), patch("main.time.sleep"):
            run_monitor(engine, broker, "SPY", days=30,
                        stop_requested=lambda: engine.monitor.call_count >= 3)
        self.assertEqual(engine.monitor.call_count, 3)
        pool.submit.assert_called_once_with(broker.get_bars, "SPY", days=30)
        engine.run_latest.assert_not_called()

    def test_bar_download_failure_does_not_end_monitoring(self):
        engine, broker = Mock(), Mock()
        engine.monitor.return_value = TradeDecision(Signal.HOLD, "monitoring")
        future = Future()
        future.set_exception(ConnectionError("history unavailable"))
        pool = Mock()
        pool.submit.return_value = future
        with patch("main.ThreadPoolExecutor", return_value=pool), patch("main.time.sleep"), self.assertLogs("voidsea", "ERROR"):
            run_monitor(engine, broker, "SPY", days=30,
                        stop_requested=lambda: engine.monitor.call_count >= 3)
        self.assertEqual(engine.monitor.call_count, 3)
        engine.run_latest.assert_not_called()

    def test_monitor_exception_is_retried(self):
        engine, broker = Mock(), Mock()
        engine.monitor.side_effect = [ConnectionError("timeout"), TradeDecision(Signal.HOLD, "monitoring")]
        pool = Mock()
        pool.submit.return_value = Future()
        with patch("main.ThreadPoolExecutor", return_value=pool), patch("main.time.sleep"), self.assertLogs("voidsea", "ERROR"):
            run_monitor(engine, broker, "SPY", days=30,
                        stop_requested=lambda: engine.monitor.call_count >= 2)
        self.assertEqual(engine.monitor.call_count, 2)
