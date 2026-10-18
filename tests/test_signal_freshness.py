from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from broker.base import MarketQuote, PositionSnapshot, TradingSession
from engine.trading_engine import TradingEngine
from engine.trading_state import SymbolTradingState, TradingStateStore
from market_data.bars import signal_rejection_reason
from test_live_execution import AsyncBroker, PresetStrategy, NOW, bars

NY = ZoneInfo("America/New_York")


class SignalFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.broker = AsyncBroker()
        self.store = TradingStateStore("paper", Path(self.temp.name) / "state.json", account_id="account-a")
        self.engine = TradingEngine(self.broker, PresetStrategy(), state_store=self.store)

    def evaluate(self, now, data=None):
        self.broker.quote = MarketQuote(100, 100, now)
        return self.engine.run_latest("SPY", bars() if data is None else data, execute=True, as_of=now)[1]

    def test_previous_bar_expires_exactly_when_next_bar_is_ready(self):
        result = self.evaluate(NOW + timedelta(minutes=5, seconds=-5))
        self.assertEqual(result.reason, "stale_signal")
        self.assertEqual(self.broker.submissions, [])
        self.assertIsNone(self.store.load("SPY").last_processed_bar)

    def test_current_latest_bar_remains_eligible_before_boundary(self):
        result = self.evaluate(NOW + timedelta(minutes=5, seconds=-6))
        self.assertEqual(result.signal.value, "BUY")
        self.assertEqual(len(self.broker.submissions), 1)

    def test_previous_session_is_rejected_even_with_a_fresh_quote(self):
        monday = datetime(2026, 8, 31, 9, 36, tzinfo=NY)
        self.assertEqual(self.evaluate(monday).reason, "signal_outside_session")
        self.assertEqual(self.broker.submissions, [])

    def test_holiday_from_calendar_blocks_even_if_clock_claims_open(self):
        self.broker.get_trading_session = Mock(return_value=None)
        self.assertEqual(self.evaluate(NOW).reason, "market_closed")
        self.assertEqual(self.broker.submissions, [])

    def test_weekend_blocks_even_if_clock_claims_open(self):
        saturday = datetime(2026, 8, 29, 10, 35, 10, tzinfo=NY)
        self.assertEqual(self.evaluate(saturday).reason, "market_closed")

    def test_early_close_and_preopen_are_respected(self):
        session = TradingSession(datetime(2026, 11, 27, 9, 30, tzinfo=NY),
                                 datetime(2026, 11, 27, 13, tzinfo=NY))
        self.broker.get_trading_session = Mock(return_value=session)
        for now in (session.opens_at - timedelta(seconds=1), session.closes_at,
                    session.closes_at + timedelta(minutes=30)):
            with self.subTest(now=now):
                self.assertEqual(self.evaluate(now).reason, "market_closed")
        self.assertEqual(self.broker.submissions, [])

    def test_clock_is_rechecked_before_order_intent_is_written(self):
        self.broker.is_market_open = Mock(side_effect=[True, False])
        self.assertEqual(self.evaluate(NOW).reason, "market_closed")
        self.assertEqual(self.broker.submissions, [])
        self.assertIsNone(self.store.load("SPY").pending_order)
        self.assertEqual(self.store.load("SPY").order_sequence, 0)

    def test_bar_expiring_during_account_request_is_not_submitted(self):
        self.assert_delayed_execution_rejected(NOW + timedelta(minutes=5, seconds=-5), "stale_signal")

    def test_quote_expiring_during_account_request_is_not_submitted(self):
        self.assert_delayed_execution_rejected(NOW + timedelta(seconds=11), "stale_quote")

    def assert_delayed_execution_rejected(self, delayed_time, expected_reason):
        clock_time = [NOW]
        original = self.broker.get_account
        calls = 0
        def delayed_account():
            nonlocal calls
            calls += 1
            if calls == 2:
                clock_time[0] = delayed_time
            return original()
        self.broker.get_account = delayed_account
        with patch("engine.live_execution.datetime") as clock:
            clock.now.side_effect = lambda tz: clock_time[0]
            _, result = self.engine.run_latest("SPY", bars(), execute=True)
        self.assertEqual(result.reason, expected_reason)
        self.assertEqual(self.broker.submissions, [])
        self.assertIsNone(self.store.load("SPY").pending_order)

    def test_stale_signal_does_not_suppress_a_risk_exit(self):
        self.broker.positions = [PositionSnapshot("SPY", 10, 100, 970)]
        self.store.save("SPY", SymbolTradingState(entry_atr=1, stop_price=98))
        self.broker.quote = MarketQuote(97, 97.1, NOW)
        stale = bars()
        stale.index = stale.index - timedelta(minutes=30)
        _, result = self.engine.run_latest("SPY", stale, execute=True, as_of=NOW)
        self.assertEqual(result.reason, "atr_stop_loss")
        self.assertEqual(result.signal.value, "SELL")

    def test_bar_close_buffer_and_future_bar_boundaries(self):
        session = self.broker.get_trading_session(NOW)
        start = bars().index[0]
        self.assertEqual(signal_rejection_reason(start, now=NOW - timedelta(seconds=6), session=session),
                         "incomplete_signal")
        self.assertIsNone(signal_rejection_reason(start, now=NOW - timedelta(seconds=5), session=session))
        self.assertEqual(signal_rejection_reason(NOW, now=NOW, session=session), "incomplete_signal")

    def test_naive_signal_and_bar_crossing_early_close_are_rejected(self):
        session = TradingSession(datetime(2026, 11, 27, 9, 30, tzinfo=NY),
                                 datetime(2026, 11, 27, 13, tzinfo=NY))
        now = session.closes_at - timedelta(seconds=1)
        self.assertEqual(signal_rejection_reason(now.replace(tzinfo=None), now=now, session=session),
                         "invalid_signal_time")
        self.assertEqual(signal_rejection_reason(now - timedelta(minutes=1), now=now, session=session),
                         "signal_outside_session")
