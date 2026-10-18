from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide as AlpacaSide, OrderStatus
from alpaca.trading.models import Calendar
from requests import HTTPError, Response

from broker.alpaca_broker import AlpacaBroker, _TradingClient
from broker.base import OrderSide


class AlpacaExecutionTests(unittest.TestCase):
    def broker(self, mode="paper", allow=True):
        broker = AlpacaBroker(mode=mode, api_key="test-key", secret_key="test-secret", allow_live_trading=allow)
        broker.trading_client = Mock()
        return broker

    def order(self):
        return SimpleNamespace(symbol="SPY", side=AlpacaSide.BUY, qty="25", status=OrderStatus.PARTIALLY_FILLED,
                               filled_qty="10", filled_avg_price="102", filled_at=None,
                               id="order-123", client_order_id="voidsea-test")

    def test_submission_passes_client_id_and_uses_cumulative_fills(self):
        broker = self.broker()
        broker.trading_client.submit_order.return_value = self.order()
        result = broker.submit_market_order("spy", OrderSide.BUY, 25, client_order_id="voidsea-test")
        request = broker.trading_client.submit_order.call_args.kwargs["order_data"]
        self.assertEqual(request.client_order_id, "voidsea-test")
        self.assertEqual(result.filled_qty, 10)
        self.assertEqual(result.qty, 25)
        self.assertEqual(result.fill_price, 102)
        self.assertEqual(result.status, "partially_filled")

    def test_lookup_only_treats_404_as_missing(self):
        broker = self.broker()
        for status in (404, 401, 429, 500):
            response = Response()
            response.status_code = status
            broker.trading_client.get_order_by_client_id.side_effect = APIError(
                '{"message":"error"}', HTTPError(response=response))
            if status == 404:
                self.assertIsNone(broker.get_order("client-id"))
            else:
                with self.assertRaises(APIError):
                    broker.get_order("client-id")

    def test_reconciliation_uses_client_id(self):
        broker = self.broker()
        broker.trading_client.get_order_by_client_id.return_value = self.order()
        self.assertEqual(broker.get_order("voidsea-test").filled_qty, 10)
        broker.trading_client.get_order_by_client_id.assert_called_once_with("voidsea-test")

    def test_live_guard_applies_to_submit_and_cancel(self):
        broker = self.broker("live", allow=False)
        with self.assertRaises(PermissionError):
            broker.submit_market_order("SPY", OrderSide.BUY, 1)
        with self.assertRaises(PermissionError):
            broker.cancel_order("order-123")
        broker.trading_client.submit_order.assert_not_called()
        broker.trading_client.cancel_order_by_id.assert_not_called()

    def test_account_identity_and_previous_close_equity_are_preserved(self):
        broker = self.broker()
        broker.trading_client.get_account.return_value = SimpleNamespace(
            cash="1000", equity="9600", buying_power="1000", id="account-a", last_equity="10000")
        account = broker.get_account()
        self.assertEqual(account.account_id, "account-a")
        self.assertEqual(account.previous_close_equity, 10000)

    def test_latest_quote_uses_separate_data_client(self):
        broker = self.broker()
        broker.quote_client = Mock()
        now = datetime.now(timezone.utc)
        broker.quote_client.get_stock_latest_quote.return_value = {
            "SPY": SimpleNamespace(bid_price=97, ask_price=98, timestamp=now)}
        quote = broker.get_quote("spy")
        self.assertEqual(quote.bid, 97)
        self.assertEqual(quote.timestamp, now)

    def test_calendar_uses_new_york_timezone_in_summer_and_winter(self):
        for day, utc_hour in [("2026-08-28", 13), ("2026-12-01", 14)]:
            with self.subTest(day=day):
                broker = self.broker()
                broker.trading_client.get_calendar.return_value = [Calendar(date=day, open="09:30", close="16:00")]
                at = datetime.fromisoformat(day + "T16:00:00+00:00")
                session = broker.get_trading_session(at)
                self.assertEqual(session.opens_at.astimezone(timezone.utc).hour, utc_hour)
                self.assertEqual(session.opens_at.tzinfo, ZoneInfo("America/New_York"))

    def test_calendar_honors_early_close_and_caches_the_day(self):
        broker = self.broker()
        broker.trading_client.get_calendar.return_value = [Calendar(date="2026-11-27", open="09:30", close="13:00")]
        at = datetime(2026, 11, 27, 17, tzinfo=timezone.utc)
        session = broker.get_trading_session(at)
        self.assertEqual(session.closes_at.hour, 13)
        self.assertTrue(session.contains(at))
        self.assertFalse(session.contains(datetime(2026, 11, 27, 18, tzinfo=timezone.utc)))
        self.assertEqual(broker.get_trading_session(at), session)
        broker.trading_client.get_calendar.assert_called_once()

    def test_empty_calendar_means_closed_and_next_date_is_fetched(self):
        broker = self.broker()
        broker.trading_client.get_calendar.side_effect = [[], [Calendar(date="2026-12-28", open="09:30", close="16:00")]]
        holiday = datetime(2026, 12, 25, 16, tzinfo=timezone.utc)
        self.assertIsNone(broker.get_trading_session(holiday))
        self.assertIsNone(broker.get_trading_session(holiday))
        self.assertIsNotNone(broker.get_trading_session(datetime(2026, 12, 28, 16, tzinfo=timezone.utc)))
        self.assertEqual(broker.trading_client.get_calendar.call_count, 2)

    def test_calendar_lookup_uses_new_york_date_at_utc_midnight(self):
        broker = self.broker()
        broker.trading_client.get_calendar.return_value = [Calendar(date="2026-08-28", open="09:30", close="16:00")]
        broker.get_trading_session(datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc))
        request = broker.trading_client.get_calendar.call_args.args[0]
        self.assertEqual(request.start.isoformat(), "2026-08-28")
        self.assertEqual(request.end, request.start)

    def test_calendar_failure_propagates_and_is_not_cached_as_closed(self):
        broker = self.broker()
        broker.trading_client.get_calendar.side_effect = [ConnectionError("calendar unavailable"), []]
        now = datetime(2026, 8, 28, 15, tzinfo=timezone.utc)
        with self.assertRaises(ConnectionError):
            broker.get_trading_session(now)
        self.assertIsNone(broker.get_trading_session(now))
        self.assertEqual(broker.trading_client.get_calendar.call_count, 2)

    def test_wrong_calendar_date_is_rejected(self):
        broker = self.broker()
        broker.trading_client.get_calendar.return_value = [Calendar(date="2026-08-27", open="09:30", close="16:00")]
        with self.assertRaisesRegex(ValueError, "unexpected"):
            broker.get_trading_session(datetime(2026, 8, 28, 15, tzinfo=timezone.utc))

    def test_sdk_transport_has_timeout_and_no_automatic_resubmission(self):
        client = _TradingClient("key", "secret", paper=True)
        response = Mock(status_code=429, text='{"message":"rate limited"}')
        response.raise_for_status.side_effect = HTTPError(response=response)
        client._session = Mock()
        client._session.request.return_value = response
        with self.assertRaises(APIError):
            client.post("/orders", {"client_order_id": "voidsea-test"})
        self.assertEqual(client._session.request.call_count, 1)
        self.assertEqual(client._session.request.call_args.kwargs["timeout"], (3, 5))


if __name__ == "__main__":
    unittest.main()
