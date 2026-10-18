from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from zoneinfo import ZoneInfo
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import pandas as pd

from broker.base import AccountSnapshot, Broker, MarketQuote, OrderResult, OrderSide, PositionSnapshot, TradingSession
from engine.trading_engine import TradingEngine
from engine.trading_state import SymbolTradingState, TradingStateStore
from strategy.strategy import Strategy

NOW = datetime(2026, 8, 28, 14, 35, 10, tzinfo=timezone.utc)


class PresetStrategy(Strategy):
    def prepare_bars(self, bars, **kwargs):
        return bars.copy()


def bars(signal="BUY"):
    return pd.DataFrame({"open": [100.], "high": [101.], "low": [99.],
                         "close": [100.], "volume": [1000], "atr_14": [1.],
                         "signal": [signal]},
                        index=pd.DatetimeIndex([NOW - timedelta(minutes=5, seconds=10)]))


class AsyncBroker(Broker):
    requires_monitoring = True

    def __init__(self):
        self.account = AccountSnapshot(10000, 10000, 10000, "account-a", 10000)
        self.positions = []
        self.orders = {}
        self.submissions = []
        self.cancellations = []
        self.quote = MarketQuote(100., 100., NOW)
        self.open = True
        self.lose_response = False
        self.lookup_error = None

    def get_account(self):
        return self.account

    def get_positions(self):
        return self.positions

    def get_quote(self, symbol):
        return self.quote

    def is_market_open(self):
        return self.open

    def get_trading_session(self, timestamp):
        local = timestamp.astimezone(ZoneInfo("America/New_York"))
        if local.weekday() >= 5:
            return None
        return TradingSession(local.replace(hour=9, minute=30, second=0, microsecond=0),
                              local.replace(hour=16, minute=0, second=0, microsecond=0))

    def get_order(self, client_order_id):
        if self.lookup_error:
            raise self.lookup_error
        return self.orders.get(client_order_id)

    def cancel_order(self, order_id):
        self.cancellations.append(order_id)

    def submit_market_order(self, symbol, side, qty, *, client_order_id=None, **kwargs):
        if client_order_id in self.orders:
            raise AssertionError("Duplicate submission")
        self.submissions.append(client_order_id)
        result = OrderResult(symbol, side, qty, "accepted", order_id=client_order_id,
                             client_order_id=client_order_id)
        self.orders[client_order_id] = result
        if self.lose_response:
            raise TimeoutError("response lost AFTER broker accepted")
        return result

    def update(self, client_id, *, status, filled=0., price=None, sync_position=True):
        order = self.orders[client_id]
        delta = filled - order.filled_qty
        self.orders[client_id] = replace(order, status=status, filled_qty=filled,
                                         fill_price=price, timestamp=NOW)
        if sync_position:
            existing = self.positions[0].qty if self.positions else 0
            quantity = existing + delta * (1 if order.side is OrderSide.BUY else -1)
            if quantity <= 0:
                self.positions = []
            else:
                average = price if order.side is OrderSide.BUY else self.positions[0].avg_entry_price
                self.positions = [PositionSnapshot("SPY", quantity, average, quantity * average)]


class ProcessBroker(AsyncBroker):
    """Filesystem-backed fake, shared by independent spawned processes."""
    def __init__(self, backend):
        super().__init__()
        self.backend = Path(backend)

    def get_order(self, client_order_id):
        data = json.loads(self.backend.read_text()) if self.backend.exists() else {}
        raw = data.get(client_order_id)
        return OrderResult("SPY", OrderSide(raw["side"]), raw["qty"], "accepted",
                           order_id=client_order_id, client_order_id=client_order_id) if raw else None

    def submit_market_order(self, symbol, side, qty, *, client_order_id=None, **kwargs):
        data = json.loads(self.backend.read_text()) if self.backend.exists() else {}
        if client_order_id in data:
            raise AssertionError("duplicate submission")
        data[client_order_id] = {"side": side.value, "qty": qty}
        self.backend.write_text(json.dumps(data))
        return self.get_order(client_order_id)


def run_in_process(state_path, backend, ready, start):
    broker = ProcessBroker(backend)
    engine = TradingEngine(broker, PresetStrategy(), state_store=TradingStateStore(
        "paper", state_path, account_id="account-a"))
    ready.put(True)
    if not start.wait(10):
        raise RuntimeError("Process barrier timed out")
    engine.run_latest("SPY", bars(), execute=True, as_of=NOW)


class LiveExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.json"
        self.broker = AsyncBroker()
        self.store = TradingStateStore("paper", self.path, account_id="account-a")
        self.engine = self.make_engine()

    def make_engine(self):
        return TradingEngine(self.broker, PresetStrategy(), state_store=TradingStateStore(
            "paper", self.path, account_id="account-a"))

    def buy(self):
        return self.engine.run_latest("SPY", bars(), execute=True, as_of=NOW)[1]

    def seed_position(self):
        self.broker.positions = [PositionSnapshot("SPY", 25, 100, 2500)]
        self.store.save("SPY", SymbolTradingState(entry_atr=1, stop_price=98))

    def test_accepted_buy_has_no_fictitious_fill_or_stop(self):
        result = self.buy()
        state = self.store.load("SPY")
        self.assertEqual(result.quantity, 0)
        self.assertIsNone(state.stop_price)
        self.assertIsNone(state.last_trade_bar)
        self.assertIsNotNone(state.pending_order)

    def test_partial_fill_uses_actual_average_and_original_atr_after_restart(self):
        order = self.buy().order
        self.broker.update(order.client_order_id, status="partially_filled", filled=10, price=102)
        self.make_engine().monitor("SPY", execute=True, as_of=NOW)
        state = self.store.load("SPY")
        self.assertEqual(state.stop_price, 100)
        self.assertEqual(state.entry_atr, 1)
        self.assertEqual(state.pending_order.filled_qty, 10)
        self.assertIsNotNone(state.last_trade_bar)

    def test_rejected_buy_does_not_start_cooldown(self):
        order = self.buy().order
        self.broker.update(order.client_order_id, status="rejected")
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        state = self.store.load("SPY")
        self.assertIsNone(state.pending_order)
        self.assertIsNone(state.last_trade_bar)
        self.assertIsNone(state.stop_price)
        self.buy()
        self.assertEqual(len(self.broker.submissions), 1)

    def test_sell_accept_partial_and_reject_keep_protection(self):
        self.seed_position()
        _, decision = self.engine.run_latest("SPY", bars("SELL"), execute=True, as_of=NOW)
        self.assertEqual(self.store.load("SPY").stop_price, 98)
        self.broker.update(decision.order.client_order_id, status="partially_filled", filled=5, price=100)
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(self.store.load("SPY").stop_price, 98)
        self.broker.update(decision.order.client_order_id, status="canceled", filled=5, price=100)
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(self.store.load("SPY").stop_price, 98)
        self.assertEqual(self.broker.positions[0].qty, 20)

    def test_full_sell_waits_for_position_endpoint_before_clearing(self):
        self.seed_position()
        _, result = self.engine.run_latest("SPY", bars("SELL"), execute=True, as_of=NOW)
        self.broker.update(result.order.client_order_id, status="filled", filled=25,
                           price=99, sync_position=False)
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.assertIsNotNone(self.store.load("SPY").pending_order)
        self.assertEqual(self.store.load("SPY").stop_price, 98)
        self.assertEqual(len(self.broker.submissions), 1)
        self.broker.positions = []
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.assertIsNone(self.store.load("SPY").pending_order)
        self.assertIsNone(self.store.load("SPY").stop_price)

    def test_intrabar_stop_ignores_duplicate_bar_and_recovers_after_restart(self):
        self.seed_position()
        self.engine.run_latest("SPY", bars("HOLD"), execute=True, as_of=NOW)
        self.broker.quote = MarketQuote(97, 97.1, NOW)
        _, decision = self.make_engine().run_latest("SPY", bars("HOLD"), execute=True, as_of=NOW)
        self.assertEqual(decision.reason, "atr_stop_loss")
        self.assertEqual(decision.signal.value, "SELL")
        self.assertEqual(len(self.broker.submissions), 1)
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(len(self.broker.submissions), 1)

    def test_risk_monitor_needs_no_bars(self):
        self.seed_position()
        self.broker.quote = MarketQuote(90, 90.1, NOW)
        result = self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(result.reason, "atr_stop_loss")

    def test_stop_cancels_partial_buy_then_sells_only_confirmed_remaining(self):
        order = self.buy().order
        self.broker.update(order.client_order_id, status="partially_filled", filled=10, price=100)
        self.broker.quote = MarketQuote(97, 97.1, NOW)
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(self.broker.cancellations, [order.order_id])
        self.assertEqual(len(self.broker.submissions), 1)
        self.broker.update(order.client_order_id, status="canceled", filled=10, price=100)
        self.broker.quote = MarketQuote(101, 101.1, NOW)  # recovery must not undo exit
        result = self.make_engine().monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(result.signal.value, "SELL")
        self.assertEqual(result.order.qty, 10)
        self.assertEqual(result.reason, "atr_stop_loss")

    def test_rejected_risk_sell_retries_with_new_id_and_keeps_stop(self):
        self.seed_position()
        self.broker.quote = MarketQuote(97, 97.1, NOW)
        order = self.engine.monitor("SPY", execute=True, as_of=NOW).order
        self.broker.update(order.client_order_id, status="rejected")
        self.broker.quote = MarketQuote(101, 101.1, NOW)
        retry = self.make_engine().monitor("SPY", execute=True, as_of=NOW).order
        self.assertNotEqual(retry.client_order_id, order.client_order_id)
        self.assertEqual(self.store.load("SPY").stop_price, 98)

    def test_response_loss_recovers_without_resubmission(self):
        self.broker.lose_response = True
        with self.assertRaises(TimeoutError):
            self.buy()
        self.assertIsNotNone(self.store.load("SPY").pending_order)
        self.make_engine().monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(len(self.broker.submissions), 1)

    def test_crash_after_journal_before_submission_blocks_unknown_outcome(self):
        self.broker.submit_market_order = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("crash"))
        with self.assertRaises(RuntimeError):
            self.buy()
        result = self.make_engine().monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(result.reason, "order_outcome_unknown")
        self.assertEqual(len(self.broker.submissions), 0)

    def test_lookup_error_never_causes_resubmission(self):
        self.buy()
        self.broker.lookup_error = ConnectionError("unavailable")
        with self.assertRaises(ConnectionError):
            self.make_engine().monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(len(self.broker.submissions), 1)

    def test_two_engine_instances_submit_once(self):
        other = self.make_engine()
        barrier = threading.Barrier(2)
        errors = []
        def run(engine):
            try:
                barrier.wait()
                engine.run_latest("SPY", bars(), execute=True, as_of=NOW)
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=run, args=(e,)) for e in (self.engine, other)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(self.broker.submissions), 1)

    def test_independent_processes_submit_once(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Queue()
        start = context.Event()
        backend = str(self.path.parent / "broker.json")
        processes = [context.Process(target=run_in_process, args=(str(self.path), backend, ready, start)) for _ in range(2)]
        try:
            for process in processes:
                process.start()
            for _ in processes:
                self.assertTrue(ready.get(timeout=15))
            start.set()
            for process in processes:
                process.join(timeout=15)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(len(json.loads(Path(backend).read_text())), 1)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            ready.close()
            ready.join_thread()

    def test_failed_journal_write_prevents_submission(self):
        with patch.object(self.engine.state_store, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.buy()
        self.assertEqual(self.broker.submissions, [])

    def test_failed_post_submission_save_recovers_from_intent(self):
        original = self.engine.state_store.save
        calls = 0
        def save(symbol, state):
            nonlocal calls
            if state.pending_order is not None:
                calls += 1
                if calls == 2:
                    raise OSError("disk full after submission")
            return original(symbol, state)
        with patch.object(self.engine.state_store, "save", side_effect=save):
            with self.assertRaises(OSError):
                self.buy()
        self.make_engine().monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(len(self.broker.submissions), 1)
        self.assertEqual(self.store.load("SPY").pending_order.order_id, self.broker.submissions[0])

    def test_utc_midnight_does_not_reset_new_york_daily_halt(self):
        self.broker.account = replace(self.broker.account, equity=9600)
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.broker.account = replace(self.broker.account, equity=10000)
        midnight_utc = datetime(2026, 8, 29, 0, 30, tzinfo=timezone.utc)
        self.make_engine().monitor("SPY", execute=True, as_of=midnight_utc)
        self.assertTrue(self.store.load_risk()["halted"])
        self.assertEqual(self.store.load_risk()["session_date"], "2026-08-28")

    def test_dry_run_reconciles_without_cancelling_or_changing_state(self):
        order = self.buy().order
        self.broker.update(order.client_order_id, status="partially_filled", filled=10, price=100)
        self.broker.quote = MarketQuote(97, 97.1, NOW)
        before = self.path.read_bytes()
        self.engine.monitor("SPY", as_of=NOW)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.broker.cancellations, [])
        self.assertEqual(len(self.broker.submissions), 1)

    def test_daily_loss_survives_restart_and_recovery(self):
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.broker.account = replace(self.broker.account, equity=9600)
        self.make_engine().monitor("SPY", execute=True, as_of=NOW)
        self.broker.account = replace(self.broker.account, equity=10000)
        _, result = self.make_engine().run_latest("SPY", bars(), execute=True, as_of=NOW)
        self.assertEqual(result.reason, "max_daily_loss")
        self.assertTrue(self.store.load_risk()["halted"])
        self.assertEqual(self.store.load_risk()["start_equity"], 10000)

    def test_starting_midday_uses_broker_baseline_not_current_equity(self):
        self.broker.account = replace(self.broker.account, equity=9600)
        result = self.buy()
        self.assertEqual(result.reason, "max_daily_loss")
        self.assertEqual(len(self.broker.submissions), 0)

    def test_daily_exit_still_works_without_a_fresh_quote(self):
        self.seed_position()
        self.broker.account = replace(self.broker.account, equity=9600)
        self.broker.quote = MarketQuote(100, 100, NOW - timedelta(days=1))
        result = self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.assertEqual(result.reason, "max_daily_loss")

    def test_next_new_york_day_resets_daily_halt(self):
        self.broker.account = replace(self.broker.account, equity=9600)
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        self.broker.account = replace(self.broker.account, previous_close_equity=9600)
        self.make_engine().monitor("SPY", execute=True, as_of=NOW + timedelta(days=1))
        self.assertFalse(self.store.load_risk()["halted"])
        self.assertEqual(self.store.load_risk()["start_equity"], 9600)

    def test_dry_run_has_no_state_or_orders(self):
        self.engine.run_latest("SPY", bars(), as_of=NOW)
        self.assertFalse(self.path.exists())
        self.assertEqual(self.broker.submissions, [])

    def test_accounts_and_modes_do_not_share_risk_state(self):
        self.broker.account = replace(self.broker.account, equity=9600)
        self.engine.monitor("SPY", execute=True, as_of=NOW)
        for mode, account_id in [("paper", "account-b"), ("live", "account-a")]:
            store = TradingStateStore(mode, self.path, account_id=account_id)
            self.assertIsNone(store.load_risk())
            self.assertIsNone(store.load("SPY").pending_order)

    def test_closed_market_and_stale_quote_do_not_submit(self):
        self.broker.open = False
        self.assertEqual(self.buy().reason, "market_closed")
        self.broker.open = True
        self.broker.quote = MarketQuote(100, 100, NOW - timedelta(minutes=1))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.buy()
        self.assertEqual(self.broker.submissions, [])

    def test_corrupt_state_fails_closed(self):
        self.path.write_text("")
        with self.assertRaises(json.JSONDecodeError):
            self.buy()
        self.assertEqual(self.broker.submissions, [])

    def test_invalid_atr_never_opens_unprotected_position(self):
        data = bars()
        data["atr_14"] = float("nan")
        _, result = self.engine.run_latest("SPY", data, execute=True, as_of=NOW)
        self.assertEqual(result.reason, "invalid_entry_atr")
        self.assertEqual(self.broker.submissions, [])


if __name__ == "__main__":
    unittest.main()
