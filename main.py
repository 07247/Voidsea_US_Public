"""Evaluate once in dry-run mode, or continuously monitor paper/live execution."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import time

from broker.alpaca_broker import AlpacaBroker
from config import BAR_CLOSE_BUFFER_SECONDS, SYMBOL, TIMEFRAME_MINUTES, TRADING_MODE
from engine.trading_engine import TradingEngine
from engine.trading_state import TradingStateStore

LOG = logging.getLogger("voidsea")


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 2 or seconds > 5:
        raise argparse.ArgumentTypeError("poll interval must be in [2, 5] seconds")
    return seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or continuously execute Voidsea")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--mode", choices=("paper", "live"), default=TRADING_MODE)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--execute", action="store_true",
                        help="Continuously monitor risk, reconcile orders and execute signals")
    parser.add_argument("--poll-seconds", type=positive_seconds, default=2.0,
                        help="Risk/order polling interval, defaults to 2 seconds (maximum 5)")
    return parser.parse_args()


def log_decision(decision) -> None:
    if decision.reason not in {"monitoring", "bar_already_processed", "no_action", "market_closed"}:
        LOG.info("%s reason=%s filled_qty=%s order_status=%s", decision.signal.value,
                 decision.reason, decision.quantity, decision.order.status if decision.order else "none")


def run_monitor(engine, broker, symbol, *, days, poll_seconds=2.0,
                stop_requested=lambda: False):
    """Bar downloads run separately so a slow download cannot stop risk polling.

    Order/account/quote errors are logged and retried; pending intents stay
    durable. No ambiguous submission is automatically repeated.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bar-download")
    future = None
    next_bars_at = 0.0
    try:
        while not stop_requested():
            started = time.monotonic()
            try:
                log_decision(engine.monitor(symbol, execute=True))
            except Exception:
                LOG.exception("Risk/order monitoring failed; retrying with durable state")
            if future is not None and future.done():
                try:
                    bars = future.result()
                    _, decision = engine.run_latest(symbol, bars, execute=True)
                    log_decision(decision)
                    interval = TIMEFRAME_MINUTES * 60
                    refresh_delay = interval - time.time() % interval + BAR_CLOSE_BUFFER_SECONDS
                except Exception:
                    LOG.exception("Signal evaluation failed; risk monitoring continues")
                    refresh_delay = 10
                future = None
                next_bars_at = time.monotonic() + refresh_delay
            if future is None and time.monotonic() >= next_bars_at:
                future = pool.submit(broker.get_bars, symbol, days=days)
            time.sleep(max(0, poll_seconds - (time.monotonic() - started)))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    broker = AlpacaBroker(mode=args.mode)
    if args.execute:
        broker.validate_execution()
    account = broker.get_account()
    store = TradingStateStore(args.mode, account_id=account.account_id)
    engine = TradingEngine(broker, state_store=store)
    symbol = args.symbol.upper()
    if not args.execute:
        prepared, decision = engine.run_latest(symbol, broker.get_bars(symbol, days=args.days))
        print(prepared.tail(20).to_string())
        print(f"{decision.signal.value}: {decision.reason}; quantity={decision.quantity}; dry-run")
        return
    LOG.info("Continuous %s execution for %s; quote stops require this process and network", args.mode, symbol)
    try:
        run_monitor(engine, broker, symbol, days=args.days, poll_seconds=args.poll_seconds)
    except KeyboardInterrupt:
        LOG.warning("Monitoring stopped. Open positions no longer have local stop monitoring.")


if __name__ == "__main__":
    main()
