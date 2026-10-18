"""CLI for downloading/reusing history and running the unified engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Direct execution puts ``backtest/`` first on sys.path. Add the repository root
# so project packages resolve identically to ``python -m backtest.run_backtest``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.runner import run_backtest
from backtest.reporting import (
    save_backtest_csv,
)
from broker.alpaca_broker import AlpacaBroker
from config import HISTORICAL_DAYS, SYMBOL, TIMEFRAME_MINUTES, TRADING_MODE
from market_data.history import (
    download_history,
    latest_history_file,
    load_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Voidsea US Stocks backtest")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--days", type=int, default=HISTORICAL_DAYS)
    parser.add_argument("--data", type=Path, help="Use a specific historical CSV")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download a new historical CSV even when a local cache exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    source = args.data
    if source is None and not args.refresh:
        source = latest_history_file(
            symbol, timeframe_minutes=TIMEFRAME_MINUTES, min_days=args.days,
        )

    if source is not None:
        print(f"Loading historical data: {source}")
        bars = load_history(source)
    else:
        print(f"Downloading {args.days} days of {symbol} history...")
        broker = AlpacaBroker(mode=TRADING_MODE)
        bars, source = download_history(
            broker, symbol, days=args.days, timeframe_minutes=TIMEFRAME_MINUTES,
        )
        print(f"Saved {len(bars):,} bars to {source}")

    result, trades, equity = run_backtest(
        bars,
        symbol=symbol,
    )
    print("\n====== BACKTEST RESULT ======")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("\n====== LATEST TRADES ======")
    print(trades.tail(20).to_string(index=False) if not trades.empty else "No trades")

    result_dir = save_backtest_csv(
        symbol=symbol,
        result=result,
        trades=trades,
        equity=equity,
    )

    print(
        f"\nBacktest files saved to: {result_dir}"
    )

if __name__ == "__main__":
    main()

