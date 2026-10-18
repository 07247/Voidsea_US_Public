"""Download and cache Alpaca bar data without committing large CSV files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd

from broker.alpaca_broker import AlpacaBroker
from config import TIMEFRAME_MINUTES

DEFAULT_HISTORY_DIR = Path(__file__).resolve().parents[1] / "data" / "historical"


def download_history(
    broker: AlpacaBroker,
    symbol: str,
    *,
    days: int,
    output_dir: Path = DEFAULT_HISTORY_DIR,
    end: datetime | None = None,
    timeframe_minutes: int = TIMEFRAME_MINUTES,
) -> tuple[pd.DataFrame, Path]:
    if days < 180:
        raise ValueError("History downloads must cover at least 180 days")
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")
    end = end or datetime.now(timezone.utc)
    bars = broker.get_bars(symbol, days=days, end=end, timeframe_minutes=timeframe_minutes)
    if bars.empty:
        raise RuntimeError(f"Alpaca returned no historical bars for {symbol.upper()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    start_label = (end.date() - pd.Timedelta(days=days)).strftime("%Y%m%d")
    end_label = end.date().strftime("%Y%m%d")
    path = output_dir / (
        f"{symbol.upper()}_{timeframe_minutes}min_{start_label}_{end_label}.csv"
    )
    bars.to_csv(path)
    return bars, path


def load_history(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    data = pd.read_csv(source, parse_dates=["timestamp"])
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Historical CSV is missing columns: {sorted(missing)}")
    if "symbol" in data.columns:
        return data.set_index(["symbol", "timestamp"]).sort_index()
    return data.set_index("timestamp").sort_index()


def latest_history_file(
    symbol: str,
    directory: Path = DEFAULT_HISTORY_DIR,
    *,
    timeframe_minutes: int = TIMEFRAME_MINUTES,
    min_days: int | None = None,
) -> Path | None:
    """Choose the newest compatible download window, never by lexicographic path.

    Coverage is the requested date window encoded by download_history. When
    end dates tie, prefer the earliest start (longest coverage). It does not
    assert that the CSV contains every exchange bar in that window.
    """
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")
    if min_days is not None and min_days <= 0:
        raise ValueError("min_days must be positive")
    directory = Path(directory)
    if not directory.exists():
        return None
    pattern = re.compile(
        rf"{re.escape(symbol.strip().upper())}_{timeframe_minutes}min_"
        r"(?P<start>[0-9]{8})_(?P<end>[0-9]{8})\.csv"
    )
    candidates = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        try:
            start = datetime.strptime(match["start"], "%Y%m%d").date()
            end = datetime.strptime(match["end"], "%Y%m%d").date()
        except ValueError:
            continue
        if start >= end or (min_days is not None and (end - start).days < min_days):
            continue
        candidates.append((end, -start.toordinal(), path.name, path))
    return max(candidates)[-1] if candidates else None


__all__ = [
    "DEFAULT_HISTORY_DIR",
    "download_history",
    "latest_history_file",
    "load_history",
]

