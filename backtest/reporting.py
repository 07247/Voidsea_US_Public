"""Save backtest results to CSV files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "backtest"
    / "results"
)


def save_backtest_csv(
    *,
    symbol: str,
    result: dict[str, float | int | str],
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """
    Save one backtest run into its own folder.

    Files:
        summary.csv
        trades.csv
        equity.csv
    """

    symbol = symbol.upper()

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = (
        output_dir
        / f"{symbol}_{timestamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # 1. Summary
    # ==================================================

    summary = pd.DataFrame(
        [result]
    )

    summary.to_csv(
        run_dir / "summary.csv",
        index=False,
    )

    # ==================================================
    # 2. Trades
    # ==================================================

    trades.to_csv(
        run_dir / "trades.csv",
        index=False,
    )

    # ==================================================
    # 3. Equity Curve
    # ==================================================

    equity_output = (
        equity.reset_index()
    )

    equity_output.to_csv(
        run_dir / "equity.csv",
        index=False,
    )

    return run_dir


__all__ = [
    "save_backtest_csv",
]