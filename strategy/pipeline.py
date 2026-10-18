"""Compatibility wrapper for callers that still import ``run_strategy``."""

from __future__ import annotations

import pandas as pd

from strategy.strategy import Strategy


def run_strategy(bars: pd.DataFrame) -> pd.DataFrame:
    return Strategy().prepare_bars(bars)
