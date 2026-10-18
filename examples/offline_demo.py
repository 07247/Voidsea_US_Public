"""Run a reproducible backtest on synthetic prices, without network or keys."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from backtest.runner import run_backtest

def main():
    index = pd.date_range("2026-01-05 09:30", periods=78, freq="5min", tz="America/New_York")
    close = 100 + 3 * np.sin(np.arange(len(index)) / 6)
    bars = pd.DataFrame({"open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close,
                         "volume": 1000}, index=index)
    result, _, _ = run_backtest(bars, symbol="DEMO")
    print("Synthetic demonstration only; not historical investment performance.")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
