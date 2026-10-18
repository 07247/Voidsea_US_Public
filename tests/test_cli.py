from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class BacktestCliTests(unittest.TestCase):
    def test_direct_script_execution_resolves_project_packages(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "backtest" / "run_backtest.py"

        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=project_root.parent,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Voidsea US Stocks backtest", completed.stdout)


if __name__ == "__main__":
    unittest.main()
