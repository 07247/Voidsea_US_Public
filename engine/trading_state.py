"""Durable account-scoped state, protected by a cross-process file lock."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from math import isfinite
import os
from pathlib import Path
import tempfile

from filelock import FileLock

DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / "data/runtime/trading_state.json"


@dataclass
class PendingOrder:
    client_order_id: str
    side: str
    quantity: float
    bar_time: str
    reason: str
    entry_atr: float | None = None
    order_id: str | None = None
    filled_qty: float = 0.0
    cancel_requested: bool = False
    position_qty_before: float = 0.0


@dataclass
class SymbolTradingState:
    last_processed_bar: datetime | None = None
    last_trade_bar: datetime | None = None
    last_trade_time: datetime | None = None
    entry_atr: float | None = None
    stop_price: float | None = None
    pending_order: PendingOrder | None = None
    # A latched exit survives quote recovery, rejection, and process restarts.
    exit_requested: str | None = None
    # Monotonic sequence prevents reuse of an ID after a terminal rejection.
    order_sequence: int = 0


class TradingStateStore:
    def __init__(self, mode: str, path: str | Path = DEFAULT_STATE_PATH,
                 *, account_id: str = "backtest") -> None:
        self.mode = mode.strip().lower()
        if self.mode not in {"paper", "live"}:
            raise ValueError("mode must be 'paper' or 'live'")
        self.account_id = account_id
        self.path = Path(path)
        self._lock = FileLock(str(self.path) + ".lock", timeout=10)

    @property
    def namespace(self) -> str:
        return f"{self.mode}:{self.account_id}"

    @contextmanager
    def transaction(self):
        """Hold across read/reconcile/journal/submit/save, not only file writes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            yield

    def load(self, symbol: str) -> SymbolTradingState:
        with self.transaction():
            data = self._read_all()
            scope = data.get(self.namespace, {})
            # Legacy state has no account identity. Never silently apply it to
            # another account. Operators must explicitly migrate while stopped.
            if not scope and self.account_id != "backtest" and data.get(self.mode):
                raise ValueError(
                    "Legacy account-unscoped state found; migrate it to "
                    f"{self.namespace!r} after verifying the account and positions"
                )
            if self.account_id == "backtest" and not scope:
                scope = data.get(self.mode, {})
            raw = dict(scope.get(symbol.upper(), {}))
            for key in ("last_processed_bar", "last_trade_bar", "last_trade_time"):
                raw[key] = datetime.fromisoformat(raw[key]) if raw.get(key) else None
            if raw.get("pending_order"):
                raw["pending_order"] = PendingOrder(**raw["pending_order"])
            state = SymbolTradingState(**raw)
            for key in ("last_processed_bar", "last_trade_bar", "last_trade_time"):
                value = getattr(state, key)
                if value is not None and value.tzinfo is None:
                    raise ValueError(f"Persisted {key} must be timezone-aware")
            for key in ("entry_atr", "stop_price"):
                value = getattr(state, key)
                if value is not None and (not isfinite(value) or value < 0):
                    raise ValueError(f"Invalid persisted {key}")
            if not isinstance(state.order_sequence, int) or state.order_sequence < 0:
                raise ValueError("Invalid persisted order sequence")
            return state

    def save(self, symbol: str, state: SymbolTradingState) -> None:
        with self.transaction():
            data = self._read_all()
            raw = asdict(state)
            for key in ("last_processed_bar", "last_trade_bar", "last_trade_time"):
                raw[key] = raw[key].isoformat() if raw[key] else None
            data.setdefault(self.namespace, {})[symbol.upper()] = raw
            self._write_all(data)

    def load_risk(self) -> dict | None:
        with self.transaction():
            return self._read_all().get(self.namespace, {}).get("_daily_risk")

    def save_risk(self, state: dict) -> None:
        with self.transaction():
            data = self._read_all()
            data.setdefault(self.namespace, {})["_daily_risk"] = state
            self._write_all(data)

    def _read_all(self) -> dict:
        if not self.path.exists():
            return {}
        # Empty/truncated state is an error, never an invitation to trade anew.
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Trading state file must contain a JSON object")
        return data

    def _write_all(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
            if os.name == "posix":
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)


__all__ = ["PendingOrder", "SymbolTradingState", "TradingStateStore"]
