"""Position sizing and account-level risk limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo
from math import isfinite

from broker.base import AccountSnapshot, PositionSnapshot
from config import (
    ATR_STOP_MULTIPLIER,
    MAX_DAILY_LOSS_PCT,
    MAX_POSITION_PCT,
    RISK_PER_TRADE_PCT,
)


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = RISK_PER_TRADE_PCT
    max_position_pct: float = MAX_POSITION_PCT
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT
    atr_stop_multiplier: float = ATR_STOP_MULTIPLIER

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade_pct <= 1:
            raise ValueError("risk_per_trade_pct must be in (0, 1]")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        if not 0 < self.max_daily_loss_pct <= 1:
            raise ValueError("max_daily_loss_pct must be in (0, 1]")
        if self.atr_stop_multiplier <= 0:
            raise ValueError("atr_stop_multiplier must be positive")


class RiskManager:
    def __init__(self, settings: RiskConfig | None = None) -> None:
        self.settings = settings or RiskConfig()
        self._session_date = None
        self._session_start_equity: float | None = None
        self._daily_halted = False

    def restore_session(self, state: dict | None) -> None:
        if state is None:
            return
        self._session_date = date.fromisoformat(state["session_date"])
        baseline = float(state["start_equity"])
        if not isfinite(baseline) or baseline <= 0:
            raise ValueError("Invalid persisted daily equity baseline")
        self._session_start_equity = baseline
        self._daily_halted = bool(state["halted"])

    def session_state(self) -> dict:
        return {
            "session_date": self._session_date.isoformat(),
            "start_equity": self._session_start_equity,
            "halted": self._daily_halted,
        }

    def observe_equity(self, timestamp: datetime, equity: float,
                       *, baseline: float | None = None) -> None:
        if timestamp.tzinfo is None:
            raise ValueError("Risk observation must be timezone-aware")
        session_date = timestamp.astimezone(ZoneInfo("America/New_York")).date()
        if not isfinite(equity) or equity < 0:
            raise ValueError("Invalid account equity")
        if session_date != self._session_date:
            start = float(equity if baseline is None else baseline)
            if not isfinite(start) or start <= 0:
                raise ValueError("Daily equity baseline must be finite and positive")
            self._session_date = session_date
            self._session_start_equity = start
            self._daily_halted = False
        self.daily_loss_breached(equity)

    def daily_loss_breached(self, equity: float) -> bool:
        if self._session_start_equity is not None:
            loss = 1 - float(equity) / self._session_start_equity
            self._daily_halted |= loss >= self.settings.max_daily_loss_pct
        return self._daily_halted

    def calculate_position_size(
        self,
        account: AccountSnapshot,
        *,
        price: float,
        atr: float | None,
    ) -> float:
        """Size an entry by both ATR risk and maximum portfolio allocation."""

        if price <= 0 or account.equity <= 0 or account.buying_power <= 0:
            return 0.0
        allocation_qty = account.equity * self.settings.max_position_pct / price
        buying_power_qty = account.buying_power / price
        quantities = [allocation_qty, buying_power_qty]
        if atr is not None and isfinite(atr) and atr > 0:
            risk_budget = account.equity * self.settings.risk_per_trade_pct
            stop_distance = atr * self.settings.atr_stop_multiplier
            quantities.append(risk_budget / stop_distance)
        return max(0.0, min(quantities))

    def entry_rejection_reason(self, account: AccountSnapshot) -> str | None:
        if self.daily_loss_breached(account.equity):
            return "max_daily_loss"
        if account.buying_power <= 0:
            return "no_buying_power"
        return None

    def initial_stop_price(
            self,
            *,
            entry_price: float,
            entry_atr: float | None,
    ) -> float | None:
        """
        根据实际入场价格和入场时 ATR
        计算固定的初始止损价。
        """

        if entry_price <= 0:
            raise ValueError(
                "entry_price must be positive"
            )

        if (
                entry_atr is None
                or not isfinite(entry_atr)
                or entry_atr <= 0
        ):
            return None

        stop_price = (
                entry_price
                - entry_atr
                * self.settings.atr_stop_multiplier
        )

        return max(
            0.0,
            stop_price,
        )

    def exit_reason(
            self,
            position: PositionSnapshot,
            account: AccountSnapshot,
            *,
            price: float,
            stop_price: float | None,
    ) -> str | None:
        del position

        if self.daily_loss_breached(
                account.equity
        ):
            return "max_daily_loss"

        if (
                stop_price is not None
                and isfinite(stop_price)
                and stop_price > 0
                and price <= stop_price
        ):
            return "atr_stop_loss"

        return None

__all__ = ["RiskConfig", "RiskManager"]

