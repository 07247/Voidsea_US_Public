"""Backtest performance metrics."""

from __future__ import annotations

import pandas as pd


def calculate_backtest_metrics(
    *,
    symbol: str,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_cash: float,
    final_equity: float,
    total_commission: float,
    total_slippage_cost: float,
) -> dict[str, float | int | str]:
    """
    Calculate summary metrics for a completed backtest.
    """

    # ==================================================
    # 1. 总收益率
    # ==================================================

    total_return_pct = (
        final_equity
        / initial_cash
        - 1
    ) * 100

    # ==================================================
    # 2. 最大回撤
    # ==================================================

    if equity.empty:
        max_drawdown_pct = 0.0

    else:
        equity_curve = pd.to_numeric(
            equity["equity"],
            errors="coerce",
        ).dropna()

        if equity_curve.empty:
            max_drawdown_pct = 0.0

        else:
            drawdown = (
                equity_curve
                / equity_curve.cummax()
                - 1
            )

            max_drawdown_pct = float(
                drawdown.min() * 100
            )

    # ==================================================
    # 3. 没有交易时
    # ==================================================

    if trades.empty:
        return {
            "symbol": symbol.upper(),
            "initial_cash": initial_cash,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,

            "trade_count": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,

            "average_trade_return_pct": 0.0,

            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,

            "average_win": 0.0,
            "average_loss": 0.0,
            "payoff_ratio": 0.0,

            "average_win_return_pct": 0.0,
            "average_loss_return_pct": 0.0,

            "average_holding_minutes": 0.0,
            "median_holding_minutes": 0.0,
            "max_holding_minutes": 0.0,

            "max_consecutive_losses": 0,

            "max_drawdown_pct": max_drawdown_pct,

            "total_commission": total_commission,
            "total_slippage_cost": total_slippage_cost,
        }

    # ==================================================
    # 4. 每笔交易 PnL
    # ==================================================

    pnl = pd.to_numeric(
        trades["pnl"],
        errors="coerce",
    ).fillna(0.0)

    returns = pd.to_numeric(
        trades["return_pct"],
        errors="coerce",
    ).fillna(0.0)

    win_mask = pnl > 0
    loss_mask = pnl < 0

    trade_count = len(trades)

    winning_trades = int(
        win_mask.sum()
    )

    losing_trades = int(
        loss_mask.sum()
    )

    win_rate_pct = (
        winning_trades
        / trade_count
        * 100
    )

    average_trade_return_pct = float(
        returns.mean()
    )

    # ==================================================
    # 5. Gross Profit / Gross Loss
    # ==================================================

    gross_profit = float(
        pnl[win_mask].sum()
    )

    gross_loss = abs(
        float(
            pnl[loss_mask].sum()
        )
    )

    # ==================================================
    # 6. Profit Factor
    #
    # 总盈利 ÷ 总亏损
    # ==================================================

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:
        profit_factor = float("inf")

    else:
        profit_factor = 0.0

    # ==================================================
    # 7. 平均盈利 / 平均亏损
    # ==================================================

    average_win = (
        float(
            pnl[win_mask].mean()
        )
        if winning_trades > 0
        else 0.0
    )

    average_loss = (
        float(
            pnl[loss_mask].mean()
        )
        if losing_trades > 0
        else 0.0
    )

    # ==================================================
    # 8. 平均盈利率 / 平均亏损率
    # ==================================================

    average_win_return_pct = (
        float(
            returns[win_mask].mean()
        )
        if winning_trades > 0
        else 0.0
    )

    average_loss_return_pct = (
        float(
            returns[loss_mask].mean()
        )
        if losing_trades > 0
        else 0.0
    )

    # ==================================================
    # 9. 盈亏比
    #
    # 平均盈利 ÷ 平均亏损绝对值
    # ==================================================

    if average_loss < 0:
        payoff_ratio = (
            average_win
            / abs(average_loss)
        )

    elif average_win > 0:
        payoff_ratio = float("inf")

    else:
        payoff_ratio = 0.0

    # ==================================================
    # 10. 持仓时间
    # ==================================================

    holding_minutes = _holding_minutes(
        trades
    )

    if holding_minutes.empty:
        average_holding_minutes = 0.0
        median_holding_minutes = 0.0
        max_holding_minutes = 0.0

    else:
        average_holding_minutes = float(
            holding_minutes.mean()
        )

        median_holding_minutes = float(
            holding_minutes.median()
        )

        max_holding_minutes = float(
            holding_minutes.max()
        )

    # ==================================================
    # 11. 最大连续亏损次数
    # ==================================================

    max_consecutive_losses = (
        _max_consecutive_losses(
            pnl
        )
    )

    # ==================================================
    # 12. 最终结果
    # ==================================================

    return {
        "symbol": symbol.upper(),

        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,

        "trade_count": trade_count,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate_pct,

        "average_trade_return_pct": (
            average_trade_return_pct
        ),

        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,

        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": payoff_ratio,

        "average_win_return_pct": (
            average_win_return_pct
        ),

        "average_loss_return_pct": (
            average_loss_return_pct
        ),

        "average_holding_minutes": (
            average_holding_minutes
        ),

        "median_holding_minutes": (
            median_holding_minutes
        ),

        "max_holding_minutes": (
            max_holding_minutes
        ),

        "max_consecutive_losses": (
            max_consecutive_losses
        ),

        "max_drawdown_pct": (
            max_drawdown_pct
        ),

        "total_commission": (
            total_commission
        ),

        "total_slippage_cost": (
            total_slippage_cost
        ),
    }


def _holding_minutes(
    trades: pd.DataFrame,
) -> pd.Series:
    """
    Calculate holding time for each trade.
    """

    if (
        "entry_time" not in trades.columns
        or "exit_time" not in trades.columns
    ):
        return pd.Series(
            dtype=float
        )

    durations: list[float] = []

    for entry_time, exit_time in zip(
        trades["entry_time"],
        trades["exit_time"],
    ):
        if (
            pd.isna(entry_time)
            or pd.isna(exit_time)
        ):
            continue

        duration = (
            pd.Timestamp(exit_time)
            - pd.Timestamp(entry_time)
        )

        durations.append(
            duration.total_seconds()
            / 60
        )

    return pd.Series(
        durations,
        dtype=float,
    )


def _max_consecutive_losses(
    pnl: pd.Series,
) -> int:
    """
    Calculate the maximum consecutive losing trades.
    """

    current = 0
    maximum = 0

    for value in pnl:
        if float(value) < 0:
            current += 1

            maximum = max(
                maximum,
                current,
            )

        else:
            current = 0

    return maximum


__all__ = [
    "calculate_backtest_metrics",
]