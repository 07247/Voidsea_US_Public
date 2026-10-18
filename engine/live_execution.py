"""Restart-safe execution and intrabar risk monitoring for paper/live trading."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from math import isclose, isfinite

import pandas as pd

from broker.base import OrderSide
from engine.trading_state import PendingOrder, SymbolTradingState
from strategy.strategy import Signal
from market_data.bars import signal_rejection_reason


class LiveExecution:
    def __init__(self, engine):
        self.engine = engine
        self.broker = engine.broker
        self.risk = engine.risk_manager
        self.store = engine.state_store
        self._states = {}

    def _decision(self, signal=Signal.HOLD, reason="no_action", quantity=0, order=None):
        from engine.trading_engine import TradeDecision
        return TradeDecision(signal, reason, quantity, order)

    def _save(self, symbol, state, execute):
        if execute:
            if self.store:
                self.store.save(symbol, state)
            else:
                self._states[symbol] = state

    def _save_risk(self, execute):
        if execute and self.store:
            self.store.save_risk(self.risk.session_state())

    def _quote(self, symbol, now):
        quote = self.broker.get_quote(symbol)
        if quote.timestamp.tzinfo is None:
            raise ValueError("Quote timestamp must be timezone-aware")
        observed_at = now or datetime.now(timezone.utc)
        age = (observed_at - quote.timestamp).total_seconds()
        if not -2 <= age <= 10:
            raise ValueError("Quote is stale or in the future; execution blocked")
        if not all(isfinite(x) and x > 0 for x in (quote.bid, quote.ask)) or quote.ask < quote.bid:
            raise ValueError("Invalid market quote; execution blocked")
        return quote

    def run(self, symbol, *, data=None, execute=False, as_of=None):
        symbol = symbol.upper()
        now = pd.Timestamp(as_of or datetime.now(timezone.utc)).to_pydatetime()
        if now.tzinfo is None:
            raise ValueError("Execution time must be timezone-aware")
        if self.broker.requires_monitoring and self.store is None:
            raise ValueError("Paper/live execution requires durable state")
        if execute:
            self.broker.validate_execution()
        with self.store.transaction() if self.store else nullcontext():
            account = self.broker.get_account()
            if self.store and self.broker.requires_monitoring:
                if self.store.account_id != account.account_id:
                    raise ValueError("State store account does not match broker account")
            state = self.store.load(symbol) if self.store else deepcopy(self._states.get(symbol, SymbolTradingState()))
            if self.store:
                self.risk.restore_session(self.store.load_risk())
            if self.broker.requires_monitoring and account.previous_close_equity is None:
                raise ValueError("Broker did not supply the daily equity baseline")
            self.risk.observe_equity(now, account.equity, baseline=account.previous_close_equity)
            self._save_risk(execute)

            # Always reconcile and monitor before checking the completed-bar
            # watermark. A duplicate bar must never suppress a stop or a fill.
            pending_result = None
            if state.pending_order:
                pending_result = self.broker.get_order(state.pending_order.client_order_id)
                if pending_result is not None:
                    self._apply_order(symbol, state, pending_result, now)
                    self._save(symbol, state, execute)

            position = self.broker.get_position(symbol)
            if position is None and state.pending_order is None:
                state.entry_atr = state.stop_price = None
                state.exit_requested = None
                self._save(symbol, state, execute)

            if not self.broker.is_market_open():
                return self._decision(reason="market_closed")
            session = None
            if self.broker.requires_monitoring:
                now = pd.Timestamp(as_of or datetime.now(timezone.utc)).to_pydatetime()
                session = self.broker.get_trading_session(now)
                if session is None or not session.contains(now):
                    return self._decision(reason="market_closed")

            reference = float(data.iloc[-1]["close"]) if data is not None else None
            quote = None
            daily_exit = self.risk.daily_loss_breached(account.equity)
            risk_exit = state.exit_requested
            if daily_exit and state.pending_order and state.pending_order.side == "BUY":
                risk_exit = "max_daily_loss"
            # A latched account-level exit needs no quote to submit a market
            # liquidation. A missing quote must not prevent daily-loss control.
            if position is not None and daily_exit:
                risk_exit = "max_daily_loss"
                reference = position.avg_entry_price
            elif position is not None and not risk_exit:
                if self.broker.requires_monitoring:
                    quote = self._quote(symbol, now if as_of is not None else None)
                    reference = quote.bid
                if reference is not None:
                    risk_exit = self.risk.exit_reason(position, account, price=reference, stop_price=state.stop_price)
                if state.stop_price is None and self.broker.requires_monitoring and state.pending_order is None:
                    raise ValueError("Unmanaged position has no entry ATR/stop; reconcile it before trading")

            # Once crossed, an exit remains requested even if price recovers
            # while a partial entry is being canceled or a sell is rejected.
            if risk_exit:
                state.exit_requested = risk_exit
                self._save(symbol, state, execute)

            if state.pending_order is not None:
                pending = state.pending_order
                if (risk_exit or daily_exit or pending.cancel_requested) and pending.side == "BUY":
                    if execute and pending.order_id and pending_result and not pending_result.terminal:
                        pending.cancel_requested = True
                        self._save(symbol, state, execute)
                        if pending_result.status != "pending_cancel":
                            self.broker.cancel_order(pending.order_id)
                reason = "order_pending" if pending_result else "order_outcome_unknown"
                return self._decision(reason=reason, order=pending_result)

            if position is not None and risk_exit:
                decision = self._decision(Signal.SELL, risk_exit, position.qty)
                bar_time = data.index[-1] if data is not None else now
                return self._submit(symbol, state, decision, bar_time, None, reference, now, execute,
                                    mark_processed=data is not None, session=session, as_of=as_of)

            if data is None:
                return self._decision(reason="monitoring")

            timestamp = data.index[-1]
            if (execute and state.last_processed_bar is not None
                    and pd.Timestamp(timestamp) <= pd.Timestamp(state.last_processed_bar)):
                return self._decision(reason="bar_already_processed")
            self.engine._bars_since_trade = (
                self.engine.cooldown_bars if state.last_trade_bar is None else
                self.engine._count_bars_since_trade(data, state.last_trade_bar)
            )
            signal = self.engine.strategy.latest_signal(data)
            atr = self.engine._clean_number(data.iloc[-1].get(self.engine._atr_column))
            if self.broker.requires_monitoring:
                rejection = signal_rejection_reason(
                    timestamp, now=as_of or datetime.now(timezone.utc), session=session,
                    timeframe_minutes=self.engine.timeframe_minutes,
                    buffer_seconds=self.engine.bar_close_buffer_seconds,
                )
                if rejection:
                    return self._decision(reason=rejection)
                quote = quote or self._quote(symbol, now if as_of is not None else None)
                reference = quote.ask if signal is Signal.BUY else quote.bid
            if signal is Signal.BUY and (atr is None or atr <= 0):
                decision = self._decision(reason="invalid_entry_atr")
            elif signal is Signal.BUY and self.risk.initial_stop_price(
                    entry_price=reference, entry_atr=atr) <= 0:
                decision = self._decision(reason="invalid_stop_distance")
            else:
                decision = self.engine._handle_signal(
                    symbol=symbol, signal=signal, price=reference, atr=atr,
                    stop_price=state.stop_price, timestamp=timestamp, execute=False,
                )
            self._save_risk(execute)
            if decision.signal is Signal.HOLD:
                if execute:
                    state.last_processed_bar = pd.Timestamp(timestamp).to_pydatetime()
                    self._save(symbol, state, execute)
                return decision
            return self._submit(symbol, state, decision, timestamp, atr, reference, now, execute,
                                session=session, as_of=as_of, signal_time=timestamp, quote=quote)

    def _submit(self, symbol, state, decision, bar_time, atr, price, now, execute,
                *, mark_processed=True, session=None, as_of=None, signal_time=None, quote=None):
        if not execute:
            return decision
        position = self.broker.get_position(symbol)
        if self.broker.requires_monitoring:
            # Network calls during evaluation can cross a close/bar boundary.
            # Re-check before writing an intent; a skipped signal is retryable.
            if not self.broker.is_market_open():
                return self._decision(reason="market_closed")
            current = pd.Timestamp(as_of or datetime.now(timezone.utc))
            if session is None or not session.contains(current):
                return self._decision(reason="market_closed")
            if signal_time is not None:
                rejection = signal_rejection_reason(
                    signal_time, now=current, session=session,
                    timeframe_minutes=self.engine.timeframe_minutes,
                    buffer_seconds=self.engine.bar_close_buffer_seconds,
                )
                if rejection:
                    return self._decision(reason=rejection)
                if quote is None or not -2 <= (current - quote.timestamp).total_seconds() <= 10:
                    return self._decision(reason="stale_quote")
        bar_time = pd.Timestamp(bar_time).to_pydatetime()
        if bar_time.tzinfo is None:
            raise ValueError("Order bar timestamp must be timezone-aware")
        scope = self.store.namespace if self.store else "backtest"
        sequence = state.order_sequence + 1
        identity = f"{scope}|{symbol}|{bar_time.isoformat()}|{decision.signal.value}|{sequence}"
        client_id = "voidsea-" + sha256(identity.encode()).hexdigest()[:32]
        state.order_sequence = sequence
        state.pending_order = PendingOrder(
            client_order_id=client_id, side=decision.signal.value,
            quantity=decision.quantity, bar_time=bar_time.isoformat(),
            reason=decision.reason, entry_atr=atr,
            position_qty_before=position.qty if position else 0.0,
        )
        if mark_processed and (state.last_processed_bar is None or bar_time > state.last_processed_bar):
            state.last_processed_bar = bar_time
        # Intent is fsynced BEFORE any network submission. An exception or
        # crash leaves this intent pending. Recovery only queries the ID; it
        # never resubmits an ambiguous request, including a confirmed 404.
        self._save(symbol, state, execute)
        order = self.broker.submit_market_order(
            symbol, OrderSide(decision.signal.value), decision.quantity,
            price=price, timestamp=now, client_order_id=client_id,
        )
        self._apply_order(symbol, state, order, now)
        self._save(symbol, state, execute)
        self._save_risk(execute)
        return self._decision(decision.signal, decision.reason, order.filled_qty, order)

    def _apply_order(self, symbol, state, order, now):
        pending = state.pending_order
        if order.symbol != symbol or order.side.value != pending.side:
            raise ValueError("Broker returned an order for a different symbol/side")
        if order.client_order_id and order.client_order_id != pending.client_order_id:
            raise ValueError("Broker returned a different client order ID")
        if not isfinite(order.filled_qty) or order.filled_qty < pending.filled_qty:
            raise ValueError("Invalid or regressing cumulative filled quantity")
        if order.status == "filled" and (order.filled_qty <= 0 or not isclose(order.filled_qty, order.qty)):
            raise ValueError("Filled order has inconsistent cumulative quantity")
        if order.filled_qty > pending.quantity + 1e-6:
            raise ValueError("Filled quantity exceeds the journaled order")
        pending.order_id = order.order_id or pending.order_id
        if order.filled_qty > pending.filled_qty:
            if order.fill_price is None or not isfinite(order.fill_price) or order.fill_price <= 0:
                raise ValueError("A fill requires a valid actual average fill price")
            state.last_trade_bar = datetime.fromisoformat(pending.bar_time)
            state.last_trade_time = order.timestamp or now
            if pending.side == "BUY":
                state.entry_atr = pending.entry_atr
                state.stop_price = self.risk.initial_stop_price(
                    entry_price=order.fill_price, entry_atr=pending.entry_atr,
                )
                if state.stop_price is None or state.stop_price <= 0:
                    state.exit_requested = "invalid_filled_stop"
            pending.filled_qty = order.filled_qty
        if not order.terminal:
            return
        # Wait for the position endpoint to agree before allowing another
        # order. It may lag a terminal order response, especially a full sell.
        position = self.broker.get_position(symbol)
        actual = position.qty if position else 0.0
        direction = 1 if pending.side == "BUY" else -1
        expected = pending.position_qty_before + direction * order.filled_qty
        if not isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-6):
            return
        state.pending_order = None
        if actual <= 0:
            state.entry_atr = state.stop_price = None
            state.exit_requested = None
