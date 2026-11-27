"""Event-driven, strictly causal crypto-perpetuals portfolio backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quantcritic.config import INTERVAL_TIMEDELTA_HOURS, BacktestConfig
from quantcritic.data.loader import MarketData
from quantcritic.engine.metrics import compute_metrics
from quantcritic.risk.sizing import fixed_fractional_size

if TYPE_CHECKING:
    from quantcritic.strategies.base import Strategy

TRADE_COLUMNS = [
    "symbol",
    "side",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "qty",
    "risk_amount",
    "R",
    "gross_pnl",
    "fees",
    "funding",
    "net_pnl",
    "bars_held",
]


@dataclass
class _OpenPosition:
    symbol_idx: int
    symbol: str
    direction: int
    entry_time: pd.Timestamp
    entry_global_idx: int
    entry_price: float
    stop_price: float
    tp_price: float
    max_bars: int
    qty: float
    risk_amount: float
    entry_fee: float
    accrued_funding: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestResult:
    """Output bundle from running a portfolio backtest."""

    trades: pd.DataFrame
    equity_curve: pd.Series
    exposure_series: pd.Series
    metrics: dict[str, Any]
    config: BacktestConfig
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PreparedSymbol:
    symbol: str
    index: pd.Index
    bar_map: np.ndarray
    last_close_global: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    bar_funding_sum: np.ndarray


@dataclass
class PreparedMarket:
    """Pre-aligned market arrays reused across parameter sweeps and cost stress grids."""

    all_times: pd.DatetimeIndex
    time_list: list[pd.Timestamp]
    day_end_indices: np.ndarray
    symbols: list[_PreparedSymbol]


@dataclass
class _CompiledSymbolData:
    symbol: str
    bar_map: np.ndarray
    last_close_global: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    signals: np.ndarray
    stop_dists: np.ndarray
    tp_rs: np.ndarray
    max_bars: np.ndarray
    exit_signals: np.ndarray
    bar_funding_sum: np.ndarray


def _precompute_bar_funding(
    bar_times: pd.DatetimeIndex,
    funding_series: pd.Series | None,
    interval: str,
) -> np.ndarray:
    """Sum funding rates strictly inside (bar_open, next_bar_open] for each local bar."""
    n = len(bar_times)
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or funding_series is None or funding_series.empty:
        return out

    bar_delta_ns = pd.Timedelta(hours=INTERVAL_TIMEDELTA_HOURS[interval]).value
    bar_ns = pd.DatetimeIndex(bar_times).as_unit("ns").asi8
    next_ns = np.empty(n, dtype=np.int64)
    if n > 1:
        next_ns[:-1] = bar_ns[1:]
    next_ns[-1] = bar_ns[-1] + bar_delta_ns

    f_sorted = funding_series.sort_index()
    f_ns = pd.DatetimeIndex(f_sorted.index).as_unit("ns").asi8
    f_vals = f_sorted.to_numpy(dtype=np.float64)
    cum = np.empty(len(f_vals) + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(f_vals, out=cum[1:])

    start_idx = np.searchsorted(f_ns, bar_ns, side="right")
    end_idx = np.searchsorted(f_ns, next_ns, side="right")
    out[:] = cum[end_idx] - cum[start_idx]
    return out


def prepare_market(data: MarketData) -> PreparedMarket:
    """Pre-align multi-symbol OHLCV and funding arrays onto a unified UTC timeline."""
    non_empty = [sym for sym, df in data.ohlcv.items() if not df.empty]
    if not non_empty:
        empty_idx = pd.DatetimeIndex([], tz="UTC").as_unit("ns")
        return PreparedMarket(
            all_times=empty_idx,
            time_list=[],
            day_end_indices=np.array([], dtype=np.int32),
            symbols=[],
        )

    all_ns = np.unique(
        np.concatenate([pd.DatetimeIndex(data.ohlcv[s].index).as_unit("ns").asi8 for s in non_empty])
    )
    all_times = pd.DatetimeIndex(pd.to_datetime(all_ns, unit="ns", utc=True)).as_unit("ns")
    n_global = len(all_times)
    g_arange = np.arange(n_global, dtype=np.int32)

    prep_syms: list[_PreparedSymbol] = []
    for sym in non_empty:
        df = data.ohlcv[sym]
        local_ns = pd.DatetimeIndex(df.index).as_unit("ns").asi8
        global_positions = np.searchsorted(all_ns, local_ns)
        bar_map = np.full(n_global, -1, dtype=np.int32)
        bar_map[global_positions] = np.arange(len(df), dtype=np.int32)

        opens = df["open"].to_numpy(dtype=np.float64)
        highs = df["high"].to_numpy(dtype=np.float64)
        lows = df["low"].to_numpy(dtype=np.float64)
        closes = df["close"].to_numpy(dtype=np.float64)

        # Pure NumPy forward-fill of last known close onto global timeline
        idx_arr = np.where(bar_map >= 0, g_arange, -1)
        last_valid_g = np.maximum.accumulate(idx_arr)
        safe_g = np.maximum(last_valid_g, 0)
        last_close = np.where(last_valid_g >= 0, closes[bar_map[safe_g]], float(opens[0]))

        bar_funding_sum = _precompute_bar_funding(df.index, data.funding.get(sym), data.interval)
        prep_syms.append(
            _PreparedSymbol(
                symbol=sym,
                index=df.index,
                bar_map=bar_map,
                last_close_global=last_close,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                bar_funding_sum=bar_funding_sum,
            )
        )

    days = all_ns // 86_400_000_000_000
    day_changes = np.flatnonzero(np.diff(days) > 0)
    day_end_indices = np.append(day_changes, n_global - 1).astype(np.int32)
    return PreparedMarket(
        all_times=all_times,
        time_list=list(all_times),
        day_end_indices=day_end_indices,
        symbols=prep_syms,
    )


def _as_float_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.to_numpy(dtype=np.float64, copy=False)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _compile_with_signals(
    prepared: PreparedMarket,
    signals_map: dict[str, pd.DataFrame],
) -> list[_CompiledSymbolData]:
    """Attach strategy signal arrays to pre-aligned market symbol arrays."""
    compiled: list[_CompiledSymbolData] = []
    for ps in prepared.symbols:
        sig_df = signals_map[ps.symbol]
        if not sig_df.index.equals(ps.index):
            sig_df = sig_df.reindex(ps.index)

        n_loc = len(ps.opens)
        sig_ser = sig_df["signal"]
        if pd.api.types.is_numeric_dtype(sig_ser.dtype):
            sig_arr = sig_ser.to_numpy()
            if np.isnan(sig_arr).any():
                sig_raw = np.nan_to_num(sig_arr, nan=0.0).astype(np.int8)
            else:
                sig_raw = sig_arr.astype(np.int8, copy=False)
        else:
            sig_raw = pd.to_numeric(sig_ser, errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        sig_raw = np.clip(sig_raw, -1, 1)

        stop_dists = _as_float_array(sig_df["stop_dist"])
        tp_rs = _as_float_array(sig_df["tp_r"]) if "tp_r" in sig_df.columns else np.full(n_loc, np.nan, dtype=np.float64)

        if "max_bars" in sig_df.columns:
            mb_ser = sig_df["max_bars"]
            if pd.api.types.is_numeric_dtype(mb_ser.dtype):
                mb_arr = mb_ser.to_numpy()
                max_bars = np.nan_to_num(mb_arr, nan=0.0).astype(np.int32) if np.isnan(mb_arr).any() else mb_arr.astype(np.int32, copy=False)
            else:
                max_bars = pd.to_numeric(mb_ser, errors="coerce").fillna(0).to_numpy(dtype=np.int32)
        else:
            max_bars = np.zeros(n_loc, dtype=np.int32)

        if "exit_signal" in sig_df.columns:
            ex_ser = sig_df["exit_signal"]
            if pd.api.types.is_bool_dtype(ex_ser.dtype):
                exit_signals = ex_ser.to_numpy(dtype=bool, copy=False)
            else:
                exit_signals = ex_ser.fillna(False).to_numpy(dtype=bool)
        else:
            exit_signals = np.zeros(n_loc, dtype=bool)

        compiled.append(
            _CompiledSymbolData(
                symbol=ps.symbol,
                bar_map=ps.bar_map,
                last_close_global=ps.last_close_global,
                opens=ps.opens,
                highs=ps.highs,
                lows=ps.lows,
                closes=ps.closes,
                signals=sig_raw,
                stop_dists=stop_dists,
                tp_rs=tp_rs,
                max_bars=max_bars,
                exit_signals=exit_signals,
                bar_funding_sum=ps.bar_funding_sum,
            )
        )
    return compiled


def _fast_daily_sharpe(equity_vals: np.ndarray, times_ns: np.ndarray, initial_capital: float) -> float:
    """Annualized (sqrt 365) Sharpe of UTC end-of-day equity, day 0 anchored at initial capital."""
    if len(equity_vals) == 0:
        return 0.0
    days = times_ns // 86_400_000_000_000
    day_ends = np.append(np.flatnonzero(np.diff(days) > 0), len(days) - 1)
    d_vals = np.concatenate([[float(initial_capital)], equity_vals[day_ends]])
    prev_d = np.where(d_vals[:-1] > 0, d_vals[:-1], np.nan)
    d_rets = (d_vals[1:] - prev_d) / prev_d
    d_rets = d_rets[np.isfinite(d_rets)]
    if len(d_rets) >= 2 and np.std(d_rets, ddof=1) > 1e-12:
        return float((np.mean(d_rets) / np.std(d_rets, ddof=1)) * np.sqrt(365.0))
    return 0.0


def run_backtest(
    data: MarketData,
    strategy_or_signals: "Strategy | dict[str, pd.DataFrame]",
    params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    prepared_market: PreparedMarket | None = None,
    fast_metrics: bool = False,
    trade_start: pd.Timestamp | None = None,
) -> BacktestResult:
    """Run an event-driven multi-symbol perpetuals portfolio backtest.

    `trade_start` evaluates a window inside a longer history: signals are computed on the whole
    history (indicator warm-up), but entries are only allowed at bars >= `trade_start`, and the
    returned equity curve / exposure / metrics cover only bars >= `trade_start`. The equity before
    the window is flat at `initial_capital` because no position can exist before it.
    """
    cfg = config or BacktestConfig(interval=data.interval)
    used_params = dict(params or {})

    if isinstance(strategy_or_signals, dict):
        signals_map = strategy_or_signals
    else:
        if not used_params:
            used_params = dict(strategy_or_signals.default_params)
        signals_map = {
            sym: strategy_or_signals.generate(df, used_params)
            for sym, df in data.ohlcv.items()
        }

    prepared = prepared_market or prepare_market(data)
    all_times = prepared.all_times
    time_list = prepared.time_list
    compiled = _compile_with_signals(prepared, signals_map)
    n_global = len(all_times)
    if n_global == 0 or not compiled:
        empty_trades = pd.DataFrame(columns=TRADE_COLUMNS)
        empty_eq = pd.Series(dtype=float)
        empty_exp = pd.Series(dtype=float)
        metrics = compute_metrics(empty_eq, empty_trades, cfg.initial_capital, empty_exp)
        return BacktestResult(
            trades=empty_trades,
            equity_curve=empty_eq,
            exposure_series=empty_exp,
            metrics=metrics,
            config=cfg,
            params=used_params,
        )

    n_syms = len(compiled)
    open_positions: list[_OpenPosition | None] = [None] * n_syms
    cash = float(cfg.initial_capital)
    slip = float(cfg.slippage_bps) / 10_000.0
    fee_rate = float(cfg.fee_rate)
    delay = int(cfg.entry_delay_bars)
    funding_enabled = bool(cfg.funding_enabled)
    max_lev = float(cfg.max_leverage)
    risk_per_trade = float(cfg.risk_per_trade)

    equity_arr = np.empty(n_global, dtype=np.float64)
    exposure_arr = np.zeros(n_global, dtype=np.float64)
    trades_records: list[dict[str, Any]] = []
    time_list = prepared.time_list
    all_ns = all_times.asi8
    if trade_start is not None:
        ts_start = pd.Timestamp(trade_start)
        if ts_start.tzinfo is None:
            ts_start = ts_start.tz_localize("UTC")
        first_trade_g = int(np.searchsorted(all_ns, ts_start.as_unit("ns").value, side="left"))
    else:
        first_trade_g = 0
    fast_n_tr = 0
    fast_sum_r = 0.0
    fast_w_sum = 0.0
    fast_l_sum = 0.0

    def _close_position(
        s_idx: int,
        pos: _OpenPosition,
        raw_exit_px: float,
        exit_ts: pd.Timestamp | None,
        exit_reason: str,
    ) -> None:
        nonlocal cash, fast_n_tr, fast_sum_r, fast_w_sum, fast_l_sum
        exit_px = float(raw_exit_px) * (1.0 - pos.direction * slip)
        gross_pnl = pos.direction * pos.qty * (exit_px - pos.entry_price)
        exit_fee = pos.qty * exit_px * fee_rate
        total_fees = pos.entry_fee + exit_fee
        cash += gross_pnl - exit_fee
        net_pnl = gross_pnl - total_fees + pos.accrued_funding
        r_mult = net_pnl / pos.risk_amount if pos.risk_amount > 1e-12 else 0.0

        if fast_metrics:
            fast_n_tr += 1
            fast_sum_r += r_mult
            if net_pnl > 0.0:
                fast_w_sum += net_pnl
            elif net_pnl < 0.0:
                fast_l_sum += abs(net_pnl)
        else:
            trades_records.append(
                {
                    "symbol": pos.symbol,
                    "side": "LONG" if pos.direction > 0 else "SHORT",
                    "entry_time": pos.entry_time,
                    "entry_price": pos.entry_price,
                    "exit_time": exit_ts,
                    "exit_price": exit_px,
                    "exit_reason": exit_reason,
                    "qty": pos.qty,
                    "risk_amount": pos.risk_amount,
                    "R": r_mult,
                    "gross_pnl": gross_pnl,
                    "fees": total_fees,
                    "funding": pos.accrued_funding,
                    "net_pnl": net_pnl,
                    "bars_held": pos.bars_held,
                }
            )
        open_positions[s_idx] = None

    for g in range(n_global):
        ts_g = None if fast_metrics else time_list[g]

        # 1. Process scheduled open-of-bar exits across all symbols first
        for s_idx in range(n_syms):
            pos = open_positions[s_idx]
            if pos is None:
                continue
            cs = compiled[s_idx]
            k = int(cs.bar_map[g])
            if k < 0:
                continue
            src = k - delay
            reason = ""
            if pos.max_bars > 0 and pos.bars_held >= pos.max_bars:
                reason = "time_stop"
            elif src >= 0:
                if cs.exit_signals[src]:
                    reason = "exit_signal"
                elif cs.signals[src] == -pos.direction:
                    reason = "opposite_signal"

            if reason:
                _close_position(s_idx, pos, float(cs.opens[k]), ts_g, reason)

        # 2. Check if any flat symbol has an actionable entry signal at bar g
        candidate_indices: list[int] = []
        for s_idx in range(n_syms if g >= first_trade_g else 0):
            if open_positions[s_idx] is not None:
                continue
            cs = compiled[s_idx]
            k = int(cs.bar_map[g])
            if k < 0:
                continue
            src = k - delay
            if src >= 0 and cs.signals[src] != 0:
                sd = float(cs.stop_dists[src])
                if np.isfinite(sd) and sd > 0.0:
                    candidate_indices.append(s_idx)

        if candidate_indices:
            # Compute mark-to-market equity at bar g open and open gross notional
            mtm_open = cash
            open_notional = 0.0
            for s_idx in range(n_syms):
                pos = open_positions[s_idx]
                if pos is None:
                    continue
                cs = compiled[s_idx]
                k = int(cs.bar_map[g])
                ref_px = float(cs.opens[k]) if k >= 0 else float(cs.last_close_global[g])
                mtm_open += pos.direction * pos.qty * (ref_px - pos.entry_price)
                open_notional += pos.qty * ref_px

            # 3. Execute new entries at bar g open
            for s_idx in candidate_indices:
                if mtm_open <= 0.0:
                    break
                cs = compiled[s_idx]
                k = int(cs.bar_map[g])
                src = k - delay
                sig = int(cs.signals[src])
                sd = float(cs.stop_dists[src])
                raw_open = float(cs.opens[k])
                if raw_open <= 0.0:
                    continue
                entry_px = raw_open * (1.0 + sig * slip)
                available_notional = max(0.0, max_lev * mtm_open - open_notional)
                qty, risk_amt = fixed_fractional_size(
                    equity=mtm_open,
                    entry_price=entry_px,
                    stop_dist=sd,
                    risk_per_trade=risk_per_trade,
                    max_leverage=max_lev,
                    available_notional=available_notional,
                )
                if qty <= 0.0 or risk_amt <= 0.0:
                    continue

                entry_fee = qty * entry_px * fee_rate
                cash -= entry_fee
                open_notional += qty * entry_px

                if sig > 0:
                    stop_px = max(entry_px - sd, entry_px * 1e-6)
                else:
                    stop_px = entry_px + sd

                tp_val = float(cs.tp_rs[src])
                tp_px = (
                    (entry_px + sig * tp_val * sd)
                    if (np.isfinite(tp_val) and tp_val > 0.0)
                    else np.nan
                )
                mb = int(cs.max_bars[src]) if cs.max_bars[src] > 0 else 0

                open_positions[s_idx] = _OpenPosition(
                    symbol_idx=s_idx,
                    symbol=cs.symbol,
                    direction=sig,
                    entry_time=ts_g,
                    entry_global_idx=g,
                    entry_price=entry_px,
                    stop_price=stop_px,
                    tp_price=tp_px,
                    max_bars=mb,
                    qty=qty,
                    risk_amount=risk_amt,
                    entry_fee=entry_fee,
                    accrued_funding=0.0,
                    bars_held=0,
                )

        # 4. Intraday protective stop / TP checks (from entry bar onward) & funding accrual
        mtm_close = cash
        any_open = False
        for s_idx in range(n_syms):
            pos = open_positions[s_idx]
            if pos is None:
                continue
            cs = compiled[s_idx]
            k = int(cs.bar_map[g])
            if k >= 0:
                o_px = float(cs.opens[k])
                h_px = float(cs.highs[k])
                l_px = float(cs.lows[k])
                c_px = float(cs.closes[k])
                entered_earlier = pos.entry_global_idx < g
                has_tp = np.isfinite(pos.tp_price)

                exited_intraday = False
                if pos.direction > 0:
                    if entered_earlier and o_px <= pos.stop_price:
                        pos.bars_held += 1
                        _close_position(s_idx, pos, o_px, ts_g, "stop")
                        exited_intraday = True
                    elif l_px <= pos.stop_price:
                        # Worst-case: if both stop and TP touched in same bar, assume stop first
                        pos.bars_held += 1
                        _close_position(s_idx, pos, pos.stop_price, ts_g, "stop")
                        exited_intraday = True
                    elif has_tp and h_px >= pos.tp_price:
                        # Conservative: fill TP at tp_price even on gap in favour
                        pos.bars_held += 1
                        _close_position(s_idx, pos, pos.tp_price, ts_g, "take_profit")
                        exited_intraday = True
                else:
                    if entered_earlier and o_px >= pos.stop_price:
                        pos.bars_held += 1
                        _close_position(s_idx, pos, o_px, ts_g, "stop")
                        exited_intraday = True
                    elif h_px >= pos.stop_price:
                        pos.bars_held += 1
                        _close_position(s_idx, pos, pos.stop_price, ts_g, "stop")
                        exited_intraday = True
                    elif has_tp and l_px <= pos.tp_price:
                        pos.bars_held += 1
                        _close_position(s_idx, pos, pos.tp_price, ts_g, "take_profit")
                        exited_intraday = True

                if exited_intraday:
                    # Cash was updated by _close_position; sync mtm_close delta accurately below
                    continue

                # Position survived bar k
                pos.bars_held += 1
                if funding_enabled:
                    f_sum = float(cs.bar_funding_sum[k])
                    if f_sum != 0.0:
                        f_pnl = -pos.direction * (pos.qty * c_px) * f_sum
                        cash += f_pnl
                        pos.accrued_funding += f_pnl

                any_open = True
            else:
                any_open = True

        # Recompute mtm_close accurately with updated cash (in case intraday exits changed cash)
        mtm_close = cash
        for s_idx in range(n_syms):
            pos = open_positions[s_idx]
            if pos is not None:
                ref_c = float(compiled[s_idx].last_close_global[g])
                mtm_close += pos.direction * pos.qty * (ref_c - pos.entry_price)

        equity_arr[g] = mtm_close
        exposure_arr[g] = 1.0 if any_open else 0.0

    # Close any still-open positions at end of final bar so equity & trade ledger reconcile exactly
    last_ts = time_list[-1]
    closed_at_end = False
    for s_idx in range(n_syms):
        pos = open_positions[s_idx]
        if pos is not None:
            last_close = float(compiled[s_idx].last_close_global[-1])
            _close_position(s_idx, pos, last_close, last_ts, "end_of_data")
            closed_at_end = True
    if closed_at_end:
        equity_arr[-1] = cash

    if first_trade_g > 0:
        # Window evaluation: report only bars from `trade_start` on (equity before it is flat).
        win = slice(min(first_trade_g, n_global - 1), n_global)
        equity_arr = equity_arr[win]
        exposure_arr = exposure_arr[win]
        all_times = all_times[win]
        all_ns = all_ns[win]

    equity_series = pd.Series(equity_arr, index=all_times, name="equity")
    exposure_series = pd.Series(exposure_arr, index=all_times, name="exposure")

    if fast_metrics:
        sharpe_q = _fast_daily_sharpe(equity_arr, all_ns, cfg.initial_capital)

        peaks_q = np.maximum.accumulate(np.concatenate([[cfg.initial_capital], equity_arr]))[1:]
        peaks_q = np.maximum(peaks_q, cfg.initial_capital)
        max_dd_q = float(max(0.0, -np.min(equity_arr / peaks_q - 1.0)))
        exp_r_q = (fast_sum_r / fast_n_tr) if fast_n_tr > 0 else 0.0
        pf_q = min(fast_w_sum / fast_l_sum, 99.99) if fast_l_sum > 1e-12 else (99.99 if fast_w_sum > 0 else 0.0)
        metrics = {
            "initial_capital": float(cfg.initial_capital),
            "final_equity": float(equity_arr[-1]),
            "total_return": float(equity_arr[-1] / cfg.initial_capital - 1.0),
            "sharpe": round(sharpe_q, 4),
            "max_drawdown": round(max_dd_q, 6),
            "n_trades": fast_n_tr,
            "expectancy_r": round(exp_r_q, 4),
            "profit_factor": round(pf_q, 4),
        }
        trades_df = pd.DataFrame(columns=TRADE_COLUMNS)
    else:
        trades_df = (
            pd.DataFrame(trades_records, columns=TRADE_COLUMNS)
            if trades_records
            else pd.DataFrame(columns=TRADE_COLUMNS)
        )
        metrics = compute_metrics(equity_series, trades_df, cfg.initial_capital, exposure_series)

    return BacktestResult(
        trades=trades_df,
        equity_curve=equity_series,
        exposure_series=exposure_series,
        metrics=metrics,
        config=cfg,
        params=used_params,
    )
