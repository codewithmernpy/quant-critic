"""Chronological train/val/test splitting, grid search, and rolling walk-forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantcritic.config import AuditConfig, BacktestConfig, SplitConfig
from quantcritic.data.loader import MarketData, concat_market_data
from quantcritic.engine.backtest import BacktestResult, PreparedMarket, prepare_market, run_backtest
from quantcritic.engine.metrics import compute_metrics
from quantcritic.strategies.base import Strategy


class LockedSlice:
    """Held-out TEST MarketData that can only be read through `unlock()`; every read is counted."""

    def __init__(self, data: MarketData) -> None:
        self._data = data
        self.access_count = 0

    def unlock(self) -> MarketData:
        self.access_count += 1
        return self._data

    def __repr__(self) -> str:
        return f"LockedSlice(access_count={self.access_count})"


@dataclass
class DataSplits:
    """Chronological Train / Validation / Locked-Test MarketData partitions."""

    train: MarketData
    validation: MarketData
    train_val: MarketData
    test: LockedSlice
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_start: pd.Timestamp | None = None


def split_market_data(data: MarketData, splits: SplitConfig | None = None) -> DataSplits:
    """Split MarketData chronologically into 60% train / 20% validation / 20% test."""
    sc = splits or SplitConfig()
    all_ns = np.unique(
        np.concatenate(
            [pd.DatetimeIndex(df.index).as_unit("ns").asi8 for df in data.ohlcv.values() if not df.empty]
        )
    )
    if len(all_ns) < 30:
        raise ValueError("Insufficient bars to construct chronological train/val/test splits.")

    all_times = pd.DatetimeIndex(pd.to_datetime(all_ns, unit="ns", utc=True)).as_unit("ns")
    n = len(all_times)
    idx_train_end = max(1, int(n * sc.train_ratio) - 1)
    idx_val_end = max(idx_train_end + 1, int(n * (sc.train_ratio + sc.val_ratio)) - 1)
    idx_val_end = min(idx_val_end, n - 2)

    train_start = pd.Timestamp(all_times[0])
    train_end = pd.Timestamp(all_times[idx_train_end])
    val_start = pd.Timestamp(all_times[idx_train_end + 1])
    val_end = pd.Timestamp(all_times[idx_val_end])
    test_start = pd.Timestamp(all_times[idx_val_end + 1])
    test_end = pd.Timestamp(all_times[-1])

    train_md = data.slice_index_bounds(end=train_end)
    val_md = data.slice_index_bounds(start=val_start, end=val_end)
    train_val_md = data.slice_index_bounds(end=val_end)
    test_md = data.slice_index_bounds(start=test_start, end=test_end)
    # train_val keeps funding settlements up to the CLOSE of its last bar (val_end + 1 bar) so a
    # position held through that close accrues them; it contains no OHLCV bar of the test window.

    return DataSplits(
        train=train_md,
        validation=val_md,
        train_val=train_val_md,
        test=LockedSlice(test_md),
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        train_start=train_start,
    )


def run_window_backtest(
    history: MarketData,
    strategy: Strategy,
    params: dict[str, Any],
    cfg: BacktestConfig,
    window_start: pd.Timestamp,
    signals: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    """Evaluate the window [window_start, end of `history`] with warmed-up signals.

    Signals are generated on the whole `history` (which must end at the window end), so indicators
    and stateful strategies are warm at the window start; only trades entered at bars >= window_start
    are taken and the equity curve / metrics cover the window only.
    """
    sigs = signals if signals is not None else {
        sym: strategy.generate(df, params) for sym, df in history.ohlcv.items()
    }
    return run_backtest(history, sigs, params, cfg, trade_start=window_start)


def _select_best_grid_combo(
    data: MarketData,
    strategy: Strategy,
    combos: list[dict[str, Any]],
    cfg: BacktestConfig,
    min_trades: int = 30,
    trade_start: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], BacktestResult, list[dict[str, Any]]]:
    """Evaluate all grid combinations on `data` and pick best Sharpe with >= min_trades."""
    prep = prepare_market(data)
    grid_rows: list[dict[str, Any]] = []
    qualified: list[tuple[float, dict[str, Any]]] = []
    fallback: list[tuple[float, dict[str, Any]]] = []

    for p in combos:
        res = run_backtest(
            data, strategy, p, cfg, prepared_market=prep, fast_metrics=True, trade_start=trade_start
        )
        m = res.metrics
        row = {
            "params": dict(p),
            "sharpe": float(m["sharpe"]),
            "total_return": float(m["total_return"]),
            "max_drawdown": float(m["max_drawdown"]),
            "n_trades": int(m["n_trades"]),
            "expectancy_r": float(m["expectancy_r"]),
            "profit_factor": float(m["profit_factor"]),
        }
        grid_rows.append(row)
        if m["n_trades"] >= min_trades:
            qualified.append((float(m["sharpe"]), p))
        elif m["n_trades"] > 0:
            fallback.append((float(m["sharpe"]), p))

    if qualified:
        qualified.sort(key=lambda x: x[0], reverse=True)
        best_p = qualified[0][1]
    elif fallback:
        fallback.sort(key=lambda x: x[0], reverse=True)
        best_p = fallback[0][1]
    else:
        best_p = dict(strategy.default_params)

    best_res = run_backtest(
        data, strategy, best_p, cfg, prepared_market=prep, fast_metrics=False, trade_start=trade_start
    )
    return best_p, best_res, grid_rows


def _rolling_walk_forward(
    train_val_data: MarketData,
    strategy: Strategy,
    combos: list[dict[str, Any]],
    cfg: BacktestConfig,
    train_months: int = 12,
    test_months: int = 3,
    min_trades: int = 10,
    warmup_data: MarketData | None = None,
) -> dict[str, Any]:
    """Rolling walk-forward strictly inside train+validation (never touches locked test)."""
    all_ns = np.unique(
        np.concatenate(
            [pd.DatetimeIndex(df.index).as_unit("ns").asi8 for df in train_val_data.ohlcv.values() if not df.empty]
        )
    )
    if len(all_ns) == 0:
        return {"folds": [], "oos_metrics": {}, "oos_equity": pd.Series(dtype=float)}

    all_times = pd.DatetimeIndex(pd.to_datetime(all_ns, unit="ns", utc=True)).as_unit("ns")
    start_ts = pd.Timestamp(all_times[0])
    end_ts = pd.Timestamp(all_times[-1])

    # Fold parameter SELECTION uses signals generated once on train+validation (plus any pre-range
    # warm-up history) and sliced to each 12m training window. For any strategy that passes the
    # truncation test, generate(df[:t+1]) == generate(df)[:t+1], so this equals per-fold generation;
    # a strategy that fails it is REJECTED by the integrity gate. Each fold's OOS window is evaluated
    # with signals regenerated on history that ends exactly at that window's end (see below).
    full_tv = concat_market_data(warmup_data, train_val_data) if warmup_data is not None else train_val_data
    combo_signals = [
        (p, {sym: strategy.generate(df, p) for sym, df in full_tv.ohlcv.items()})
        for p in combos
    ]

    folds: list[dict[str, Any]] = []
    oos_equity_segments: list[pd.Series] = []
    oos_trades_list: list[pd.DataFrame] = []
    oos_exposure_segments: list[pd.Series] = []

    cursor = start_ts + pd.DateOffset(months=train_months)
    capital_cursor = float(cfg.initial_capital)

    while cursor < end_ts:
        win_train_start = cursor - pd.DateOffset(months=train_months)
        win_train_end = cursor - pd.Timedelta(seconds=1)
        win_test_end = min(end_ts, cursor + pd.DateOffset(months=test_months) - pd.Timedelta(seconds=1))
        if (win_test_end - cursor).days < 14:
            break

        tr_slice = train_val_data.slice_index_bounds(start=win_train_start, end=win_train_end)
        te_slice = train_val_data.slice_index_bounds(start=cursor, end=win_test_end)
        if all(df.empty for df in tr_slice.ohlcv.values()) or all(df.empty for df in te_slice.ohlcv.values()):
            cursor = cursor + pd.DateOffset(months=test_months)
            continue

        prep_tr = prepare_market(tr_slice)
        best_score = -1e9
        best_p = combo_signals[0][0]
        best_tr_sharpe = 0.0
        found_qualified = False

        for p, sigs in combo_signals:
            sliced_sigs = {
                sym: sigs[sym].loc[df.index[0] : df.index[-1]] if not df.empty else sigs[sym].iloc[:0]
                for sym, df in tr_slice.ohlcv.items()
            }
            res_tr = run_backtest(
                tr_slice, sliced_sigs, p, cfg, prepared_market=prep_tr, fast_metrics=True
            )
            nt = res_tr.metrics["n_trades"]
            sh = float(res_tr.metrics["sharpe"])
            if nt >= min_trades:
                if not found_qualified or sh > best_score:
                    found_qualified = True
                    best_score = sh
                    best_p = p
                    best_tr_sharpe = sh
            elif not found_qualified and nt > 0 and sh > best_score:
                best_score = sh
                best_p = p
                best_tr_sharpe = sh

        # OOS window: signals regenerated on ALL history up to the window end (warm-up, nothing
        # after it); the engine only takes entries from `cursor` on and reports the window only.
        te_cfg = cfg.model_copy(update={"initial_capital": capital_cursor})
        hist_to_oos_end = full_tv.slice_index_bounds(end=win_test_end)
        oos_sigs = {sym: strategy.generate(df, best_p) for sym, df in hist_to_oos_end.ohlcv.items()}
        engine_md = train_val_data.slice_index_bounds(start=win_train_start, end=win_test_end)
        res_te = run_backtest(engine_md, oos_sigs, best_p, te_cfg, trade_start=cursor)

        if not res_te.equity_curve.empty:
            capital_cursor = float(res_te.equity_curve.iloc[-1])
            oos_equity_segments.append(res_te.equity_curve)
            oos_exposure_segments.append(res_te.exposure_series)
        if not res_te.trades.empty:
            oos_trades_list.append(res_te.trades)

        folds.append(
            {
                "train_start": str(win_train_start.date()),
                "train_end": str(win_train_end.date()),
                "oos_start": str(cursor.date()),
                "oos_end": str(win_test_end.date()),
                "chosen_params": best_p,
                "train_sharpe": round(best_tr_sharpe, 3),
                "oos_sharpe": res_te.metrics["sharpe"],
                "oos_return": res_te.metrics["total_return"],
                "oos_trades": res_te.metrics["n_trades"],
            }
        )
        cursor = cursor + pd.DateOffset(months=test_months)

    if oos_equity_segments:
        stitched_eq = pd.concat(oos_equity_segments).sort_index()
        stitched_eq = stitched_eq[~stitched_eq.index.duplicated(keep="last")]
        stitched_exp = pd.concat(oos_exposure_segments).sort_index()
        stitched_exp = stitched_exp[~stitched_exp.index.duplicated(keep="last")]
        stitched_trades = (
            pd.concat(oos_trades_list, ignore_index=True)
            if oos_trades_list
            else pd.DataFrame()
        )
        oos_metrics = compute_metrics(
            stitched_eq, stitched_trades, cfg.initial_capital, stitched_exp
        )
    else:
        stitched_eq = pd.Series(dtype=float)
        oos_metrics = compute_metrics(stitched_eq, pd.DataFrame(), cfg.initial_capital)

    return {
        "folds": folds,
        "oos_metrics": oos_metrics,
        "oos_equity": stitched_eq,
        "combo_signals": combo_signals,
    }


def run_walkforward_audit(
    splits: DataSplits,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    audit_cfg: AuditConfig | None = None,
    chosen_params: dict[str, Any] | None = None,
    warmup_data: MarketData | None = None,
) -> dict[str, Any]:
    """Run train grid search, train->val degradation check, and rolling WF on train+val ONLY."""
    cfg = config or BacktestConfig(interval=splits.train.interval)
    ac = audit_cfg or AuditConfig()
    combos = strategy.expand_param_grid()

    train_hist = concat_market_data(warmup_data, splits.train) if warmup_data is not None else splits.train
    train_val_hist = concat_market_data(warmup_data, splits.train_val) if warmup_data is not None else splits.train_val
    trade_start = splits.train_start if warmup_data is not None else None

    best_params, train_res, grid_rows = _select_best_grid_combo(
        train_hist, strategy, combos, cfg, min_trades=ac.min_train_trades, trade_start=trade_start
    )
    eval_params = strategy.merge_params(chosen_params) if chosen_params else best_params
    if chosen_params:
        train_eval_res = run_backtest(
            train_hist, strategy, eval_params, cfg, trade_start=trade_start
        )
    else:
        train_eval_res = train_res

    # Validation: signals warmed up on train (plus any pre-range) history, only trades inside validation count
    val_res = run_window_backtest(train_val_hist, strategy, eval_params, cfg, splits.val_start)

    tr_sh = float(train_eval_res.metrics["sharpe"])
    va_sh = float(val_res.metrics["sharpe"])
    if tr_sh > 0.1:
        sharpe_deg_pct = ((tr_sh - va_sh) / tr_sh) * 100.0
    elif tr_sh < -0.1:
        sharpe_deg_pct = 0.0
    else:
        sharpe_deg_pct = 0.0

    wf = _rolling_walk_forward(
        splits.train_val,
        strategy,
        combos,
        cfg,
        train_months=ac.wf_train_months,
        test_months=ac.wf_test_months,
        min_trades=max(10, ac.min_train_trades // 2),
        warmup_data=warmup_data,
    )

    oos_m = wf["oos_metrics"]
    oos_ret = float(oos_m.get("total_return", 0.0))
    oos_sh = float(oos_m.get("sharpe", 0.0))
    val_exp = float(val_res.metrics["expectancy_r"])

    if va_sh < -0.5 or val_exp < -0.15 or oos_ret < -0.25:
        status = "fail"
        explanation = (
            f"Out-of-sample failure: validation Sharpe {va_sh:.2f} (train {tr_sh:.2f}), "
            f"rolling WF OOS return {oos_ret * 100:+.1f}%."
        )
    elif va_sh < 0.30 or (tr_sh > 0.5 and sharpe_deg_pct > 60.0) or oos_sh < 0.2:
        status = "warn"
        explanation = (
            f"Noticeable OOS decay: train Sharpe {tr_sh:.2f} -> val {va_sh:.2f} "
            f"({sharpe_deg_pct:+.0f}% decay); rolling WF Sharpe {oos_sh:.2f}."
        )
    else:
        status = "pass"
        explanation = (
            f"Solid OOS retention: train Sharpe {tr_sh:.2f} -> val {va_sh:.2f}; "
            f"rolling WF stitched return {oos_ret * 100:+.1f}% (Sharpe {oos_sh:.2f})."
        )

    train_start_val = splits.train_start
    if train_start_val is None:
        non_empty_tr = [df for df in splits.train.ohlcv.values() if not df.empty]
        train_start_val = min(pd.Timestamp(df.index[0]) for df in non_empty_tr) if non_empty_tr else splits.train_end

    return {
        "name": "walkforward",
        "status": status,
        "explanation": explanation,
        "best_train_params": best_params,
        "evaluated_params": eval_params,
        "train_metrics": train_eval_res.metrics,
        "validation_metrics": val_res.metrics,
        "sharpe_degradation_pct": round(sharpe_deg_pct, 2),
        "grid_results": grid_rows,
        "rolling_folds": wf["folds"],
        "rolling_oos_metrics": oos_m,
        "rolling_oos_equity": wf["oos_equity"],
        "_combo_signals": wf.get("combo_signals"),
        "splits_summary": {
            "train_start": str(train_start_val),
            "train_end": str(splits.train_end),
            "val_start": str(splits.val_start),
            "val_end": str(splits.val_end),
            "test_start": str(splits.test_start),
            "test_end": str(splits.test_end),
        },
    }
