"""Integrity and lookahead bias tests: truncation causality & 1-bar entry delay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.config import BacktestConfig
from quantcritic.data.loader import MarketData, parse_utc_bound, to_utc_ns_index
from quantcritic.engine.backtest import BacktestResult, run_backtest
from quantcritic.strategies.base import Strategy


def run_truncation_test(
    data: MarketData,
    strategy: Strategy,
    params: dict[str, Any] | None = None,
    n_cuts: int = 25,
    window: int = 100,
    seed: int = 42,
    trade_start: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Verify strict causality by truncating DataFrames at random points t.

    Compares signal, stop_dist, tp_r, max_bars and exit_signal on the last `window` bars <= t
    between the truncated run and the full-sample run.
    """
    used_params = strategy.merge_params(params)
    rng = np.random.default_rng(seed)
    examples: list[dict[str, Any]] = []
    total_mismatches = 0
    cuts_checked = 0
    cuts_with_mismatch = 0
    ts_start = parse_utc_bound(trade_start, is_end=False)

    for sym, df in data.ohlcv.items():
        n = len(df)
        if n < 30:
            continue

        if ts_start is not None:
            idx_ns = to_utc_ns_index(df.index).asi8
            first_win_idx = int(np.searchsorted(idx_ns, ts_start.value, side="left"))
            if first_win_idx >= n - 2:
                continue
        else:
            first_win_idx = 0

        eff_n = n - first_win_idx
        if eff_n < 30:
            continue

        full_sig = strategy.generate(df, used_params)
        min_cut = min(first_win_idx + max(30, eff_n // 5), n - 2)
        max_cut = n - 2
        if max_cut <= min_cut:
            cut_indices = np.array([max_cut], dtype=int)
        else:
            count = min(n_cuts, max_cut - min_cut + 1)
            cut_indices = np.sort(rng.choice(np.arange(min_cut, max_cut + 1), size=count, replace=False))

        for t in cut_indices:
            cuts_checked += 1
            t_int = int(t)
            sub_df = df.iloc[: t_int + 1]
            trunc_sig = strategy.generate(sub_df, used_params)

            lo = max(first_win_idx, t_int - window + 1)
            hi = t_int + 1
            f_slice = full_sig.iloc[lo:hi]
            t_slice = trunc_sig.iloc[lo:hi]

            cut_has_mismatch = False
            # Compare EVERY order-relevant output column (entry, stop, take-profit, time stop, exit)
            comparisons: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
            for col in ("signal", "stop_dist", "tp_r", "max_bars", "exit_signal"):
                if col not in f_slice.columns and col not in t_slice.columns:
                    continue
                if col not in f_slice.columns or col not in t_slice.columns:
                    raise ValueError(f"Strategy output column {col!r} present in only one of the runs")
                if col == "exit_signal":
                    f_arr = f_slice[col].fillna(False).to_numpy(dtype=bool)
                    t_arr = t_slice[col].fillna(False).to_numpy(dtype=bool)
                    diff_mask = f_arr != t_arr
                elif col in ("signal", "max_bars"):
                    f_arr = f_slice[col].fillna(0).to_numpy(dtype=np.int64)
                    t_arr = t_slice[col].fillna(0).to_numpy(dtype=np.int64)
                    diff_mask = f_arr != t_arr
                else:
                    # Floats: NaN == NaN; otherwise relative tolerance for round-off only
                    f_arr = f_slice[col].to_numpy(dtype=np.float64)
                    t_arr = t_slice[col].to_numpy(dtype=np.float64)
                    both_nan = np.isnan(f_arr) & np.isnan(t_arr)
                    close = np.isclose(f_arr, t_arr, rtol=1e-9, atol=1e-12)
                    diff_mask = (~both_nan) & ~close
                comparisons.append((col, np.where(diff_mask)[0], f_arr, t_arr))

            n_diff_here = sum(len(c[1]) for c in comparisons)
            if n_diff_here > 0:
                cut_has_mismatch = True
                total_mismatches += n_diff_here
                cut_ts = str(df.index[t_int])
                for col, idxs, f_arr, t_arr in comparisons:
                    for rel_i in idxs[:5]:
                        if len(examples) >= 20:
                            break
                        bar_ts = str(f_slice.index[int(rel_i)])
                        fv = f_arr[int(rel_i)]
                        tv = t_arr[int(rel_i)]
                        examples.append(
                            {
                                "symbol": sym,
                                "cut_time": cut_ts,
                                "bar_time": bar_ts,
                                "column": col,
                                "full_value": None if (isinstance(fv, float) and np.isnan(fv)) else (int(fv) if isinstance(fv, (np.integer, int)) else (bool(fv) if isinstance(fv, (np.bool_, bool)) else round(float(fv), 6))),
                                "truncated_value": None if (isinstance(tv, float) and np.isnan(tv)) else (int(tv) if isinstance(tv, (np.integer, int)) else (bool(tv) if isinstance(tv, (np.bool_, bool)) else round(float(tv), 6))),
                            }
                        )
            if cut_has_mismatch:
                cuts_with_mismatch += 1

    passed = total_mismatches == 0
    return {
        "passed": passed,
        "cuts_checked": cuts_checked,
        "cuts_with_mismatch": cuts_with_mismatch,
        "total_mismatches": total_mismatches,
        "examples": examples,
    }


def run_delay_test(
    data: MarketData,
    strategy: Strategy,
    params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    base_result: BacktestResult | None = None,
    trade_start: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Rerun backtest with entries delayed by 1 extra bar (`entry_delay_bars=2`)."""
    cfg_base = config or BacktestConfig(interval=data.interval)
    used_params = strategy.merge_params(params)
    signals_map = {sym: strategy.generate(df, used_params) for sym, df in data.ohlcv.items()}

    if base_result is None:
        base_res = run_backtest(data, signals_map, used_params, cfg_base, trade_start=trade_start)
    else:
        base_res = base_result

    cfg_delayed = cfg_base.model_copy(update={"entry_delay_bars": cfg_base.entry_delay_bars + 1})
    delayed_res = run_backtest(data, signals_map, used_params, cfg_delayed, trade_start=trade_start)

    base_exp = float(base_res.metrics["expectancy_r"])
    del_exp = float(delayed_res.metrics["expectancy_r"])
    base_sharpe = float(base_res.metrics["sharpe"])
    del_sharpe = float(delayed_res.metrics["sharpe"])

    if base_exp > 1e-6:
        collapse_pct = (base_exp - del_exp) / base_exp
    elif abs(base_exp) > 1e-6:
        collapse_pct = (base_exp - del_exp) / abs(base_exp)
    else:
        collapse_pct = 0.0

    warn = bool(base_exp > 0.0 and collapse_pct > 0.70)
    return {
        "base_expectancy_r": round(base_exp, 4),
        "delayed_expectancy_r": round(del_exp, 4),
        "base_sharpe": round(base_sharpe, 4),
        "delayed_sharpe": round(del_sharpe, 4),
        "expectancy_collapse_pct": round(collapse_pct * 100.0, 2),
        "warn": warn,
    }


def run_lookahead_audit(
    data: MarketData,
    strategy: Strategy,
    params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    n_cuts: int = 25,
    window: int = 100,
    seed: int = 42,
    base_result: BacktestResult | None = None,
    trade_start: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Run full integrity check combining truncation causality and 1-bar delay test."""
    trunc = run_truncation_test(
        data=data,
        strategy=strategy,
        params=params,
        n_cuts=n_cuts,
        window=window,
        seed=seed,
        trade_start=trade_start,
    )
    delay = run_delay_test(
        data=data,
        strategy=strategy,
        params=params,
        config=config,
        base_result=base_result,
        trade_start=trade_start,
    )

    if not trunc["passed"]:
        status = "fail"
        explanation = (
            f"Lookahead FAIL: {trunc['total_mismatches']} signal/stop mismatches across "
            f"{trunc['cuts_with_mismatch']}/{trunc['cuts_checked']} truncation cuts."
        )
    elif delay["warn"]:
        status = "warn"
        explanation = (
            f"Truncation passed, but 1-bar delay collapsed expectancy by "
            f"{delay['expectancy_collapse_pct']:.1f}% (edge depends on unrealistic timing)."
        )
    else:
        status = "pass"
        explanation = (
            f"Passed {trunc['cuts_checked']} causal truncation cuts with 0 mismatches; "
            f"1-bar entry delay changes expectancy by {-delay['expectancy_collapse_pct']:+.1f}% "
            f"({delay['base_expectancy_r']:+.3f}R -> {delay['delayed_expectancy_r']:+.3f}R)."
        )

    return {
        "name": "lookahead",
        "status": status,
        "explanation": explanation,
        "truncation": trunc,
        "delay": delay,
    }
