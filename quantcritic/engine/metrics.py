"""Performance and risk metrics for crypto-perpetuals backtests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_monthly_returns_table(
    equity_curve: pd.Series, initial_capital: float
) -> dict[str, dict[str, float]]:
    """Compute compounded monthly returns keyed by year -> month (01..12)."""
    if equity_curve.empty:
        return {}

    eq = equity_curve.sort_index()
    idx = pd.DatetimeIndex(eq.index)
    years = idx.year
    months = idx.month

    month_ends: list[tuple[int, int, float]] = []
    # Group preserving chronological order
    df_tmp = pd.DataFrame({"year": years, "month": months, "equity": eq.to_numpy(dtype=np.float64)})
    grouped = df_tmp.groupby(["year", "month"], sort=True)["equity"].last()

    table: dict[str, dict[str, float]] = {}
    prev_eq = float(initial_capital)
    for (yr, mo), end_eq in grouped.items():
        ret = (float(end_eq) / prev_eq) - 1.0 if prev_eq > 0 else 0.0
        yr_str = str(int(yr))
        mo_str = f"{int(mo):02d}"
        table.setdefault(yr_str, {})[mo_str] = round(ret, 6)
        prev_eq = float(end_eq)
    return table


def _side_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute trade statistics for a single directional subset (LONG or SHORT)."""
    if df.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "net_pnl": 0.0,
            "profit_factor": 0.0,
        }
    wins = df.loc[df["net_pnl"] > 0.0, "net_pnl"].sum()
    losses = abs(df.loc[df["net_pnl"] < 0.0, "net_pnl"].sum())
    pf = float(wins / losses) if losses > 1e-12 else (99.99 if wins > 0 else 0.0)
    return {
        "n_trades": int(len(df)),
        "win_rate": round(float((df["net_pnl"] > 0.0).mean()), 4),
        "avg_r": round(float(df["R"].mean()), 4),
        "net_pnl": round(float(df["net_pnl"].sum()), 2),
        "profit_factor": round(min(pf, 99.99), 4),
    }


def compute_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    exposure_series: pd.Series | None = None,
) -> dict[str, Any]:
    """Compute portfolio equity, drawdown, and trade-level performance metrics."""
    if equity_curve.empty:
        return {
            "initial_capital": float(initial_capital),
            "final_equity": float(initial_capital),
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_days": 0.0,
            "calmar": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "expectancy_r": 0.0,
            "expectancy_usd": 0.0,
            "profit_factor": 0.0,
            "trades_per_month": 0.0,
            "exposure_pct": 0.0,
            "best_month": 0.0,
            "worst_month": 0.0,
            "total_fees": 0.0,
            "total_funding": 0.0,
            "monthly_returns": {},
            "long_stats": _side_stats(pd.DataFrame()),
            "short_stats": _side_stats(pd.DataFrame()),
        }

    eq = equity_curve.sort_index()
    eq_vals = eq.to_numpy(dtype=np.float64)
    final_equity = float(eq_vals[-1])
    total_return = (final_equity / initial_capital) - 1.0 if initial_capital > 0 else 0.0

    start_ts = pd.Timestamp(eq.index[0])
    end_ts = pd.Timestamp(eq.index[-1])
    elapsed_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1.0)
    years = elapsed_days / 365.0
    if final_equity > 0 and initial_capital > 0 and years > 0:
        log_ratio = np.log(final_equity / initial_capital) / years
        cagr = float(np.expm1(min(log_ratio, 20.0)))
    else:
        cagr = -1.0 if final_equity <= 0 else 0.0

    # Daily equity resampling for Sharpe & Sortino (annualized by sqrt(365) for 24/7 crypto)
    daily_eq = eq.resample("1D").last().dropna()
    if len(daily_eq) >= 1:
        # Anchor day 0 at initial capital so the first day's P&L is part of the return series
        daily_vals = np.concatenate([[float(initial_capital)], daily_eq.to_numpy(dtype=np.float64)])
        prev = np.where(daily_vals[:-1] > 0, daily_vals[:-1], np.nan)
        daily_rets = (daily_vals[1:] - prev) / prev
        daily_rets = daily_rets[np.isfinite(daily_rets)]
    else:
        daily_rets = np.array([], dtype=np.float64)

    if len(daily_rets) >= 2:
        mean_d = float(np.mean(daily_rets))
        std_d = float(np.std(daily_rets, ddof=1))
        sharpe = float((mean_d / std_d) * np.sqrt(365.0)) if std_d > 1e-12 else 0.0
        downside = np.minimum(daily_rets, 0.0)
        down_dev = float(np.sqrt(np.mean(downside ** 2)))
        sortino = float((mean_d / down_dev) * np.sqrt(365.0)) if down_dev > 1e-12 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Drawdown and duration
    eq_with_start = np.concatenate([[initial_capital], eq_vals])
    ts_ns = pd.DatetimeIndex(eq.index).as_unit("ns").asi8
    running_max = np.maximum.accumulate(eq_with_start)[1:]
    running_max = np.maximum(running_max, initial_capital)
    underwater = np.where(running_max > 0, eq_vals / running_max - 1.0, 0.0)
    max_drawdown = float(max(0.0, -np.min(underwater)))

    peak_val = float(initial_capital)
    peak_ns = int(ts_ns[0])
    max_dd_ns = 0
    for i in range(len(eq_vals)):
        if eq_vals[i] >= peak_val:
            peak_val = float(eq_vals[i])
            peak_ns = int(ts_ns[i])
        else:
            dur = int(ts_ns[i]) - peak_ns
            if dur > max_dd_ns:
                max_dd_ns = dur
    max_dd_duration_days = float(max_dd_ns / (86400.0 * 1e9))

    calmar = float(cagr / max_drawdown) if max_drawdown > 1e-9 else 0.0

    monthly_table = compute_monthly_returns_table(eq, initial_capital)
    flat_months = [r for m_map in monthly_table.values() for r in m_map.values()]
    best_month = float(max(flat_months)) if flat_months else 0.0
    worst_month = float(min(flat_months)) if flat_months else 0.0

    n_trades = int(len(trades))
    if n_trades > 0:
        win_rate = float((trades["net_pnl"] > 0.0).mean())
        avg_r = float(trades["R"].mean())
        expectancy_usd = float(trades["net_pnl"].mean())
        wins = float(trades.loc[trades["net_pnl"] > 0.0, "net_pnl"].sum())
        losses = float(abs(trades.loc[trades["net_pnl"] < 0.0, "net_pnl"].sum()))
        profit_factor = float(wins / losses) if losses > 1e-12 else (99.99 if wins > 0 else 0.0)
        total_fees = float(trades["fees"].sum())
        total_funding = float(trades["funding"].sum())
        long_stats = _side_stats(trades.loc[trades["side"] == "LONG"])
        short_stats = _side_stats(trades.loc[trades["side"] == "SHORT"])
    else:
        win_rate = 0.0
        avg_r = 0.0
        expectancy_usd = 0.0
        profit_factor = 0.0
        total_fees = 0.0
        total_funding = 0.0
        long_stats = _side_stats(pd.DataFrame())
        short_stats = _side_stats(pd.DataFrame())

    months_elapsed = max(elapsed_days / 30.4375, 1e-6)
    trades_per_month = float(n_trades / months_elapsed)
    exposure_pct = (
        float(exposure_series.mean() * 100.0)
        if exposure_series is not None and not exposure_series.empty
        else 0.0
    )

    return {
        "initial_capital": round(float(initial_capital), 2),
        "final_equity": round(final_equity, 2),
        "total_return": round(float(total_return), 6),
        "cagr": round(float(cagr), 6),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "max_drawdown": round(float(max_drawdown), 6),
        "max_drawdown_duration_days": round(float(max_dd_duration_days), 2),
        "calmar": round(float(calmar), 4),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "avg_r": round(avg_r, 4),
        "expectancy_r": round(avg_r, 4),
        "expectancy_usd": round(expectancy_usd, 2),
        "profit_factor": round(min(profit_factor, 99.99), 4),
        "trades_per_month": round(trades_per_month, 2),
        "exposure_pct": round(exposure_pct, 2),
        "best_month": round(best_month, 6),
        "worst_month": round(worst_month, 6),
        "total_fees": round(total_fees, 2),
        "total_funding": round(total_funding, 2),
        "monthly_returns": monthly_table,
        "long_stats": long_stats,
        "short_stats": short_stats,
    }
