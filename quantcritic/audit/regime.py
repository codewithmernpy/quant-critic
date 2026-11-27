"""Regime, calendar-year, symbol concentration, and outlier-removal consistency audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.data.loader import MarketData
from quantcritic.engine.backtest import BacktestResult


def classify_btc_regimes(data: MarketData) -> pd.Series:
    """Causally classify each UTC date as bull / bear / chop from BTC daily closes.

    Inputs known at the close of day d: 200-day SMA (full 200 days required), 30-day realized vol
    and its EXPANDING 80th percentile (no full-sample statistic). The label for day d is shifted by
    one day, i.e. it only uses data up to the close of d-1, so a trade entered intraday on day d is
    tagged with information available before it was opened. Days without enough history are NaN
    (reported as unclassified).

    bull = close >= SMA200 and vol30 <= q80; bear = close < SMA200; chop = close >= SMA200 with
    vol30 > q80 (unstable, high-volatility tape above the trend line).
    """
    ref_sym = "BTCUSDT" if "BTCUSDT" in data.ohlcv else next(iter(data.ohlcv.keys()))
    df = data.ohlcv[ref_sym]
    if df.empty:
        return pd.Series(dtype=object, name="regime")

    daily_close = df["close"].resample("1D").last().dropna()
    if daily_close.empty:
        return pd.Series(dtype=object, name="regime")

    sma_200 = daily_close.rolling(200, min_periods=200).mean()
    daily_ret = daily_close.pct_change()
    vol_30 = daily_ret.rolling(30, min_periods=30).std(ddof=1) * np.sqrt(365.0)
    vol_q80 = vol_30.expanding(min_periods=90).quantile(0.80)

    known = sma_200.notna() & vol_30.notna() & vol_q80.notna()
    regimes = pd.Series(np.nan, index=daily_close.index, dtype=object)
    regimes[known] = "chop"
    regimes[known & (daily_close >= sma_200) & (vol_30 <= vol_q80)] = "bull"
    regimes[known & (daily_close < sma_200)] = "bear"
    regimes = regimes.shift(1)
    regimes.index = regimes.index.normalize()
    regimes.name = "regime"
    return regimes


def _subset_trade_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n_trades": 0, "win_rate": 0.0, "avg_r": 0.0, "net_pnl": 0.0, "profit_factor": 0.0}
    wins = float(df.loc[df["net_pnl"] > 0.0, "net_pnl"].sum())
    losses = float(abs(df.loc[df["net_pnl"] < 0.0, "net_pnl"].sum()))
    pf = float(wins / losses) if losses > 1e-12 else (99.99 if wins > 0 else 0.0)
    return {
        "n_trades": int(len(df)),
        "win_rate": round(float((df["net_pnl"] > 0.0).mean()), 4),
        "avg_r": round(float(df["R"].mean()), 4),
        "net_pnl": round(float(df["net_pnl"].sum()), 2),
        "profit_factor": round(min(pf, 99.99), 4),
    }


def run_regime_audit(data: MarketData, result: BacktestResult) -> dict[str, Any]:
    """Break down performance by calendar year, BTC market regime, symbol, side, and top-5% outlier removal."""
    trades = result.trades
    eq = result.equity_curve
    initial_capital = float(result.config.initial_capital)

    # 1. Per-year breakdown
    yearly_stats: list[dict[str, Any]] = []
    if not eq.empty:
        daily_eq = eq.resample("1D").last().dropna()
        years = sorted(pd.DatetimeIndex(daily_eq.index).year.unique().tolist())
        prev_eq = initial_capital
        for yr in years:
            yr_eq = daily_eq[pd.DatetimeIndex(daily_eq.index).year == yr]
            if yr_eq.empty:
                continue
            end_eq = float(yr_eq.iloc[-1])
            yr_ret = (end_eq / prev_eq) - 1.0 if prev_eq > 0 else 0.0
            d_rets = yr_eq.pct_change().dropna()
            yr_sh = (
                float((d_rets.mean() / d_rets.std(ddof=1)) * np.sqrt(365.0))
                if len(d_rets) >= 2 and d_rets.std(ddof=1) > 1e-12
                else 0.0
            )
            rm = np.maximum.accumulate(np.concatenate([[prev_eq], yr_eq.to_numpy(dtype=np.float64)]))[1:]
            yr_dd = float(max(0.0, np.max(1.0 - yr_eq.to_numpy(dtype=np.float64) / np.maximum(rm, 1e-9))))
            if not trades.empty:
                yr_tr = trades[pd.to_datetime(trades["entry_time"], utc=True).dt.year == yr]
                nt = int(len(yr_tr))
                wr = float((yr_tr["net_pnl"] > 0).mean()) if nt > 0 else 0.0
                avg_r = float(yr_tr["R"].mean()) if nt > 0 else 0.0
            else:
                nt, wr, avg_r = 0, 0.0, 0.0
            yearly_stats.append(
                {
                    "year": int(yr),
                    "return": round(yr_ret, 4),
                    "sharpe": round(yr_sh, 3),
                    "max_drawdown": round(yr_dd, 4),
                    "n_trades": nt,
                    "win_rate": round(wr, 4),
                    "avg_r": round(avg_r, 4),
                }
            )
            prev_eq = end_eq

    # 2. BTC Regime breakdown
    regimes = classify_btc_regimes(data)
    regime_stats: dict[str, dict[str, Any]] = {}
    unclassified_trades = 0
    if not trades.empty and not regimes.empty:
        entry_dates = pd.to_datetime(trades["entry_time"], utc=True).dt.normalize()
        mapped_reg = entry_dates.map(regimes)
        unclassified_trades = int(mapped_reg.isna().sum())
        for reg_name in ("bull", "bear", "chop"):
            sub = trades.loc[mapped_reg == reg_name]
            regime_stats[reg_name] = _subset_trade_stats(sub)
    else:
        unclassified_trades = int(len(trades))
        for reg_name in ("bull", "bear", "chop"):
            regime_stats[reg_name] = _subset_trade_stats(pd.DataFrame())

    # 3. Symbol share of profit
    symbol_stats: list[dict[str, Any]] = []
    total_net_pnl = float(trades["net_pnl"].sum()) if not trades.empty else 0.0
    for sym in data.symbols:
        sub = trades.loc[trades["symbol"] == sym] if not trades.empty else pd.DataFrame()
        st = _subset_trade_stats(sub)
        share = (st["net_pnl"] / abs(total_net_pnl) * 100.0) if abs(total_net_pnl) > 1e-9 else 0.0
        st["symbol"] = sym
        st["profit_share_pct"] = round(share, 2)
        symbol_stats.append(st)

    # 4. Outlier sensitivity: remove best 5% of trades
    if not trades.empty and len(trades) >= 10:
        n_remove = max(1, int(np.ceil(0.05 * len(trades))))
        trimmed = trades.sort_values("net_pnl", ascending=True).iloc[:-n_remove]
        trimmed_pnl = float(trimmed["net_pnl"].sum())
        trimmed_avg_r = float(trimmed["R"].mean())
        still_profitable = trimmed_pnl > 0.0
    elif not trades.empty:
        n_remove = 1
        trimmed = trades.sort_values("net_pnl", ascending=True).iloc[:-1]
        trimmed_pnl = float(trimmed["net_pnl"].sum()) if not trimmed.empty else 0.0
        trimmed_avg_r = float(trimmed["R"].mean()) if not trimmed.empty else 0.0
        still_profitable = trimmed_pnl > 0.0
    else:
        n_remove = 0
        trimmed_pnl = 0.0
        trimmed_avg_r = 0.0
        still_profitable = False

    outlier_check = {
        "trades_removed": n_remove,
        "original_net_pnl": round(total_net_pnl, 2),
        "trimmed_net_pnl": round(trimmed_pnl, 2),
        "trimmed_avg_r": round(trimmed_avg_r, 4),
        "still_profitable": bool(still_profitable),
    }

    pos_years = sum(1 for y in yearly_stats if y["return"] > 0.0)
    pos_regimes = sum(1 for r in regime_stats.values() if r["n_trades"] > 0 and r["net_pnl"] > 0.0)
    max_sym_share = max((s["profit_share_pct"] for s in symbol_stats), default=0.0)

    if total_net_pnl <= 0.0:
        status = "fail"
        explanation = f"Strategy is net unprofitable (${total_net_pnl:+,.0f}); {pos_years}/{len(yearly_stats)} positive years."
    elif not still_profitable:
        status = "fail"
        explanation = (
            f"Outlier-dependent: removing the top 5% ({n_remove}) trades flips net PnL from "
            f"${total_net_pnl:+,.0f} to ${trimmed_pnl:+,.0f}."
        )
    elif pos_years < max(1, len(yearly_stats) // 2) or pos_regimes < 2 or (len(symbol_stats) > 1 and max_sym_share > 90.0):
        status = "warn"
        explanation = (
            f"Regime/asset concentration: {pos_years}/{len(yearly_stats)} positive years, "
            f"{pos_regimes}/3 profitable regimes, max symbol share {max_sym_share:.0f}%."
        )
    else:
        status = "pass"
        explanation = (
            f"Consistent across {pos_years}/{len(yearly_stats)} years and {pos_regimes}/3 BTC regimes; "
            f"remains profitable (${trimmed_pnl:+,.0f}) after removing top 5% of trades."
        )

    return {
        "name": "regime",
        "status": status,
        "explanation": explanation,
        "yearly": yearly_stats,
        "regimes": regime_stats,
        "unclassified_regime_trades": unclassified_trades,
        "symbols": symbol_stats,
        "long_vs_short": {
            "long": result.metrics.get("long_stats", {}),
            "short": result.metrics.get("short_stats", {}),
        },
        "outlier_check": outlier_check,
    }
