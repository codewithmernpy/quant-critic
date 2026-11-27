"""Luck and significance tests: vectorized trade bootstrap and random-entry baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.config import BacktestConfig
from quantcritic.data.loader import MarketData
from quantcritic.engine.backtest import BacktestResult, _precompute_bar_funding
from quantcritic.strategies.base import compute_atr


def run_trade_bootstrap(
    trades: pd.DataFrame,
    risk_per_trade: float = 0.005,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap trade R-multiples 1000x to estimate return, drawdown, and loss probabilities."""
    if trades.empty or len(trades) < 5:
        return {
            "n_simulations": 0,
            "median_return": 0.0,
            "p5_return": 0.0,
            "p95_return": 0.0,
            "p95_max_drawdown": 0.0,
            "median_max_drawdown": 0.0,
            "prob_loss": 1.0,
            "sharpe_ci_low": 0.0,
            "sharpe_ci_high": 0.0,
            "return_samples": [],
            "max_dd_samples": [],
        }

    rng = np.random.default_rng(seed)
    r_vals = trades["R"].to_numpy(dtype=np.float64)
    n_trades = len(r_vals)
    # Approximate trade-level fractional return using configured risk_per_trade
    trade_rets = np.clip(r_vals * risk_per_trade, -0.95, 5.0)

    idx = rng.integers(0, n_trades, size=(n_simulations, n_trades))
    sampled = trade_rets[idx]  # (B, N)

    # Compounded equity paths starting at 1.0
    paths = np.cumprod(1.0 + sampled, axis=1)
    paths_with_one = np.hstack([np.ones((n_simulations, 1), dtype=np.float64), paths])
    peaks = np.maximum.accumulate(paths_with_one, axis=1)[:, 1:]
    drawdowns = np.where(peaks > 0.0, 1.0 - paths / peaks, 0.0)

    final_returns = paths[:, -1] - 1.0
    max_dds = np.max(drawdowns, axis=1)

    # Approximate annualized trade Sharpe across simulations using empirical trades/year
    span_days = max(
        (pd.Timestamp(trades["exit_time"].max()) - pd.Timestamp(trades["entry_time"].min())).total_seconds()
        / 86400.0,
        30.0,
    )
    trades_per_year = n_trades / (span_days / 365.0)
    means = np.mean(sampled, axis=1)
    stds = np.std(sampled, axis=1, ddof=1)
    sim_sharpes = np.where(stds > 1e-12, (means / stds) * np.sqrt(trades_per_year), 0.0)

    return {
        "n_simulations": n_simulations,
        "median_return": round(float(np.median(final_returns)), 4),
        "p5_return": round(float(np.quantile(final_returns, 0.05)), 4),
        "p95_return": round(float(np.quantile(final_returns, 0.95)), 4),
        "median_max_drawdown": round(float(np.median(max_dds)), 4),
        "p95_max_drawdown": round(float(np.quantile(max_dds, 0.95)), 4),
        "prob_loss": round(float(np.mean(final_returns < 0.0)), 4),
        "sharpe_ci_low": round(float(np.quantile(sim_sharpes, 0.025)), 4),
        "sharpe_ci_high": round(float(np.quantile(sim_sharpes, 0.975)), 4),
        "return_samples": [round(float(x), 4) for x in final_returns],
        "max_dd_samples": [round(float(x), 4) for x in max_dds],
    }


def run_random_entry_baseline(
    data: MarketData,
    real_result: BacktestResult,
    params: dict[str, Any],
    n_runs: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare real expectancy against 200 vectorized random-entry portfolios.

    Matches per-symbol trade count, holding-time distribution, long/short ratio,
    ATR stop distance (ATR known at the close of the bar before entry, like a real signal),
    optional TP rule, fees, slippage, and perpetual funding (same accrual rule as the engine),
    so real and random R-multiples are measured on identical terms.
    """
    trades = real_result.trades
    if trades.empty or len(trades) < 5:
        return {
            "n_runs": 0,
            "real_expectancy_r": 0.0,
            "random_mean_expectancy_r": 0.0,
            "random_std_expectancy_r": 0.0,
            "p_value": 1.0,
            "baseline_expectancies": [],
        }

    rng = np.random.default_rng(seed)
    cfg = real_result.config
    slip = float(cfg.slippage_bps) / 10_000.0
    fee_rate = float(cfg.fee_rate)
    funding_enabled = bool(cfg.funding_enabled)
    atr_period = int(params.get("atr_period", 14))
    atr_mult = float(params.get("atr_stop_mult", 2.5))
    tp_r_param = float(params.get("tp_r", 0.0) or 0.0)
    has_tp = np.isfinite(tp_r_param) and tp_r_param > 0.0

    # Accumulate R-sums across runs for each symbol
    total_r_sum_per_run = np.zeros(n_runs, dtype=np.float64)
    total_trades_count = 0

    for sym, df in data.ohlcv.items():
        sym_trades = trades.loc[trades["symbol"] == sym]
        n_sym = len(sym_trades)
        if n_sym == 0 or len(df) < atr_period + 10:
            continue

        total_trades_count += n_sym
        n_bars = len(df)
        warmup = min(atr_period + 5, n_bars // 4)
        max_entry_idx = max(warmup + 1, n_bars - 2)

        opens = df["open"].to_numpy(dtype=np.float64)
        highs = df["high"].to_numpy(dtype=np.float64)
        lows = df["low"].to_numpy(dtype=np.float64)
        closes = df["close"].to_numpy(dtype=np.float64)
        stop_dists = (compute_atr(df, atr_period) * atr_mult).fillna(0.0).to_numpy(dtype=np.float64)
        if funding_enabled:
            bar_funding = _precompute_bar_funding(df.index, data.funding.get(sym), data.interval)
        else:
            bar_funding = np.zeros(n_bars, dtype=np.float64)

        holds_pool = np.maximum(sym_trades["bars_held"].to_numpy(dtype=np.int32), 1)
        long_prob = float((sym_trades["side"] == "LONG").mean())

        m = n_runs * n_sym
        entries = rng.integers(warmup, max_entry_idx, size=m, dtype=np.int32)
        dirs = np.where(rng.random(m) < long_prob, 1, -1).astype(np.int8)
        holds = np.minimum(rng.choice(holds_pool, size=m, replace=True), 250)

        # Stop distance from the ATR known at the close of the bar BEFORE entry (no peeking at the
        # entry bar's own high/low/close); the tiny floor only guards against division by zero.
        sig_bar = entries - 1
        sd = np.maximum(stop_dists[sig_bar], closes[sig_bar] * 1e-4)
        entry_px = opens[entries] * (1.0 + dirs * slip)
        funding_acc = np.zeros(m, dtype=np.float64)
        stop_px = np.where(dirs > 0, np.maximum(entry_px - sd, entry_px * 1e-6), entry_px + sd)
        tp_px = entry_px + dirs * tp_r_param * sd if has_tp else np.zeros(m, dtype=np.float64)

        active = np.ones(m, dtype=bool)
        raw_exit_px = closes[np.minimum(entries + holds - 1, n_bars - 1)].copy()
        max_h = int(np.max(holds))

        for step in range(max_h):
            if not np.any(active):
                break
            idx_cur = entries + step
            in_bounds = active & (idx_cur < n_bars)
            if not np.any(in_bounds):
                break

            pos_idx = np.where(in_bounds)[0]
            bars = idx_cur[pos_idx]
            d_sub = dirs[pos_idx]
            o_sub = opens[bars]
            h_sub = highs[bars]
            l_sub = lows[bars]
            sp_sub = stop_px[pos_idx]

            # Gap stop on bars after entry
            gap_long = (step > 0) & (d_sub > 0) & (o_sub <= sp_sub)
            gap_short = (step > 0) & (d_sub < 0) & (o_sub >= sp_sub)
            gap_hit = gap_long | gap_short
            if np.any(gap_hit):
                hit_i = pos_idx[gap_hit]
                raw_exit_px[hit_i] = o_sub[gap_hit]
                active[hit_i] = False

            rem_mask = active[pos_idx]
            if np.any(rem_mask):
                p2 = pos_idx[rem_mask]
                d2 = dirs[p2]
                h2 = highs[entries[p2] + step]
                l2 = lows[entries[p2] + step]
                sp2 = stop_px[p2]
                stop_hit = ((d2 > 0) & (l2 <= sp2)) | ((d2 < 0) & (h2 >= sp2))
                if np.any(stop_hit):
                    hi2 = p2[stop_hit]
                    raw_exit_px[hi2] = sp2[stop_hit]
                    active[hi2] = False

            if has_tp:
                rem_mask2 = active[pos_idx]
                if np.any(rem_mask2):
                    p3 = pos_idx[rem_mask2]
                    d3 = dirs[p3]
                    h3 = highs[entries[p3] + step]
                    l3 = lows[entries[p3] + step]
                    tp3 = tp_px[p3]
                    tp_hit = ((d3 > 0) & (h3 >= tp3)) | ((d3 < 0) & (l3 <= tp3))
                    if np.any(tp_hit):
                        hi3 = p3[tp_hit]
                        raw_exit_px[hi3] = tp3[tp_hit]
                        active[hi3] = False

            # Positions that survived this bar pay/receive the funding settled in (open, next open]
            surv = active[pos_idx]
            if funding_enabled and np.any(surv):
                ps = pos_idx[surv]
                b_s = entries[ps] + step
                funding_acc[ps] += -dirs[ps] * closes[b_s] * bar_funding[b_s]

            # Time stop at end of holding horizon -> exit at next open (or close if at end)
            rem_mask3 = active[pos_idx]
            if np.any(rem_mask3):
                p4 = pos_idx[rem_mask3]
                time_hit = (step + 1) >= holds[p4]
                if np.any(time_hit):
                    hi4 = p4[time_hit]
                    next_b = np.minimum(entries[hi4] + step + 1, n_bars - 1)
                    raw_exit_px[hi4] = opens[next_b]
                    active[hi4] = False

        exit_px = raw_exit_px * (1.0 - dirs * slip)
        gross_per_unit = dirs * (exit_px - entry_px)
        fee_per_unit = (entry_px + exit_px) * fee_rate
        r_vals = (gross_per_unit - fee_per_unit + funding_acc) / sd
        r_mat = r_vals.reshape(n_runs, n_sym)
        total_r_sum_per_run += np.sum(r_mat, axis=1)

    if total_trades_count == 0:
        return {
            "n_runs": 0,
            "real_expectancy_r": 0.0,
            "random_mean_expectancy_r": 0.0,
            "random_std_expectancy_r": 0.0,
            "p_value": 1.0,
            "baseline_expectancies": [],
        }

    baseline_exp = total_r_sum_per_run / float(total_trades_count)
    real_exp = float(real_result.metrics["expectancy_r"])
    p_value = float((1 + np.sum(baseline_exp >= real_exp)) / (n_runs + 1))

    return {
        "n_runs": n_runs,
        "real_expectancy_r": round(real_exp, 4),
        "random_mean_expectancy_r": round(float(np.mean(baseline_exp)), 4),
        "random_std_expectancy_r": round(float(np.std(baseline_exp, ddof=1)), 4),
        "p_value": round(p_value, 4),
        "baseline_expectancies": [round(float(x), 4) for x in baseline_exp],
    }


def run_montecarlo_audit(
    data: MarketData,
    real_result: BacktestResult,
    params: dict[str, Any],
    mc_simulations: int = 1000,
    random_runs: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Run both trade-R bootstrap and random-entry baseline significance audits."""
    mc = run_trade_bootstrap(
        real_result.trades,
        risk_per_trade=real_result.config.risk_per_trade,
        n_simulations=mc_simulations,
        seed=seed,
    )
    rb = run_random_entry_baseline(
        data=data,
        real_result=real_result,
        params=params,
        n_runs=random_runs,
        seed=seed + 1,
    )

    p_val = float(rb["p_value"])
    prob_loss = float(mc["prob_loss"])
    if prob_loss > 0.65 or p_val > 0.25:
        status = "fail"
        explanation = (
            f"Weak statistical edge: random-entry p={p_val:.3f}, "
            f"bootstrap loss probability {prob_loss * 100:.1f}%."
        )
    elif prob_loss > 0.35 or p_val > 0.10:
        status = "warn"
        explanation = (
            f"Borderline significance: random-entry p={p_val:.3f}, "
            f"bootstrap 95th-pct drawdown {mc['p95_max_drawdown'] * 100:.1f}%."
        )
    else:
        status = "pass"
        explanation = (
            f"Outperforms random entries (p={p_val:.3f}); bootstrap median return "
            f"{mc['median_return'] * 100:+.1f}% with {mc['prob_loss'] * 100:.1f}% loss risk."
        )

    return {
        "name": "montecarlo",
        "status": status,
        "explanation": explanation,
        "bootstrap": mc,
        "random_baseline": rb,
    }
