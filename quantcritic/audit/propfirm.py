"""Generic two-step prop-firm evaluation simulation across rolling start weeks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.config import PropFirmConfig
from quantcritic.engine.backtest import BacktestResult


def run_propfirm_audit(
    result: BacktestResult,
    prop_cfg: PropFirmConfig | None = None,
) -> dict[str, Any]:
    """Simulate a two-step prop-firm challenge starting from every possible start week."""
    cfg = prop_cfg or PropFirmConfig()
    eq = result.equity_curve
    trades = result.trades

    if eq.empty or len(eq) < 30:
        return {
            "name": "propfirm",
            "status": "fail",
            "explanation": "Insufficient equity history to simulate prop-firm challenges.",
            "n_attempts": 0,
            "n_passed": 0,
            "n_failed": 0,
            "n_incomplete": 0,
            "pass_rate": 0.0,
            "median_days_to_pass": None,
            "p90_days_to_pass": None,
            "fail_reasons": {},
        }

    eq_sorted = eq.sort_index()
    idx = pd.DatetimeIndex(eq_sorted.index).as_unit("ns")
    eq_vals = eq_sorted.to_numpy(dtype=np.float64)
    # Per-bar return multipliers
    bar_mult = np.ones(len(eq_vals), dtype=np.float64)
    prev = np.where(eq_vals[:-1] > 0.0, eq_vals[:-1], 1.0)
    bar_mult[1:] = eq_vals[1:] / prev

    dates = idx.normalize()
    # Set of active trading UTC dates where trades entered/exited
    active_dates: set[pd.Timestamp] = set()
    if not trades.empty:
        for col in ("entry_time", "exit_time"):
            t_dates = pd.to_datetime(trades[col], utc=True).dt.normalize()
            active_dates.update(t_dates.tolist())

    is_active_bar_date = np.array([d in active_dates for d in dates], dtype=bool)
    date_ns = dates.asi8

    # Weekly start indices (every Monday or 7-day cadence with >= 14 days remaining)
    last_ns = int(idx[-1].value)
    min_rem_ns = 14 * 86400 * 1_000_000_000
    start_indices: list[int] = []
    next_start_ns = int(idx[0].value)
    week_ns = 7 * 86400 * 1_000_000_000
    idx_ns = idx.asi8

    for i in range(len(idx_ns)):
        if idx_ns[i] >= next_start_ns and (last_ns - idx_ns[i]) >= min_rem_ns:
            start_indices.append(i)
            next_start_ns = int(idx_ns[i]) + week_ns

    n_passed = 0
    n_failed = 0
    n_incomplete = 0
    fail_reasons: dict[str, int] = {}
    days_to_pass: list[float] = []

    t1 = 1.0 + float(cfg.step1_target)
    t2 = 1.0 + float(cfg.step2_target)
    max_dd_limit = 1.0 - float(cfg.max_total_loss)
    daily_loss_mult = 1.0 - float(cfg.max_daily_loss)
    min_days = int(cfg.min_trading_days)
    n_bars = len(eq_vals)

    for s_idx in start_indices:
        phase = 1
        equity = 1.0
        day_start_eq = 1.0
        cur_day = int(date_ns[s_idx])
        trading_days_phase: set[int] = set()
        start_ns = int(idx_ns[s_idx])
        outcome = "incomplete"

        for j in range(s_idx + 1, n_bars):
            d_j = int(date_ns[j])
            if d_j != cur_day:
                cur_day = d_j
                day_start_eq = equity

            equity *= float(bar_mult[j])
            if is_active_bar_date[j]:
                trading_days_phase.add(d_j)

            # Check static max loss and daily drawdown breach
            if equity <= max_dd_limit:
                reason = f"phase{phase}_max_loss"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                outcome = "fail"
                break
            if equity <= day_start_eq * daily_loss_mult:
                reason = f"phase{phase}_daily_loss"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                outcome = "fail"
                break

            target = t1 if phase == 1 else t2
            if equity >= target and len(trading_days_phase) >= min_days:
                if phase == 1:
                    phase = 2
                    equity = 1.0
                    day_start_eq = 1.0
                    cur_day = d_j
                    trading_days_phase = set()
                else:
                    elapsed_days = (int(idx_ns[j]) - start_ns) / (86400.0 * 1e9)
                    days_to_pass.append(elapsed_days)
                    outcome = "pass"
                    break

        if outcome == "pass":
            n_passed += 1
        elif outcome == "fail":
            n_failed += 1
        else:
            n_incomplete += 1

    n_attempts = len(start_indices)
    pass_rate = float(n_passed / n_attempts) if n_attempts > 0 else 0.0
    med_days = round(float(np.median(days_to_pass)), 1) if days_to_pass else None
    p90_days = round(float(np.quantile(days_to_pass, 0.90)), 1) if days_to_pass else None

    if pass_rate >= 0.40:
        status = "pass"
        explanation = (
            f"Passed {n_passed}/{n_attempts} two-step evaluations ({pass_rate * 100:.1f}%); "
            f"median time to pass: {med_days} days."
        )
    elif pass_rate >= 0.15:
        status = "warn"
        explanation = (
            f"Moderate prop pass rate: {n_passed}/{n_attempts} ({pass_rate * 100:.1f}%); "
            f"{n_incomplete} incomplete, {n_failed} failed."
        )
    else:
        status = "fail"
        explanation = (
            f"Low prop pass rate: {n_passed}/{n_attempts} ({pass_rate * 100:.1f}%). "
            f"Failures: {fail_reasons or 'timed out'}."
        )

    return {
        "name": "propfirm",
        "status": status,
        "explanation": explanation,
        "n_attempts": n_attempts,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_incomplete": n_incomplete,
        "pass_rate": round(pass_rate, 4),
        "median_days_to_pass": med_days,
        "p90_days_to_pass": p90_days,
        "fail_reasons": fail_reasons,
        "config": cfg.model_dump(),
    }
