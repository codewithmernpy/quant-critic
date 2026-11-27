"""Transaction cost stress test: fee multiplier and slippage bps sensitivity grid."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.config import BacktestConfig
from quantcritic.data.loader import MarketData
from quantcritic.engine.backtest import prepare_market, run_backtest
from quantcritic.strategies.base import Strategy


def run_cost_stress_audit(
    data: MarketData,
    strategy: Strategy,
    params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    fee_multipliers: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
    slippage_bps_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0),
    trade_start: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Evaluate expectancy and return across a fee multiplier x slippage bps grid."""
    cfg_base = config or BacktestConfig(interval=data.interval)
    used_params = strategy.merge_params(params)
    signals_map = {sym: strategy.generate(df, used_params) for sym, df in data.ohlcv.items()}
    prep = prepare_market(data)

    rows: list[dict[str, Any]] = []
    expectancy_matrix: list[list[float]] = []

    for mult in fee_multipliers:
        fee_val = float(cfg_base.fee_rate) * float(mult)
        row_vals: list[float] = []
        for slip_bps in slippage_bps_grid:
            c_cfg = cfg_base.model_copy(update={"fee_rate": fee_val, "slippage_bps": float(slip_bps)})
            res = run_backtest(
                data,
                signals_map,
                used_params,
                c_cfg,
                prepared_market=prep,
                fast_metrics=True,
                trade_start=trade_start,
            )
            m = res.metrics
            one_way_bps = fee_val * 10_000.0 + float(slip_bps)
            rows.append(
                {
                    "fee_mult": float(mult),
                    "fee_bps": round(fee_val * 10_000.0, 2),
                    "slippage_bps": float(slip_bps),
                    "one_way_cost_bps": round(one_way_bps, 2),
                    "expectancy_r": float(m["expectancy_r"]),
                    "total_return": float(m["total_return"]),
                    "sharpe": float(m["sharpe"]),
                    "profit_factor": float(m["profit_factor"]),
                }
            )
            row_vals.append(float(m["expectancy_r"]))
        expectancy_matrix.append(row_vals)

    # Estimate break-even one-way cost in bps by fitting/interpolating expectancy vs total one-way cost
    sorted_pts = sorted(rows, key=lambda r: r["one_way_cost_bps"])
    costs_arr = np.array([r["one_way_cost_bps"] for r in sorted_pts], dtype=np.float64)
    exp_arr = np.array([r["expectancy_r"] for r in sorted_pts], dtype=np.float64)

    break_even_bps = 0.0
    max_tested_bps = float(costs_arr.max()) if len(costs_arr) else 0.0
    if len(costs_arr) >= 2:
        # Linear fit expectancy_r = a + b * one_way_cost_bps (R is ~linear in per-side cost)
        slope, intercept = np.polyfit(costs_arr, exp_arr, 1)
        if exp_arr[0] <= 0.0:
            break_even_bps = 0.0
        elif slope < -1e-9:
            break_even_bps = float(max(0.0, -intercept / slope))
        else:
            # No measurable cost sensitivity: no zero crossing, report the largest cost tested
            break_even_bps = max_tested_bps
    break_even_extrapolated = bool(break_even_bps > max_tested_bps or (len(costs_arr) >= 2 and exp_arr[-1] > 0.0))
    be_note = f" (extrapolated beyond the {max_tested_bps:.0f} bps tested)" if break_even_extrapolated else ""

    # Look up standard 1x + 2bps and stressed 2x + 5bps cells
    base_cell = next(
        (r for r in rows if abs(r["fee_mult"] - 1.0) < 1e-6 and abs(r["slippage_bps"] - 2.0) < 1e-6),
        rows[0],
    )
    stress_cell = next(
        (r for r in rows if abs(r["fee_mult"] - 2.0) < 1e-6 and abs(r["slippage_bps"] - 5.0) < 1e-6),
        rows[-1],
    )

    base_exp = float(base_cell["expectancy_r"])
    stress_exp = float(stress_cell["expectancy_r"])

    if base_exp <= 0.0:
        status = "fail"
        explanation = (
            f"Unprofitable at baseline costs (1x fee + 2 bps -> {base_exp:+.3f}R, "
            f"break-even {break_even_bps:.1f} bps/side)."
        )
    elif stress_exp <= 0.0 or break_even_bps < 15.0:
        status = "warn"
        explanation = (
            f"Thin cost cushion: breaks even at {break_even_bps:.1f} bps/side{be_note} "
            f"(2x fee + 5 bps stress expectancy: {stress_exp:+.3f}R)."
        )
    else:
        status = "pass"
        explanation = (
            f"Cost-resilient edge: break-even at {break_even_bps:.1f} bps/side{be_note}; "
            f"still positive ({stress_exp:+.3f}R) under 2x fees + 5 bps slippage."
        )

    return {
        "name": "costs",
        "status": status,
        "explanation": explanation,
        "fee_multipliers": list(fee_multipliers),
        "slippage_bps_grid": list(slippage_bps_grid),
        "expectancy_matrix": expectancy_matrix,
        "grid_rows": rows,
        "base_expectancy_r": round(base_exp, 4),
        "stress_2x_5bps_expectancy_r": round(stress_exp, 4),
        "break_even_cost_bps": round(break_even_bps, 2),
        "break_even_extrapolated": break_even_extrapolated,
    }
