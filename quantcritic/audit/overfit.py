"""Overfitting audits: Deflated Sharpe Ratio (DSR), CSCV PBO, and parameter stability."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, rankdata, skew

from quantcritic.config import BacktestConfig
from quantcritic.data.loader import MarketData
from quantcritic.engine.backtest import prepare_market, run_backtest
from quantcritic.strategies.base import Strategy

EULER_GAMMA = 0.5772156649015329


def compute_deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    trial_sharpes: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio on periodic returns."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    t_len = len(r)
    if t_len < 10:
        return {"dsr": 0.5, "sr_observed": 0.0, "sr_benchmark": 0.0, "skew": 0.0, "kurtosis": 3.0}

    mean_r = float(np.mean(r))
    std_r = float(np.std(r, ddof=1))
    if std_r <= 1e-12:
        return {"dsr": 0.5, "sr_observed": 0.0, "sr_benchmark": 0.0, "skew": 0.0, "kurtosis": 3.0}

    sr_obs = mean_r / std_r  # Non-annualized per-period Sharpe
    sk = float(skew(r, bias=False))
    ku = float(kurtosis(r, fisher=False, bias=False))  # Pearson kurtosis (normal = 3.0)

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sr_var = float(np.var(trial_sharpes, ddof=1))
    else:
        sr_var = 1.0 / max(t_len, 2)

    if n_trials <= 1 or sr_var <= 1e-16:
        sr0 = 0.0
    else:
        z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
        z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
        sr0 = math.sqrt(sr_var) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)

    denom_sq = (1.0 - sk * sr_obs + ((ku - 1.0) / 4.0) * (sr_obs ** 2)) / max(t_len - 1, 1)
    denom = math.sqrt(max(denom_sq, 1e-12))
    z_stat = (sr_obs - sr0) / denom
    dsr = float(norm.cdf(z_stat))

    return {
        "dsr": round(dsr, 4),
        "sr_observed": round(sr_obs, 4),
        "sr_benchmark": round(sr0, 4),
        "skew": round(sk, 4),
        "kurtosis": round(ku, 4),
    }


def compute_cscv_pbo(
    return_matrix: np.ndarray,
    n_blocks: int = 10,
) -> dict[str, Any]:
    """Compute Probability of Backtest Overfitting via CSCV (Bailey et al. 2015).

    Args:
        return_matrix: Array of shape (T_periods, N_strategies).
        n_blocks: Even number of chronological blocks S (default 10 -> 252 splits).
    """
    mat = np.asarray(return_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] < 2 or mat.shape[0] < n_blocks * 5:
        return {"pbo": 0.5, "n_splits": 0, "logits": []}

    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even for CSCV")

    t_rows = (mat.shape[0] // n_blocks) * n_blocks
    mat = mat[:t_rows]
    blocks = np.array_split(mat, n_blocks, axis=0)
    n_half = n_blocks // 2
    n_strats = mat.shape[1]

    logits: list[float] = []
    underperform_count = 0

    # Precompute per-block sums & sum-of-squares for fast exact Sharpe over subsets
    block_n = np.array([b.shape[0] for b in blocks], dtype=np.float64)
    block_sum = np.vstack([np.sum(b, axis=0) for b in blocks])  # (S, M)
    block_sq = np.vstack([np.sum(b * b, axis=0) for b in blocks])  # (S, M)
    total_n = float(np.sum(block_n))
    total_sum = np.sum(block_sum, axis=0)
    total_sq = np.sum(block_sq, axis=0)

    for is_indices in itertools.combinations(range(n_blocks), n_half):
        is_mask = list(is_indices)
        n_is = float(np.sum(block_n[is_mask]))
        sum_is = np.sum(block_sum[is_mask], axis=0)
        sq_is = np.sum(block_sq[is_mask], axis=0)

        mean_is = sum_is / n_is
        var_is = np.maximum((sq_is - (sum_is ** 2) / n_is) / max(n_is - 1.0, 1.0), 0.0)
        std_is = np.sqrt(var_is)
        sr_is = np.divide(mean_is, std_is, out=np.zeros_like(mean_is), where=std_is > 1e-12)

        n_oos = total_n - n_is
        sum_oos = total_sum - sum_is
        sq_oos = total_sq - sq_is
        mean_oos = sum_oos / n_oos
        var_oos = np.maximum((sq_oos - (sum_oos ** 2) / n_oos) / max(n_oos - 1.0, 1.0), 0.0)
        std_oos = np.sqrt(var_oos)
        sr_oos = np.divide(mean_oos, std_oos, out=np.zeros_like(mean_oos), where=std_oos > 1e-12)

        best_is = int(np.argmax(sr_is))
        # Relative rank in (0, 1)
        ranks = rankdata(sr_oos, method="average") / (n_strats + 1.0)
        w_bar = float(np.clip(ranks[best_is], 1e-6, 1.0 - 1e-6))
        logit = float(math.log(w_bar / (1.0 - w_bar)))
        logits.append(round(logit, 4))
        if w_bar < 0.5:
            underperform_count += 1

    n_splits = len(logits)
    pbo = float(underperform_count / n_splits) if n_splits > 0 else 0.5
    return {
        "pbo": round(pbo, 4),
        "n_splits": n_splits,
        "median_logit": round(float(np.median(logits)) if logits else 0.0, 4),
        "logits": logits,
    }


def compute_parameter_stability(
    strategy: Strategy,
    grid_results: list[dict[str, Any]],
    best_params: dict[str, Any],
) -> dict[str, Any]:
    """Measure neighbor median Sharpe vs best Sharpe across grid axes (plateau vs spike)."""
    if not grid_results or not strategy.param_grid:
        return {
            "best_sharpe": 0.0,
            "neighbor_median_sharpe": 0.0,
            "stability_ratio": 1.0,
            "classification": "plateau",
            "n_neighbors": 0,
        }

    axes = sorted(strategy.param_grid.keys())
    pos_by_val = {k: {v: idx for idx, v in enumerate(strategy.param_grid[k])} for k in axes}

    def _coord(params: dict[str, Any]) -> tuple[int, ...] | None:
        coords: list[int] = []
        for k in axes:
            val = params.get(k)
            if val not in pos_by_val[k]:
                return None
            coords.append(pos_by_val[k][val])
        return tuple(coords)

    best_coord = _coord(best_params)
    best_sharpe = max((float(r["sharpe"]) for r in grid_results), default=0.0)
    for r in grid_results:
        if r["params"] == best_params:
            best_sharpe = float(r["sharpe"])
            break

    neighbor_sharpes: list[float] = []
    if best_coord is not None:
        for r in grid_results:
            c = _coord(r["params"])
            if c is None or c == best_coord:
                continue
            chebyshev = max(abs(a - b) for a, b in zip(c, best_coord))
            if chebyshev == 1:
                neighbor_sharpes.append(float(r["sharpe"]))

    if not neighbor_sharpes:
        neighbor_sharpes = [float(r["sharpe"]) for r in grid_results]

    med_neighbor = float(np.median(neighbor_sharpes))
    if best_sharpe > 0.05:
        ratio = float(np.clip(med_neighbor / best_sharpe, 0.0, 1.5))
    else:
        ratio = 1.0 if med_neighbor >= best_sharpe else 0.0

    if ratio >= 0.65:
        label = "plateau"
    elif ratio >= 0.35:
        label = "moderate"
    else:
        label = "spike"

    return {
        "best_sharpe": round(best_sharpe, 4),
        "neighbor_median_sharpe": round(med_neighbor, 4),
        "stability_ratio": round(ratio, 4),
        "classification": label,
        "n_neighbors": len(neighbor_sharpes),
    }


def _build_heatmap_data(
    strategy: Strategy,
    grid_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate grid Sharpe into a 2D matrix over the first two param_grid axes for reporting."""
    axes = sorted(strategy.param_grid.keys())
    if not grid_results or len(axes) == 0:
        return {"x_param": "", "y_param": "", "x_vals": [], "y_vals": [], "z_matrix": []}

    x_param = axes[0]
    y_param = axes[1] if len(axes) >= 2 else axes[0]
    x_vals = [str(v) for v in strategy.param_grid[x_param]]
    y_vals = [str(v) for v in strategy.param_grid[y_param]] if len(axes) >= 2 else ["all"]

    z_mat: list[list[float | None]] = []
    for yv in (strategy.param_grid[y_param] if len(axes) >= 2 else [None]):
        row: list[float | None] = []
        for xv in strategy.param_grid[x_param]:
            vals = [
                float(r["sharpe"])
                for r in grid_results
                if r["params"].get(x_param) == xv
                and (yv is None or r["params"].get(y_param) == yv)
            ]
            row.append(round(float(np.mean(vals)), 3) if vals else None)
        z_mat.append(row)

    return {
        "x_param": x_param,
        "y_param": y_param if len(axes) >= 2 else "strategy",
        "x_vals": x_vals,
        "y_vals": y_vals,
        "z_matrix": z_mat,
    }


def run_overfit_audit(
    train_val_data: MarketData,
    strategy: Strategy,
    best_params: dict[str, Any],
    config: BacktestConfig | None = None,
    n_blocks: int = 10,
    train_grid_results: list[dict[str, Any]] | None = None,
    precomputed_signals: list[tuple[dict[str, Any], dict[str, pd.DataFrame]]] | None = None,
    trade_start: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Run DSR, CSCV PBO, and parameter stability audits strictly on train+validation data."""
    cfg = config or BacktestConfig(interval=train_val_data.interval)
    prep = prepare_market(train_val_data)

    if precomputed_signals:
        combos_and_sigs = precomputed_signals
        combos = [p for p, _ in combos_and_sigs]
    else:
        combos = strategy.expand_param_grid()
        if not combos:
            combos = [strategy.merge_params(best_params)]
        combos_and_sigs = [(p, strategy) for p in combos]  # type: ignore[list-item]

    daily_ret_cols: list[pd.Series] = []
    trial_daily_sharpes: list[float] = []
    eval_grid_rows: list[dict[str, Any]] = []
    selected_rets = np.array([], dtype=np.float64)

    for p, src in combos_and_sigs:
        res = run_backtest(
            train_val_data,
            src,
            p,
            cfg,
            prepared_market=prep,
            fast_metrics=True,
            trade_start=trade_start,
        )
        eq = res.equity_curve
        daily_eq = eq.resample("1D").last().dropna() if not eq.empty else pd.Series(dtype=float)
        d_ret = daily_eq.pct_change().dropna()
        if not d_ret.empty:
            daily_ret_cols.append(d_ret.rename(str(len(daily_ret_cols))))
            std_d = float(d_ret.std(ddof=1))
            sr_d = float(d_ret.mean() / std_d) if std_d > 1e-12 else 0.0
            trial_daily_sharpes.append(sr_d)
        else:
            trial_daily_sharpes.append(0.0)

        eval_grid_rows.append(
            {
                "params": dict(p),
                "sharpe": float(res.metrics["sharpe"]),
                "total_return": float(res.metrics["total_return"]),
                "n_trades": int(res.metrics["n_trades"]),
            }
        )
        if p == best_params and not d_ret.empty:
            selected_rets = d_ret.to_numpy(dtype=np.float64)

    if len(selected_rets) == 0:
        res_sel = run_backtest(
            train_val_data, strategy, best_params, cfg, trade_start=trade_start
        )
        d_sel = res_sel.equity_curve.resample("1D").last().dropna().pct_change().dropna()
        selected_rets = d_sel.to_numpy(dtype=np.float64)

    dsr_res = compute_deflated_sharpe_ratio(
        returns=selected_rets,
        n_trials=len(combos),
        trial_sharpes=np.asarray(trial_daily_sharpes, dtype=np.float64),
    )

    if len(daily_ret_cols) >= 2:
        mat_df = pd.concat(daily_ret_cols, axis=1).fillna(0.0)
        pbo_res = compute_cscv_pbo(mat_df.to_numpy(dtype=np.float64), n_blocks=n_blocks)
    else:
        pbo_res = {"pbo": 0.5, "n_splits": 0, "median_logit": 0.0, "logits": []}

    stability_source = train_grid_results if train_grid_results else eval_grid_rows
    stab_res = compute_parameter_stability(strategy, stability_source, best_params)
    heatmap = _build_heatmap_data(strategy, stability_source)

    pbo_val = float(pbo_res["pbo"])
    dsr_val = float(dsr_res["dsr"])
    if pbo_val > 0.70 or (dsr_val < 0.20 and pbo_val > 0.55):
        status = "fail"
        explanation = (
            f"High overfitting risk: CSCV PBO={pbo_val:.2f}, DSR={dsr_val:.2f}, "
            f"param stability={stab_res['classification']}."
        )
    elif pbo_val > 0.45 or dsr_val < 0.50 or stab_res["classification"] == "spike":
        status = "warn"
        explanation = (
            f"Moderate overfitting risk: DSR={dsr_val:.2f}, PBO={pbo_val:.2f}, "
            f"param surface={stab_res['classification']} ({stab_res['stability_ratio']:.2f}x)."
        )
    else:
        status = "pass"
        explanation = (
            f"Low overfitting risk: DSR={dsr_val:.2f}, PBO={pbo_val:.2f}, "
            f"neighbor plateau ratio={stab_res['stability_ratio']:.2f}x."
        )

    return {
        "name": "overfit",
        "status": status,
        "explanation": explanation,
        "n_trials": len(combos),
        "dsr": dsr_res,
        "pbo": pbo_res,
        "stability": stab_res,
        "heatmap": heatmap,
    }
