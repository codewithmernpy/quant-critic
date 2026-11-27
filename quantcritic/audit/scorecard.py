"""Weighted robustness scorecard and final verdict calculator.

Scoring rule (documented in docs/METHODOLOGY.md, section 2.7):

1. Seven 0-100 sub-scores are combined with fixed weights. Only out-of-sample evidence earns
   "edge" points: the pre-test OOS evidence (validation + rolling walk-forward + random-entry
   baseline) and the locked TEST slice are separate sub-scores. In-sample (train) performance
   earns nothing directly.
2. Hard gates then cap the score and the verdict according to the selected `strictness` mode
   (`"strict"`, `"balanced"` [default], `"relaxed"`). In ALL modes, lookahead truncation failure
   triggers `REJECT` (score <= 20) and a losing locked test can never be `PROMISING`.
3. Verdict from the capped score: >= promising_threshold PROMISING (70 strict / 65 balanced /
   60 relaxed), >= 45 NEEDS WORK, < 45 REJECT.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from quantcritic.config import DEFAULT_STRICTNESS, STRICTNESS_MODES
from quantcritic.risk.sizing import estimate_risk_of_ruin

WEIGHTS = {
    "integrity": 0.15,
    "oos_edge": 0.20,
    "locked_test": 0.20,
    "robustness": 0.15,
    "cost_resilience": 0.10,
    "consistency": 0.10,
    "risk": 0.10,
}

PROMISING_THRESHOLD = 70.0
NEEDS_WORK_THRESHOLD = 45.0
MIN_TEST_TRADES = 10

STRICTNESS_RULES: dict[str, dict[str, Any]] = {
    "strict": {
        "promising_threshold": 70.0,
        "promising_cap": 69.9,
        "min_test_trades": 10,
        "max_p_value": 0.10,
        "min_dsr": 0.50,
    },
    "balanced": {
        "promising_threshold": 65.0,
        "promising_cap": 64.9,
        "min_test_trades": 10,
        "max_p_value": 0.20,
        "min_dsr": 0.30,
        "preoos_min_test_trades_override": 20,
    },
    "relaxed": {
        "promising_threshold": 60.0,
        "promising_cap": 59.9,
        "min_test_trades": 5,
        "max_p_value": None,
        "min_dsr": None,
    },
}

_VERDICT_RANK = {"REJECT": 0, "NEEDS WORK": 1, "PROMISING": 2}


def normalize_strictness(strictness: str | None) -> Literal["strict", "balanced", "relaxed"]:
    """Validate and normalize a strictness mode name."""
    mode = (strictness or DEFAULT_STRICTNESS).strip().lower()
    if mode not in STRICTNESS_MODES:
        raise ValueError(f"Unknown strictness mode {strictness!r}. Expected one of {STRICTNESS_MODES}.")
    return mode  # type: ignore[return-value]


def _clamp100(val: float) -> float:
    return float(round(min(100.0, max(0.0, val)), 1))


def verdict_from_score(score: float, strictness: str = DEFAULT_STRICTNESS) -> str:
    mode = normalize_strictness(strictness)
    promising_threshold = float(STRICTNESS_RULES[mode]["promising_threshold"])
    if score >= promising_threshold:
        return "PROMISING"
    if score >= NEEDS_WORK_THRESHOLD:
        return "NEEDS WORK"
    return "REJECT"


def cap_verdict(verdict: str, ceiling: str) -> str:
    """Return the stricter of `verdict` and `ceiling` (REJECT < NEEDS WORK < PROMISING)."""
    v = verdict if verdict in _VERDICT_RANK else "NEEDS WORK"
    return v if _VERDICT_RANK[v] <= _VERDICT_RANK[ceiling] else ceiling


def _period_lost(metrics: dict[str, Any] | None) -> bool:
    """A period 'lost' if Sharpe, total return or expectancy is not positive."""
    if not metrics:
        return True
    return (
        float(metrics.get("sharpe", 0.0)) <= 0.0
        or float(metrics.get("total_return", 0.0)) <= 0.0
        or float(metrics.get("expectancy_r", 0.0)) <= 0.0
    )


def compute_scorecard(
    lookahead_res: dict[str, Any],
    walkforward_res: dict[str, Any],
    overfit_res: dict[str, Any],
    montecarlo_res: dict[str, Any],
    cost_res: dict[str, Any],
    regime_res: dict[str, Any],
    full_metrics: dict[str, Any],
    risk_per_trade: float = 0.005,
    test_metrics: dict[str, Any] | None = None,
    trades: pd.DataFrame | None = None,
    strictness: str = DEFAULT_STRICTNESS,
) -> dict[str, Any]:
    """Compute 0-100 sub-scores, strictness-aware hard-gated overall score, and deterministic verdict."""
    mode = normalize_strictness(strictness)
    rules = STRICTNESS_RULES[mode]
    promising_threshold = float(rules["promising_threshold"])
    # 1. Integrity (0-100)
    trunc_passed = bool(lookahead_res["truncation"]["passed"])
    delay_collapse = float(lookahead_res["delay"]["expectancy_collapse_pct"])
    if not trunc_passed:
        integrity_score = 0.0
    elif lookahead_res["delay"]["warn"]:
        integrity_score = _clamp100(55.0 - max(0.0, delay_collapse - 70.0))
    else:
        integrity_score = _clamp100(100.0 - max(0.0, delay_collapse) * 0.35)

    # 2. Pre-test OOS edge (validation + rolling WF Sharpe, validation expectancy, random-entry p)
    val_m = walkforward_res.get("validation_metrics", {})
    wf_m = walkforward_res.get("rolling_oos_metrics", {})
    val_exp = float(val_m.get("expectancy_r", 0.0))
    val_sh = float(val_m.get("sharpe", 0.0))
    wf_sh = float(wf_m.get("sharpe", val_sh))
    p_val = float(montecarlo_res["random_baseline"].get("p_value", 1.0))

    oos_sh_blend = 0.5 * val_sh + 0.5 * wf_sh
    sh_pts = np.clip((oos_sh_blend + 0.2) / 1.6, 0.0, 1.0) * 45.0
    exp_pts = np.clip((val_exp + 0.05) / 0.30, 0.0, 1.0) * 25.0
    pval_pts = np.clip(1.0 - p_val / 0.30, 0.0, 1.0) * 30.0
    edge_score = _clamp100(float(sh_pts + exp_pts + pval_pts))

    # 3. Locked TEST slice (evaluated once, never used for any decision)
    test_trades = int(test_metrics.get("n_trades", 0)) if test_metrics else 0
    if test_metrics:
        test_sh = float(test_metrics.get("sharpe", 0.0))
        test_exp = float(test_metrics.get("expectancy_r", 0.0))
        test_score = _clamp100(
            float(
                np.clip((test_sh + 0.2) / 1.6, 0.0, 1.0) * 60.0
                + np.clip((test_exp + 0.05) / 0.30, 0.0, 1.0) * 40.0
            )
        )
        if test_trades < MIN_TEST_TRADES:
            test_score = _clamp100(test_score * test_trades / MIN_TEST_TRADES)
    else:
        test_score = 0.0

    # 4. Robustness (DSR, 1 - PBO, param stability)
    dsr = float(overfit_res["dsr"].get("dsr", 0.0))
    pbo = float(overfit_res["pbo"].get("pbo", 0.5))
    stab_ratio = float(overfit_res["stability"].get("stability_ratio", 0.5))
    rob_score = _clamp100(
        dsr * 40.0 + np.clip(1.0 - pbo, 0.0, 1.0) * 35.0 + np.clip(stab_ratio, 0.0, 1.0) * 25.0
    )

    # 5. Cost resilience
    be_bps = float(cost_res.get("break_even_cost_bps", 0.0))
    stress_exp = float(cost_res.get("stress_2x_5bps_expectancy_r", 0.0))
    be_pts = np.clip(be_bps / 35.0, 0.0, 1.0) * 60.0
    stress_pts = np.clip((stress_exp + 0.05) / 0.25, 0.0, 1.0) * 40.0
    cost_score = _clamp100(float(be_pts + stress_pts))

    # 6. Consistency (positive years, regime diversity, outlier survival)
    yearly = regime_res.get("yearly", [])
    pos_yrs = sum(1 for y in yearly if y["return"] > 0.0)
    yr_frac = pos_yrs / max(len(yearly), 1)
    regs = regime_res.get("regimes", {})
    pos_regs = sum(1 for r in regs.values() if r.get("n_trades", 0) > 0 and r.get("net_pnl", 0.0) > 0.0)
    reg_frac = pos_regs / 3.0
    outlier_ok = 1.0 if regime_res.get("outlier_check", {}).get("still_profitable", False) else 0.0
    consistency_score = _clamp100(yr_frac * 40.0 + reg_frac * 35.0 + outlier_ok * 25.0)

    # 7. Risk (actual max DD, MC 95th-pct DD, MC prob_loss, risk of ruin from the real payoff profile)
    mc = montecarlo_res.get("bootstrap", {})
    max_dd = float(full_metrics.get("max_drawdown", 0.5))
    mc_dd95 = float(mc.get("p95_max_drawdown", max_dd))
    prob_loss = float(mc.get("prob_loss", 0.5))
    if trades is not None and not trades.empty and "R" in trades.columns:
        r_arr = trades["R"].to_numpy(dtype=np.float64)
        wins, losses = r_arr[r_arr > 0.0], r_arr[r_arr <= 0.0]
        win_rate = float(len(wins) / len(r_arr))
        avg_win_r = float(wins.mean()) if len(wins) else 0.0
        avg_loss_r = float(abs(losses.mean())) if len(losses) and abs(losses.mean()) > 1e-9 else 1.0
        ror = estimate_risk_of_ruin(win_rate, avg_win_r, avg_loss_r, risk_per_trade=risk_per_trade)
    else:
        ror = 1.0

    dd_pts = np.clip(1.0 - max_dd / 0.40, 0.0, 1.0) * 35.0
    mc_dd_pts = np.clip(1.0 - mc_dd95 / 0.50, 0.0, 1.0) * 30.0
    ploss_pts = np.clip(1.0 - prob_loss, 0.0, 1.0) * 20.0
    ror_pts = np.clip(1.0 - ror, 0.0, 1.0) * 15.0
    risk_score = _clamp100(float(dd_pts + mc_dd_pts + ploss_pts + ror_pts))

    sub_scores = {
        "integrity": integrity_score,
        "oos_edge": edge_score,
        "locked_test": test_score,
        "robustness": rob_score,
        "cost_resilience": cost_score,
        "consistency": consistency_score,
        "risk": risk_score,
    }
    weighted_score = round(float(sum(sub_scores[k] * WEIGHTS[k] for k in WEIGHTS)), 1)

    # Hard gates by strictness mode (integrity and honesty rules always enforced)
    test_lost = _period_lost(test_metrics)
    pre_oos_lost = val_sh <= 0.0 or val_exp <= 0.0 or wf_sh <= 0.0
    gates: list[dict[str, Any]] = []
    if not trunc_passed or lookahead_res.get("status") == "fail":
        gates.append({"gate": "integrity", "cap": 20.0, "ceiling": "REJECT",
                      "reason": "Causal truncation test failed: signals use future data."})
    if mode in ("strict", "balanced"):
        if test_lost and pre_oos_lost:
            gates.append({"gate": "oos_fail", "cap": 40.0, "ceiling": "REJECT",
                          "reason": "Lost money both in pre-test OOS (validation/walk-forward) and on the locked test."})
    if test_lost:
        gates.append({"gate": "test_fail", "cap": 55.0, "ceiling": "NEEDS WORK",
                      "reason": "Locked 20% test slice lost (Sharpe, return or expectancy <= 0)."})
    if mode == "strict" and pre_oos_lost:
        gates.append({"gate": "preoos_fail", "cap": 60.0, "ceiling": "NEEDS WORK",
                      "reason": "Validation or rolling walk-forward OOS did not make money."})
    elif mode == "balanced" and pre_oos_lost and (not test_lost) and test_trades < int(rules["preoos_min_test_trades_override"]):
        gates.append({"gate": "preoos_fail", "cap": 60.0, "ceiling": "NEEDS WORK",
                      "reason": f"Validation or rolling walk-forward OOS did not make money and locked test has only {test_trades} trades (< 20)."})

    base_exp = float(cost_res.get("base_expectancy_r", 0.0))
    req_trades = int(rules["min_test_trades"])
    missing = []
    if test_trades < req_trades:
        missing.append(f"only {test_trades} test trades (< {req_trades})")
    if base_exp <= 0.0:
        missing.append("non-positive expectancy at base costs")
    max_p = rules["max_p_value"]
    if max_p is not None and p_val > float(max_p):
        missing.append(f"random-entry p={p_val:.3f} > {float(max_p):.2f}")
    min_dsr = rules["min_dsr"]
    if min_dsr is not None and dsr < float(min_dsr):
        missing.append(f"DSR={dsr:.2f} < {float(min_dsr):.2f}")
    if missing:
        gates.append({"gate": "promising_requirements", "cap": float(rules["promising_cap"]), "ceiling": "NEEDS WORK",
                      "reason": f"PROMISING requirements ({mode} mode) not met: " + "; ".join(missing) + "."})

    overall_score = weighted_score
    ceiling = "PROMISING"
    for g in gates:
        overall_score = min(overall_score, float(g["cap"]))
        ceiling = cap_verdict(ceiling, str(g["ceiling"]))
    overall_score = round(overall_score, 1)
    verdict = cap_verdict(verdict_from_score(overall_score, strictness=mode), ceiling)

    if gates:
        verdict_reason = gates[0]["reason"] if len(gates) == 1 else " ".join(g["reason"] for g in gates[:3])
    elif verdict == "PROMISING":
        if mode == "strict":
            verdict_reason = (
                "Causal, positive on validation, walk-forward AND the locked test, beats random entries, "
                "survives cost stress; suitable for paper-trading next."
            )
        elif mode == "balanced":
            verdict_reason = (
                "Causal, profitable on the locked test with balanced OOS/statistical thresholds; "
                "suitable for paper-trading next."
            )
        else:
            verdict_reason = (
                "Causal, profitable on the locked test with >= 5 test trades and positive base-cost "
                "expectancy under relaxed verdict gates; treat as exploratory."
            )
    elif verdict == "NEEDS WORK":
        verdict_reason = "Mixed robustness or thin OOS edge; requires refinement before risking capital."
    else:
        verdict_reason = "Insufficient OOS edge, poor cost resilience, or high overfitting risk."

    return {
        "strictness": mode,
        "promising_threshold": promising_threshold,
        "needs_work_threshold": NEEDS_WORK_THRESHOLD,
        "sub_scores": sub_scores,
        "weights": WEIGHTS,
        "weighted_score": weighted_score,
        "overall_score": overall_score,
        "verdict": verdict,
        "verdict_ceiling": ceiling,
        "gates": gates,
        "verdict_reason": verdict_reason,
        "risk_of_ruin": round(float(ror), 4),
    }
