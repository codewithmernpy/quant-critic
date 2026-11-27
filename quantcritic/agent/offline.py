"""Deterministic rule-based offline critic producing structured verdicts without an API key."""

from __future__ import annotations

import json
from typing import Any, Callable

from quantcritic.agent.tools import AuditSession, CriticVerdict
from quantcritic.audit.scorecard import STRICTNESS_RULES, normalize_strictness
from quantcritic.risk.sizing import capped_fractional_kelly


def synthesize_verdict_from_session(session: AuditSession) -> CriticVerdict:
    """Build a deterministic, evidence-citing CriticVerdict from completed audit results."""
    scorecard = session.ensure_complete_audit()
    la = session.cached_results["lookahead"]
    wf = session.cached_results["walkforward"]
    ov = session.cached_results["overfit"]
    mc = session.cached_results["montecarlo"]
    cs = session.cached_results["costs"]
    reg = session.cached_results["regime"]
    pf = session.cached_results["propfirm"]
    ft = session.cached_results["final_test"]
    full_m = session.cached_results["full_result"].metrics

    verdict = scorecard["verdict"]
    overall = float(scorecard["overall_score"])
    mode = normalize_strictness(scorecard.get("strictness") or session.strictness)
    mode_rules = STRICTNESS_RULES[mode]
    min_dsr_req = float(mode_rules["min_dsr"]) if mode_rules["min_dsr"] is not None else 0.30
    max_p_req = float(mode_rules["max_p_value"]) if mode_rules["max_p_value"] is not None else 0.20
    val_m = wf["validation_metrics"]
    oos_m = wf["rolling_oos_metrics"]
    test_m = ft["test_metrics"]
    dsr = float(ov["dsr"]["dsr"])
    pbo = float(ov["pbo"]["pbo"])
    stab = ov["stability"]
    bs = mc["bootstrap"]
    rb = mc["random_baseline"]

    strengths: list[str] = []
    weaknesses: list[str] = []
    red_flags: list[str] = []
    improvements: list[str] = []

    if la["truncation"]["passed"]:
        strengths.append(
            f"Passes all {la['truncation']['cuts_checked']} causal truncation checks with 0 mismatches."
        )
    else:
        red_flags.append(
            f"CRITICAL LOOKAHEAD LEAKAGE: {la['truncation']['total_mismatches']} mismatches across "
            f"{la['truncation']['cuts_with_mismatch']} truncation cuts; backtest edge is non-causal."
        )

    if la["delay"]["warn"]:
        red_flags.append(
            f"1-bar entry delay collapsed expectancy by {la['delay']['expectancy_collapse_pct']:.1f}% "
            "(fragile execution timing assumption)."
        )

    if val_m["sharpe"] >= 0.5 and val_m["expectancy_r"] > 0:
        strengths.append(
            f"Positive out-of-sample validation: Sharpe {val_m['sharpe']:.2f}, "
            f"expectancy {val_m['expectancy_r']:+.3f}R across {val_m['n_trades']} trades."
        )
    else:
        weaknesses.append(
            f"Weak OOS validation: Sharpe {val_m['sharpe']:.2f}, expectancy {val_m['expectancy_r']:+.3f}R "
            f"(train->val decay {wf['sharpe_degradation_pct']:+.0f}%)."
        )

    if test_m["sharpe"] >= 0.5 and test_m["total_return"] > 0:
        strengths.append(
            f"Held-out 20% locked test slice held up: Sharpe {test_m['sharpe']:.2f}, "
            f"return {test_m['total_return'] * 100:+.1f}%, max DD {test_m['max_drawdown'] * 100:.1f}%."
        )
    else:
        weaknesses.append(
            f"Locked 20% test slice underperformed: Sharpe {test_m['sharpe']:.2f}, "
            f"return {test_m['total_return'] * 100:+.1f}%, max DD {test_m['max_drawdown'] * 100:.1f}%."
        )

    if dsr >= min_dsr_req and pbo <= 0.45:
        strengths.append(
            f"Low selection bias: DSR={dsr:.2f}, CSCV PBO={pbo:.2f}, parameter surface is "
            f"{stab['classification']} ({stab['stability_ratio']:.2f}x)."
        )
    else:
        flag_list = red_flags if pbo > 0.65 else weaknesses
        flag_list.append(
            f"Overfitting risk elevated: DSR={dsr:.2f}, CSCV PBO={pbo:.2f}, "
            f"parameter stability={stab['classification']} ({stab['stability_ratio']:.2f}x)."
        )

    if rb["p_value"] <= max_p_req:
        strengths.append(
            f"Beats matched random entries: {rb['real_expectancy_r']:+.3f}R vs "
            f"{rb['random_mean_expectancy_r']:+.3f}R random mean (p={rb['p_value']:.3f})."
        )
    else:
        weaknesses.append(
            f"Does not clearly beat random entry timing (p={rb['p_value']:.3f}; "
            f"real {rb['real_expectancy_r']:+.3f}R vs random {rb['random_mean_expectancy_r']:+.3f}R)."
        )

    if cs["break_even_cost_bps"] >= 15.0 and cs["stress_2x_5bps_expectancy_r"] > 0:
        strengths.append(
            f"Resilient to taker costs: break-even at {cs['break_even_cost_bps']:.1f} bps/side; "
            f"2x fee + 5 bps stress retains {cs['stress_2x_5bps_expectancy_r']:+.3f}R."
        )
    else:
        red_flags.append(
            f"Cost-fragile edge: break-even at {cs['break_even_cost_bps']:.1f} bps/side "
            f"(2x fee + 5 bps expectancy: {cs['stress_2x_5bps_expectancy_r']:+.3f}R)."
        )

    if not reg["outlier_check"]["still_profitable"]:
        red_flags.append(
            f"Outlier-dependent: removing top 5% ({reg['outlier_check']['trades_removed']}) trades "
            f"drops PnL to ${reg['outlier_check']['trimmed_net_pnl']:+,.0f}."
        )

    bull_pnl = reg["regimes"].get("bull", {}).get("net_pnl", 0.0)
    bear_pnl = reg["regimes"].get("bear", {}).get("net_pnl", 0.0)
    chop_pnl = reg["regimes"].get("chop", {}).get("net_pnl", 0.0)
    if chop_pnl < 0:
        weaknesses.append(
            f"Chop regime drag: net PnL ${chop_pnl:+,.0f} in chop vs ${bull_pnl:+,.0f} bull / ${bear_pnl:+,.0f} bear."
        )

    # Hard gates from the scorecard are the decisive evidence: surface them explicitly
    for gate in scorecard.get("gates", []):
        target = weaknesses if gate["gate"] == "promising_requirements" else red_flags
        target.append(f"[{gate['gate']}] {gate['reason']}")

    # Crypto-specific risks
    if full_m["total_funding"] < -0.05 * full_m["initial_capital"]:
        weaknesses.append(
            f"Perpetual funding carry drag: ${full_m['total_funding']:+,.0f} net funding across full sample."
        )

    # Improvements are derived only from the weaknesses actually detected above
    test_lost = test_m["sharpe"] <= 0 or test_m["total_return"] <= 0 or test_m["expectancy_r"] <= 0
    if test_lost and val_m["sharpe"] > 0:
        improvements.append(
            f"Edge did not survive the most recent period (test {ft['test_period']['start'][:10]} to "
            f"{ft['test_period']['end'][:10]}: Sharpe {test_m['sharpe']:.2f}); treat validation results as "
            "regime-specific and require a fresh forward/paper period before allocating capital."
        )
    if not la["truncation"]["passed"]:
        improvements.append(
            "Remove every non-causal computation (centered windows, full-sample normalisation) and re-run "
            "the truncation test before evaluating performance again."
        )
    if pbo > 0.45 or stab["classification"] == "spike":
        improvements.append(
            f"Shrink the parameter grid / prefer plateau parameters: PBO={pbo:.2f}, neighbour ratio "
            f"{stab['stability_ratio']:.2f}x indicates selection on noise."
        )
    if rb["p_value"] > 0.10:
        improvements.append(
            f"Entry timing is not distinguishable from random (p={rb['p_value']:.3f}); test whether the "
            "result comes from the stop/exit rule alone before refining entries."
        )
    if cs["break_even_cost_bps"] < 15.0 or cs["stress_2x_5bps_expectancy_r"] <= 0:
        improvements.append(
            f"Cut turnover or test maker/limit execution: break-even is only {cs['break_even_cost_bps']:.1f} "
            "bps/side."
        )
    if chop_pnl < 0:
        improvements.append(
            f"Test a causal BTC-regime/volatility filter: chop-regime trades lost ${chop_pnl:+,.0f}."
        )
    if full_m["total_funding"] < -0.05 * full_m["initial_capital"]:
        improvements.append(
            "Add a causal funding-rate filter to avoid entering crowded high-funding positions."
        )
    if not reg["outlier_check"]["still_profitable"]:
        improvements.append(
            "Profit depends on a few outlier trades; require the edge to hold after removing the top 5%."
        )
    improvements.append(
        f"Size live risk for the bootstrap 95th-pct drawdown ({bs['p95_max_drawdown'] * 100:.1f}%); "
        f"two-step prop pass rate on train+val was {pf['pass_rate'] * 100:.1f}%."
    )

    if not weaknesses:
        weakest = min(scorecard["sub_scores"].items(), key=lambda kv: kv[1])
        weaknesses.append(f"Weakest scorecard dimension: {weakest[0]} at {weakest[1]:.1f}/100.")

    # Recommended fractional risk via capped 0.2x Kelly; only a PROMISING verdict earns live risk
    # (NEEDS WORK / REJECT -> 0: paper-trade or rework first)
    trades_df = session.cached_results["full_result"].trades
    rec_risk = 0.0
    if verdict == "PROMISING" and not trades_df.empty and full_m["avg_r"] > 0:
        wins_r = trades_df.loc[trades_df["R"] > 0, "R"]
        losses_r = trades_df.loc[trades_df["R"] < 0, "R"].abs()
        avg_win = float(wins_r.mean()) if not wins_r.empty else 1.5
        avg_loss = float(losses_r.mean()) if not losses_r.empty else 1.0
        kelly_r = capped_fractional_kelly(
            win_rate=float(full_m["win_rate"]),
            avg_win_r=avg_win,
            avg_loss_r=avg_loss,
            fraction=0.20,
            max_risk=0.01,
        )
        rec_risk = round(max(0.0025, min(kelly_r, 0.0075)), 4) if kelly_r > 0 else 0.0025

    summary = (
        f"QuantCritic offline audit scores `{session.strategy.name}` at {overall:.1f}/100 ({verdict}; "
        f"Verdict mode: {mode}). "
        f"Validation Sharpe is {val_m['sharpe']:.2f}, rolling walk-forward OOS Sharpe is "
        f"{oos_m.get('sharpe', 0.0):.2f}, locked 20% test Sharpe is {test_m['sharpe']:.2f} "
        f"({test_m['total_return'] * 100:+.1f}% return), random-entry p={rb['p_value']:.3f}, "
        f"DSR={dsr:.2f}, PBO={pbo:.2f}, and break-even cost is {cs['break_even_cost_bps']:.1f} bps/side. "
        f"Reason: {scorecard['verdict_reason']}"
    )

    return CriticVerdict(
        verdict=verdict,
        score=overall,
        summary=summary,
        strengths=strengths,
        weaknesses=weaknesses,
        red_flags=red_flags,
        improvements=improvements[:5],
        recommended_risk_per_trade=rec_risk,
        source="offline",
    )


def _is_step_completed(session: AuditSession, cache_key: str, tool_name: str) -> bool:
    """Return True if a deterministic audit step has already been computed and logged."""
    if cache_key == "final_test":
        return session.final_test_called
    if cache_key == "montecarlo":
        return cache_key in session.cached_results and any(
            entry.get("tool") == tool_name for entry in session.tool_log
        )
    return cache_key in session.cached_results


def run_offline_critic(
    session: AuditSession,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> CriticVerdict:
    """Execute the full deterministic audit sequence without calling any external API."""
    # Parameter selection (grid search on train) runs first so every later audit, including the
    # integrity/delay test, is computed for the parameters that will face the locked test.
    candidate_steps: list[tuple[str, str, str, Callable[[], Any]]] = [
        ("walkforward", "walk_forward", "Running train grid search, validation & rolling walk-forward", session.run_walkforward),
        ("overview", "get_overview", "Inspecting strategy overview & train+val baseline", session.get_overview),
        ("lookahead", "lookahead_test", "Running lookahead truncation & 1-bar delay tests", session.run_lookahead),
        ("overfit", "overfit_tests", "Computing DSR, CSCV PBO, and parameter stability", session.run_overfit),
        ("montecarlo", "monte_carlo", "Running 1000x Monte Carlo trade bootstrap", session.run_monte_carlo),
        ("montecarlo", "random_baseline", "Running 200x matched random-entry baseline", session.run_random_baseline),
        ("costs", "cost_stress", "Running fee & slippage cost stress matrix", session.run_cost_stress),
        ("regime", "regime_breakdown", "Analyzing calendar years, BTC regimes & outliers", session.run_regime_breakdown),
        ("propfirm", "prop_firm_check", "Simulating two-step prop-firm evaluations", session.run_prop_firm),
        (
            "final_test",
            "final_test",
            "Unlocking 20% held-out TEST slice once",
            lambda: session.run_final_test(json.dumps(session.chosen_params)),
        ),
    ]
    steps = [
        (label, fn)
        for cache_key, tool_name, label, fn in candidate_steps
        if not _is_step_completed(session, cache_key, tool_name)
    ]
    total = len(steps) + 1
    for i, (label, fn) in enumerate(steps, start=1):
        if progress_callback is not None:
            progress_callback(i, total, label)
        fn()

    if progress_callback is not None:
        progress_callback(total, total, "Synthesizing scorecard and final verdict")

    cv = synthesize_verdict_from_session(session)
    session.record_verdict(
        verdict=cv.verdict,
        score=cv.score,
        summary=cv.summary,
        strengths=cv.strengths,
        weaknesses=cv.weaknesses,
        red_flags=cv.red_flags,
        improvements=cv.improvements,
        recommended_risk_per_trade=cv.recommended_risk_per_trade,
        source="offline",
    )
    return session.verdict  # type: ignore[return-value]
