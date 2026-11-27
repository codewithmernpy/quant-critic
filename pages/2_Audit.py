"""Page 2: 8-Part Robustness Audit Battery & Report Export."""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from quantcritic.agent.offline import run_offline_critic, synthesize_verdict_from_session
from quantcritic.agent.tools import AuditSession, to_jsonable
from quantcritic.config import AuditConfig
from quantcritic.report.charts import (
    make_cost_stress_figure,
    make_equity_drawdown_figure,
    make_montecarlo_figure,
    make_monthly_heatmap_figure,
    make_param_heatmap,
    make_random_baseline_figure,
    make_regime_figure,
    make_scorecard_gauge,
    make_subscores_bar,
    make_walkforward_figure,
)
from quantcritic.report.html import build_report_payload, render_audit_html_from_payload
from quantcritic.strategies.registry import get_strategy
from quantcritic.ui import badge_html, init_page, load_range_data, render_sidebar

init_page("Audit Battery", icon="🧪")
ctx = render_sidebar()

start_str = ctx["start_date"].isoformat()
end_str = ctx["end_date"].isoformat()

st.title("2 · Robustness Audit Battery")
st.caption(
    f"Range: `{start_str}` to `{end_str}` UTC · Verdict mode: `{ctx['strictness']}` · "
    "Runs all 8 integrity, out-of-sample, overfitting, Monte Carlo, random baseline, "
    "cost stress, regime, and prop-firm checks with a locked 20% test slice."
)

if ctx["date_error"]:
    st.error(ctx["date_error"])
    st.stop()

try:
    _, window_data, warmup_data, _ = load_range_data(
        tuple(ctx["symbols"]),
        ctx["interval"],
        start_date=start_str,
        end_date=end_str,
        min_days=180,
        min_bars=30,
        label="Audit",
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

run_btn = st.button("Run Full Audit", type="primary")
autorun_audit = bool(st.session_state.pop("_qc_autorun_audit", False))
audit_base_key = (
    ctx["strategy_name"],
    tuple(ctx["symbols"]),
    ctx["interval"],
    ctx["config"].model_dump_json(),
    start_str,
    end_str,
    ctx["data_version"],
)

if run_btn or autorun_audit:
    progress_bar = st.progress(0, text="Initializing audit session...")
    strat = get_strategy(ctx["strategy_name"])
    session = AuditSession(
        strategy=strat,
        data=window_data,
        config=ctx["config"],
        audit_config=AuditConfig(strictness=ctx["strictness"]),
        warmup_data=warmup_data,
    )
    try:
        progress_bar.progress(5, text="Checking train+validation split and trade count...")
        session.check_min_train_val_trades(min_trades=5)
    except ValueError as exc:
        progress_bar.empty()
        st.error(str(exc))
        st.stop()

    def _on_prog(step: int, total: int, label: str) -> None:
        progress_bar.progress(int(step / total * 100), text=f"Step {step}/{total}: {label}")

    run_offline_critic(session, progress_callback=_on_prog)
    payload = build_report_payload(session)
    full_res = session.cached_results["full_result"]
    wf_eq = session.cached_results["walkforward"].get("rolling_oos_equity", pd.Series(dtype=float))
    mc = session.cached_results["montecarlo"]
    payload["_bootstrap_samples"] = mc["bootstrap"]
    payload["_random_baseline_samples"] = mc["random_baseline"]
    st.session_state["last_audit"] = {
        "key": audit_base_key,
        "strictness": ctx["strictness"],
        "session": session,
        "payload": payload,
        "equity": full_res.equity_curve,
        "wf_eq": wf_eq,
    }
    progress_bar.progress(100, text="Audit complete.")

stored = st.session_state.get("last_audit")
if not stored or stored.get("key") != audit_base_key:
    st.info("Configure strategy, universe, and costs in the sidebar, then click **Run Full Audit** to run all 8 robustness modules and unlock the held-out 20% test slice once.")
    st.stop()

if stored.get("strictness") != ctx["strictness"] and "session" in stored:
    session = stored["session"]
    session.strictness = ctx["strictness"]
    session.audit_config = session.audit_config.model_copy(update={"strictness": ctx["strictness"]})
    session.ensure_complete_audit()
    cv = synthesize_verdict_from_session(session)
    session.verdict = cv
    payload = build_report_payload(session)
    mc = session.cached_results["montecarlo"]
    payload["_bootstrap_samples"] = mc["bootstrap"]
    payload["_random_baseline_samples"] = mc["random_baseline"]
    stored["strictness"] = ctx["strictness"]
    stored["payload"] = payload

payload = stored["payload"]
equity_curve = stored["equity"]
wf_eq = stored["wf_eq"]

verdict = payload["verdict"]
sc = payload["scorecard"]
la = payload["lookahead"]
wf = payload["walkforward"]
ov = payload["overfit"]
mc = payload["montecarlo"]
cs = payload["costs"]
reg = payload["regime"]
pf = payload["propfirm"]
ft = payload["final_test"]

active_mode = payload["meta"].get("strictness", sc.get("strictness", ctx["strictness"]))
st.markdown(
    f"### Verdict: {badge_html(verdict['verdict'])} &nbsp; Score: **{verdict['score']:.1f}/100** "
    f"&nbsp; (`Verdict mode: {active_mode}`)",
    unsafe_allow_html=True,
)
splits_meta = payload["meta"]["splits"]
dr_meta = payload["meta"]["date_range"]
st.caption(
    f"Verdict mode: `{active_mode}` · Effective Range: `{dr_meta['start']}` to `{dr_meta['end']}` UTC · "
    f"Train (60%): `{splits_meta['train_start'][:10]}..{splits_meta['train_end'][:10]}` · "
    f"Validation (20%): `{splits_meta['val_start'][:10]}..{splits_meta['val_end'][:10]}` · "
    f"Locked Test (20%): `{splits_meta['test_start'][:10]}..{splits_meta['test_end'][:10]}`"
)
st.info(verdict["summary"])
st.caption(
    f"Weighted sub-score total {sc['weighted_score']:.1f}/100 -> {sc['overall_score']:.1f}/100 after hard gates ({active_mode} mode; PROMISING >= {sc.get('promising_threshold', 70.0):.0f})."
)
if sc.get("gates"):
    st.warning(
        "Hard gates applied: "
        + " | ".join(f"{g['gate']} (score cap {g['cap']:.1f}, at most {g['ceiling']})" for g in sc["gates"])
    )

g1, g2 = st.columns(2)
g1.plotly_chart(
    make_scorecard_gauge(
        verdict["score"],
        verdict["verdict"],
        promising_threshold=float(sc.get("promising_threshold", 70.0)),
    ),
    width="stretch",
)
g2.plotly_chart(make_subscores_bar(sc["sub_scores"]), width="stretch")

st.plotly_chart(
    make_equity_drawdown_figure(equity_curve, ctx["config"].initial_capital, wf.get("splits_summary")),
    width="stretch",
)

# Download buttons
dl1, dl2 = st.columns(2)
clean_payload = {k: v for k, v in payload.items() if not k.startswith("_")}
html_str = render_audit_html_from_payload(
    clean_payload,
    equity_curve=equity_curve,
    wf_equity=wf_eq,
    bootstrap_data=payload.get("_bootstrap_samples"),
    random_baseline_data=payload.get("_random_baseline_samples"),
)
safe_strat_fn = ctx["strategy_name"].replace(":", "_")
dl1.download_button(
    "Download HTML Report",
    data=html_str.encode("utf-8"),
    file_name=f"{safe_strat_fn}_audit.html",
    mime="text/html",
)
dl2.download_button(
    "Download JSON Report",
    data=json.dumps(to_jsonable(clean_payload), indent=2, allow_nan=False).encode("utf-8"),
    file_name=f"{safe_strat_fn}_audit.json",
    mime="application/json",
)

st.subheader("Audit Modules Breakdown")
modules = [
    ("1. Integrity & Lookahead", la["status"], la["explanation"]),
    ("2. Walk-Forward & OOS", wf["status"], wf["explanation"]),
    ("3. Overfitting (DSR / PBO / Stability)", ov["status"], ov["explanation"]),
    ("4. Monte Carlo & Random Baseline", mc["status"], mc["explanation"]),
    ("5. Transaction Cost Stress", cs["status"], cs["explanation"]),
    ("6. Regimes & Outlier Survival", reg["status"], reg["explanation"]),
    ("7. Two-Step Prop-Firm Check", pf["status"], pf["explanation"]),
]
for title, status, desc in modules:
    st.markdown(f"{badge_html(status.upper())} &nbsp; **{title}** — {desc}", unsafe_allow_html=True)

tabs = st.tabs(["Walk-Forward", "Overfitting", "Monte Carlo & Luck", "Cost Stress", "Regimes", "Prop-Firm"])
with tabs[0]:
    wf_rows = [
        {"Split": "Train (60%)", "Window (UTC)": f"{splits_meta['train_start'][:10]} -> {splits_meta['train_end'][:10]}", **wf["train_metrics"]},
        {"Split": "Validation (20%)", "Window (UTC)": f"{splits_meta['val_start'][:10]} -> {splits_meta['val_end'][:10]}", **wf["validation_metrics"]},
        {"Split": "Rolling WF OOS", "Window (UTC)": f"{splits_meta['train_start'][:10]} -> {splits_meta['val_end'][:10]}", **wf["rolling_oos_metrics"]},
        {"Split": "Locked Test (20%)", "Window (UTC)": f"{splits_meta['test_start'][:10]} -> {splits_meta['test_end'][:10]}", **ft["test_metrics"]},
    ]
    st.dataframe(
        pd.DataFrame(wf_rows)[["Split", "Window (UTC)", "n_trades", "total_return", "sharpe", "max_drawdown", "expectancy_r"]],
        width="stretch",
        hide_index=True,
    )
    st.plotly_chart(make_walkforward_figure({"rolling_oos_equity": wf_eq}), width="stretch")
with tabs[1]:
    st.json({"dsr": ov["dsr"], "pbo": ov["pbo"], "stability": ov["stability"]})
    st.plotly_chart(make_param_heatmap(ov["heatmap"]), width="stretch")
with tabs[2]:
    bs_full = payload.get("_bootstrap_samples", mc["bootstrap"])
    rb_full = payload.get("_random_baseline_samples", mc["random_baseline"])
    st.plotly_chart(make_montecarlo_figure(bs_full), width="stretch")
    st.plotly_chart(make_random_baseline_figure(rb_full), width="stretch")
with tabs[3]:
    st.plotly_chart(make_cost_stress_figure(cs), width="stretch")
with tabs[4]:
    st.plotly_chart(make_regime_figure(reg), width="stretch")
    st.plotly_chart(make_monthly_heatmap_figure(payload["full_metrics"]["monthly_returns"]), width="stretch")
with tabs[5]:
    st.json(pf)
