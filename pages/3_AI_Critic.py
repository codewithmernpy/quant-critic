"""Page 3: Live AI Strategy Critic (Claude tool_runner with offline deterministic fallback)."""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from quantcritic.agent.critic import run_ai_critic
from quantcritic.agent.tools import AuditSession
from quantcritic.config import AuditConfig
from quantcritic.strategies.registry import get_strategy
from quantcritic.ui import badge_html, init_page, load_range_data, render_sidebar

init_page("AI Critic", icon="🤖")
ctx = render_sidebar()

start_str = ctx["start_date"].isoformat()
end_str = ctx["end_date"].isoformat()

st.title("3 · AI Strategy Critic")
st.caption(
    f"Strategy: `{ctx['strategy_name']}` · Symbols: `{', '.join(ctx['symbols'])}` · "
    f"Interval: `{ctx['interval']}` · Range: `{start_str}` to `{end_str}` UTC · "
    f"Verdict mode: `{ctx['strictness']}`"
)
has_key = bool(ctx["api_key"] or os.getenv("ANTHROPIC_API_KEY", "").strip())
if has_key:
    st.success(f"Anthropic key detected in session/environment — AI mode ready (model `{ctx['model']}`).")
else:
    st.info("No Anthropic API key set — running deterministic offline critic (works 100% locally).")

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
        label="AI Critic",
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

run_ai_btn = st.button("Run Critic Investigation", type="primary")
if run_ai_btn:
    strat = get_strategy(ctx["strategy_name"])
    session = AuditSession(
        strategy=strat,
        data=window_data,
        config=ctx["config"],
        audit_config=AuditConfig(strictness=ctx["strictness"]),
        warmup_data=warmup_data,
    )
    try:
        session.check_min_train_val_trades(min_trades=5)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Live Investigation Stream")
    stream_box = st.empty()
    events: list[str] = []

    def _on_event(kind: str, msg: str) -> None:
        events.append(f"**[{kind.upper()}]** {msg}")
        stream_box.markdown("\n\n".join(events[-15:]))

    verdict = run_ai_critic(
        session=session,
        use_ai=has_key,
        api_key=ctx["api_key"] or None,
        model=ctx["model"],
        on_event=_on_event,
    )
    st.session_state["last_ai_critic"] = {
        "verdict": verdict.model_dump(),
        "tool_log": session.tool_log,
        "thoughts": session.agent_thoughts,
        "scorecard": session.cached_results.get("scorecard", {}),
        "range": f"{str(session.splits.train_start)[:10]} to {str(session.splits.test_end)[:10]}",
    }

saved = st.session_state.get("last_ai_critic")
if saved:
    v = saved["verdict"]
    st.divider()
    st.markdown(
        f"### Critic Verdict: {badge_html(v['verdict'])} &nbsp; "
        f"Score: **{v['score']:.1f}/100** &nbsp; (Source: `{v['source'].upper()}`)",
        unsafe_allow_html=True,
    )
    st.write(v["summary"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Strengths**")
        for item in v["strengths"]:
            st.markdown(f"- {item}")
        st.markdown("**Weaknesses**")
        for item in v["weaknesses"]:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Red Flags**")
        for item in (v["red_flags"] or ["None reported."]):
            st.markdown(f"- {item}")
        st.markdown("**Recommended Improvements**")
        for item in v["improvements"]:
            st.markdown(f"- {item}")
        st.metric("Recommended Risk / Trade", f"{v['recommended_risk_per_trade'] * 100:.2f}%")

    if saved["thoughts"]:
        with st.expander("Claude Assistant Reasoning Log", expanded=False):
            for th in saved["thoughts"]:
                st.markdown(th)

    st.subheader("Tool Execution Log")
    st.dataframe(pd.DataFrame(saved["tool_log"]), width="stretch", hide_index=True)
else:
    st.caption("Click **Run Critic Investigation** to launch the tool-calling strategy critic.")
