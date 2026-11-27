"""Page 4: Strategy Leaderboard (all built-in strategies audited & cached)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantcritic.ui import cached_leaderboard, init_page, load_range_data, render_sidebar

init_page("Leaderboard", icon="🏆")
ctx = render_sidebar()

start_str = ctx["start_date"].isoformat()
end_str = ctx["end_date"].isoformat()

st.title("4 · Strategy Leaderboard")
st.caption(
    f"Range: `{start_str}` to `{end_str}` UTC · Verdict mode: `{ctx['strictness']}` · "
    "Audits every registered strategy with the offline robustness battery and ranks them by score."
)

if ctx["date_error"]:
    st.error(ctx["date_error"])
    st.stop()

try:
    load_range_data(
        tuple(ctx["symbols"]),
        ctx["interval"],
        start_date=start_str,
        end_date=end_str,
        min_days=180,
        min_bars=30,
        label="Leaderboard",
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

col_a, col_b = st.columns([1, 2])
include_leaky = col_a.checkbox("Include leaky_demo (cheat benchmark)", value=True)
run_lb = col_b.button("Run Leaderboard", type="primary")

lb_key = (
    tuple(ctx["symbols"]),
    ctx["interval"],
    ctx["config"].model_dump_json(),
    include_leaky,
    start_str,
    end_str,
    ctx["strictness"],
    ctx["data_version"],
)
if run_lb:
    with st.spinner("Auditing all strategies (cached by parameter signature)..."):
        rows = cached_leaderboard(
            tuple(ctx["symbols"]),
            ctx["interval"],
            ctx["config"].model_dump_json(),
            include_leaky=include_leaky,
            start_date=start_str,
            end_date=end_str,
            strictness=ctx["strictness"],
        )
    st.session_state["last_leaderboard"] = {"key": lb_key, "rows": rows}

stored_lb = st.session_state.get("last_leaderboard")
if not stored_lb or stored_lb.get("key") != lb_key:
    st.info("Click **Run Leaderboard** to audit all built-in strategies and compare their robustness scorecards.")
    st.stop()

rows = stored_lb["rows"]

df = pd.DataFrame(rows)
colors = [
    "#22C55E" if v == "PROMISING" else ("#F59E0B" if v == "NEEDS WORK" else "#EF4444")
    for v in df["verdict"]
]
fig = go.Figure(
    go.Bar(
        x=df["strategy"],
        y=df["score"],
        marker_color=colors,
        text=[f"{s:.1f} ({v})" for s, v in zip(df["score"], df["verdict"])],
        textposition="outside",
    )
)
fig.update_layout(
    title="Strategy Robustness Score Comparison",
    paper_bgcolor="#131A2E",
    plot_bgcolor="#0B1020",
    font={"color": "#E6E9F2"},
    yaxis={"range": [0, 112], "title": "Score (0–100)"},
    height=380,
)
st.plotly_chart(fig, width="stretch")

st.subheader("Full Comparison Matrix")
st.dataframe(df.drop(columns=["summary"]), width="stretch", hide_index=True)
