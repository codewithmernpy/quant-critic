"""Page 5: Lookahead Lab — side-by-side leaky_demo vs an honest strategy."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantcritic.report.charts import make_equity_drawdown_figure
from quantcritic.strategies.registry import list_strategies
from quantcritic.ui import badge_html, cached_lookahead_pair, init_page, load_range_data, render_sidebar

init_page("Lookahead Lab", icon="🔬")
ctx = render_sidebar()

start_str = ctx["start_date"].isoformat()
end_str = ctx["end_date"].isoformat()

st.title("5 · Lookahead Lab")
st.caption(
    f"Range: `{start_str}` to `{end_str}` UTC · "
    "Side-by-side proof of how a realistic pandas lookahead bug (`center=True` + full-sample z-score) "
    "fools raw backtests yet fails causal truncation immediately."
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
        min_days=30,
        min_bars=30,
        label="Lookahead Lab",
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

honest_options = list_strategies(include_leaky=False)
honest_choice = st.selectbox("Honest Comparison Strategy", options=honest_options, index=honest_options.index("donchian_breakout"))
run_lab = st.button("Run Lookahead Comparison", type="primary")

lab_key = (
    honest_choice,
    tuple(ctx["symbols"]),
    ctx["interval"],
    ctx["config"].model_dump_json(),
    start_str,
    end_str,
    ctx["data_version"],
)
if run_lab:
    with st.spinner("Running truncation causality & backtest comparison..."):
        res_pair = cached_lookahead_pair(
            honest_choice,
            tuple(ctx["symbols"]),
            ctx["interval"],
            ctx["config"].model_dump_json(),
            start_date=start_str,
            end_date=end_str,
        )
    st.session_state["last_lookahead_lab"] = {"key": lab_key, "data": res_pair}

stored_lab = st.session_state.get("last_lookahead_lab")
if not stored_lab or stored_lab.get("key") != lab_key:
    st.info("Click **Run Lookahead Comparison** to run the side-by-side causal truncation test (`leaky_demo` vs honest strategy).")
    st.stop()

res_pair = stored_lab["data"]

left, right = st.columns(2)
for col, name, label in (
    (left, "leaky_demo", "leaky_demo (Deliberate Lookahead Bug)"),
    (right, honest_choice, f"{honest_choice} (Strictly Causal)"),
):
    info = res_pair[name]
    m = info["metrics"]
    la = info["lookahead"]
    tr = la["truncation"]
    with col:
        st.subheader(label)
        st.markdown(
            f"Integrity Status: {badge_html(la['status'].upper())} &nbsp; "
            f"(Cuts checked: **{tr['cuts_checked']}**, Mismatches: **{tr['total_mismatches']}**)",
            unsafe_allow_html=True,
        )
        k1, k2, k3 = st.columns(3)
        tr_pct = m["total_return"] * 100
        k1.metric("Raw Total Return", f"{tr_pct:+.1f}%" if abs(tr_pct) < 1e4 else f"{tr_pct:+.2e}%")
        k2.metric("Raw Sharpe", f"{m['sharpe']:.2f}")
        k3.metric("Raw Max DD", f"{m['max_drawdown'] * 100:.1f}%")

        st.plotly_chart(
            make_equity_drawdown_figure(
                info["equity_curve"],
                ctx["config"].initial_capital,
                title=f"{name} Equity Curve",
            ),
            width="stretch",
        )
        st.write(la["explanation"])
        if tr["examples"]:
            st.error("Exact Mismatching Bars Caught by Truncation Test:")
            st.dataframe(pd.DataFrame(tr["examples"]), width="stretch", hide_index=True)
        else:
            st.success("Zero signal/stop mismatches across all causal truncation cuts.")
