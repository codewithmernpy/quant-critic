"""QuantCritic multipage Streamlit application entry point."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantcritic.strategies.registry import get_strategy, list_strategies
from quantcritic.ui import cached_load_data, format_price, init_page, load_range_data, render_sidebar

init_page("Overview", icon="🛡️")
ctx = render_sidebar()

st.title("QuantCritic")
st.markdown("#### *Most crypto backtests lie. QuantCritic finds out how.*")
st.write(
    "A causal crypto-perpetuals portfolio backtester paired with an 8-part robustness "
    "audit battery and an autonomous Claude AI critic that investigates strategies, "
    "enforces a locked 20% test set, and delivers an evidence-backed verdict."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Strategy", ctx["strategy_name"])
col2.metric("Universe", ", ".join(ctx["symbols"]))
col3.metric("Interval", ctx["interval"])
col4.metric("Cost Assumption", f"{ctx['config'].fee_rate * 1e4:.1f} + {ctx['config'].slippage_bps:.1f} bps/side")
st.caption(
    f"Selected Window: `{ctx['start_date'].isoformat()}` to `{ctx['end_date'].isoformat()}` UTC "
    f"(Available: `{ctx['avail_min'].isoformat()}` to `{ctx['avail_max'].isoformat()}`)"
)

st.divider()

if ctx["date_error"]:
    data = cached_load_data(tuple(ctx["symbols"]), ctx["interval"])
else:
    _, data, _, _ = load_range_data(
        tuple(ctx["symbols"]),
        ctx["interval"],
        start_date=ctx["start_date"].isoformat(),
        end_date=ctx["end_date"].isoformat(),
    )
st.subheader("Loaded Market Sample")
ds_rows = []
for sym, df in data.ohlcv.items():
    f_ser = data.funding.get(sym, pd.Series(dtype=float))
    ds_rows.append(
        {
            "Symbol": sym,
            "Bars": len(df),
            "Start (UTC)": str(df.index[0]) if not df.empty else "N/A",
            "End (UTC)": str(df.index[-1]) if not df.empty else "N/A",
            "Funding Events": len(f_ser),
            "Last Close ($)": format_price(float(df["close"].iloc[-1])) if not df.empty else "N/A",
        }
    )
st.dataframe(pd.DataFrame(ds_rows), width="stretch", hide_index=True)

st.subheader("Registered Strategies")
strat_rows = []
for name in list_strategies(include_leaky=True):
    s = get_strategy(name)
    strat_rows.append(
        {
            "Strategy": s.name,
            "Grid Combos": len(s.expand_param_grid()),
            "Description": s.description,
        }
    )
st.dataframe(pd.DataFrame(strat_rows), width="stretch", hide_index=True)

st.info(
    "Use the left navigation pages to run a **Backtest**, execute the full 8-part **Audit**, "
    "stream the **AI Critic**, compare strategies on the **Leaderboard**, or inspect lookahead "
    "leakage in the **Lookahead Lab**."
)
