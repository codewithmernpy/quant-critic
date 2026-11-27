"""Page 1: Portfolio Backtest Explorer."""

from __future__ import annotations

import streamlit as st

from quantcritic.report.charts import make_equity_drawdown_figure, make_monthly_heatmap_figure
from quantcritic.ui import cached_backtest, init_page, load_range_data, render_sidebar

init_page("Backtest", icon="📈")
ctx = render_sidebar()

start_str = ctx["start_date"].isoformat()
end_str = ctx["end_date"].isoformat()

st.title("1 · Portfolio Backtest")
st.caption(
    f"Strategy: `{ctx['strategy_name']}` · Symbols: `{', '.join(ctx['symbols'])}` · "
    f"Interval: `{ctx['interval']}` · Range: `{start_str}` to `{end_str}` UTC · "
    "Next-bar open execution with protective stops, taker fees & funding."
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
        label="Backtest",
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

run_clicked = st.button("Run Backtest", type="primary")
autorun = bool(st.session_state.pop("_qc_autorun_backtest", False))
if run_clicked or autorun:
    cached_backtest.clear()

params_items = tuple(sorted(ctx["params"].items()))
res = cached_backtest(
    ctx["strategy_name"],
    tuple(ctx["symbols"]),
    ctx["interval"],
    params_items,
    ctx["config"].model_dump_json(),
    start_date=start_str,
    end_date=end_str,
)
m = res.metrics

c1, c2, c3, c4, c5, c6 = st.columns(6)
cagr_color = "normal" if m["cagr"] > 0 else ("red" if m["cagr"] < 0 else "off")
cagr_arrow = "up" if m["cagr"] > 0 else ("down" if m["cagr"] < 0 else "off")
c1.metric(
    "Total Return",
    f"{m['total_return'] * 100:+.1f}%",
    f"CAGR {m['cagr'] * 100:+.1f}%",
    delta_color=cagr_color,
    delta_arrow=cagr_arrow,
)
c2.metric("Sharpe Ratio", f"{m['sharpe']:.2f}", f"Sortino {m['sortino']:.2f}", delta_color="off", delta_arrow="off")
c3.metric("Max Drawdown", f"{m['max_drawdown'] * 100:.1f}%", f"{m['max_drawdown_duration_days']:.0f}d longest", delta_color="off", delta_arrow="off")
c4.metric("Trades", f"{m['n_trades']}", f"{m['trades_per_month']:.1f}/mo", delta_color="off", delta_arrow="off")
c5.metric("Expectancy", f"{m['expectancy_r']:+.3f}R", f"Win {m['win_rate'] * 100:.1f}% · PF {m['profit_factor']:.2f}", delta_color="off", delta_arrow="off")
fund_color = "normal" if m["total_funding"] > 0 else ("red" if m["total_funding"] < 0 else "off")
fund_arrow = "up" if m["total_funding"] > 0 else ("down" if m["total_funding"] < 0 else "off")
c6.metric(
    "Fees & Funding",
    f"${m['total_fees']:,.0f}",
    f"Fund ${m['total_funding']:+,.0f}",
    delta_color=fund_color,
    delta_arrow=fund_arrow,
)

st.plotly_chart(
    make_equity_drawdown_figure(res.equity_curve, ctx["config"].initial_capital),
    width="stretch",
)
st.plotly_chart(make_monthly_heatmap_figure(m["monthly_returns"]), width="stretch")

st.subheader("Trade Ledger")
if res.trades.empty:
    st.warning("No trades executed for this configuration.")
else:
    display_df = res.trades.copy()
    for col in ("entry_price", "exit_price", "risk_amount", "gross_pnl", "fees", "funding", "net_pnl"):
        display_df[col] = display_df[col].round(2)
    display_df["R"] = display_df["R"].round(3)
    st.dataframe(display_df, width="stretch", hide_index=True)
