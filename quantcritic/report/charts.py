"""Reusable Plotly figure builders shared across HTML reports and the Streamlit app."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = {
    "bg": "#0B1020",
    "panel": "#131A2E",
    "grid": "#232E4D",
    "text": "#E6E9F2",
    "muted": "#94A3B8",
    "primary": "#6366F1",
    "cyan": "#22D3EE",
    "green": "#22C55E",
    "amber": "#F59E0B",
    "red": "#EF4444",
}


def _apply_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": PALETTE["text"]}},
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["bg"],
        font={"color": PALETTE["text"], "family": "Inter, system-ui, sans-serif", "size": 12},
        height=height,
        margin={"l": 50, "r": 25, "t": 48, "b": 42},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    return fig


def make_scorecard_gauge(
    score: float, verdict: str, promising_threshold: float = 70.0
) -> go.Figure:
    """Create a 0-100 robustness gauge colored by verdict."""
    color = (
        PALETTE["green"]
        if verdict == "PROMISING"
        else (PALETTE["amber"] if verdict == "NEEDS WORK" else PALETTE["red"])
    )
    prom_thr = float(promising_threshold)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Overall Score · {verdict}", "font": {"size": 15, "color": PALETTE["text"]}},
            number={"suffix": "/100", "font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": PALETTE["muted"]},
                "bar": {"color": color},
                "bgcolor": PALETTE["bg"],
                "borderwidth": 1,
                "bordercolor": PALETTE["grid"],
                "steps": [
                    {"range": [0, 45], "color": "rgba(239,68,68,0.18)"},
                    {"range": [45, prom_thr], "color": "rgba(245,158,11,0.18)"},
                    {"range": [prom_thr, 100], "color": "rgba(34,197,94,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=PALETTE["panel"],
        font={"color": PALETTE["text"]},
        height=260,
        margin={"l": 20, "r": 20, "t": 45, "b": 15},
    )
    return fig


def make_subscores_bar(sub_scores: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of the 6 scorecard sub-dimensions."""
    labels = [k.replace("_", " ").title() for k in sub_scores.keys()]
    vals = [float(v) for v in sub_scores.values()]
    colors = [
        PALETTE["green"] if v >= 70 else (PALETTE["amber"] if v >= 45 else PALETTE["red"])
        for v in vals
    ]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}" for v in vals],
            textposition="outside",
        )
    )
    _apply_layout(fig, "Robustness Sub-Scores (0–100)", height=260)
    fig.update_xaxes(range=[0, 112])
    fig.update_yaxes(autorange="reversed")
    return fig


def make_equity_drawdown_figure(
    equity_curve: pd.Series,
    initial_capital: float,
    splits_summary: dict[str, str] | None = None,
    title: str = "Portfolio Equity & Underwater Drawdown",
) -> go.Figure:
    """Plot equity curve and drawdown % with optional train/validation/test shaded regions."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
    )
    if not equity_curve.empty:
        eq = equity_curve.sort_index()
        vals = eq.to_numpy(dtype=np.float64)
        peaks = np.maximum.accumulate(np.concatenate([[initial_capital], vals]))[1:]
        peaks = np.maximum(peaks, initial_capital)
        dd_pct = ((vals / peaks) - 1.0) * 100.0

        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=vals,
                name="Equity ($)",
                line={"color": PALETTE["primary"], "width": 2.0},
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.12)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=dd_pct,
                name="Drawdown (%)",
                line={"color": PALETTE["red"], "width": 1.5},
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.20)",
            ),
            row=2,
            col=1,
        )

        if splits_summary:
            t0 = eq.index[0]
            tr_end = pd.Timestamp(splits_summary["train_end"])
            val_end = pd.Timestamp(splits_summary["val_end"])
            te_start = pd.Timestamp(splits_summary["test_start"])
            te_end = eq.index[-1]

            regions = [
                (t0, tr_end, "rgba(99,102,241,0.06)", "Train (60%)"),
                (tr_end, val_end, "rgba(34,211,238,0.08)", "Validation (20%)"),
                (te_start, te_end, "rgba(245,158,11,0.10)", "Locked Test (20%)"),
            ]
            for x0, x1, fill_c, label in regions:
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=fill_c,
                    line_width=0,
                    annotation_text=label,
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color=PALETTE["muted"],
                    row="all",
                    col=1,
                )

    _apply_layout(fig, title, height=460)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig


def make_monthly_heatmap_figure(monthly_table: dict[str, dict[str, float]]) -> go.Figure:
    """Plot calendar-year x month compounded return heatmap (%)."""
    months = [f"{m:02d}" for m in range(1, 13)]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    raw_years = sorted(
        monthly_table.keys(),
        key=lambda y: (0, int(y)) if str(y).isdigit() else (1, str(y)),
    )
    years = [str(yr) for yr in raw_years]
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for raw_yr in raw_years:
        row_z: list[float | None] = []
        row_t: list[str] = []
        for mo in months:
            val = monthly_table[raw_yr].get(mo)
            if val is None:
                row_z.append(None)
                row_t.append("")
            else:
                pct = val * 100.0
                row_z.append(round(pct, 2))
                row_t.append(f"{pct:+.1f}%")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=month_names,
            y=years,
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#B91C1C"],
                [0.5, "#1E293B"],
                [1.0, "#15803D"],
            ],
            zmid=0.0,
            colorbar={"title": "Ret %"},
        )
    )
    _apply_layout(fig, "Monthly Compounded Returns (%)", height=320)
    fig.update_xaxes(type="category")
    yaxis_kwargs: dict[str, Any] = {"type": "category", "autorange": "reversed"}
    if years:
        yaxis_kwargs["categoryorder"] = "array"
        yaxis_kwargs["categoryarray"] = years
    fig.update_yaxes(**yaxis_kwargs)
    return fig


def make_param_heatmap(heatmap_data: dict[str, Any]) -> go.Figure:
    """Plot 2D grid-search Sharpe ratio surface."""
    z = heatmap_data.get("z_matrix", [])
    x_vals = [f"{heatmap_data.get('x_param', '')}={v}" for v in heatmap_data.get("x_vals", [])]
    y_vals = [f"{heatmap_data.get('y_param', '')}={v}" for v in heatmap_data.get("y_vals", [])]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_vals,
            y=y_vals,
            colorscale="Viridis",
            colorbar={"title": "Sharpe"},
        )
    )
    _apply_layout(fig, "Parameter Grid Sharpe Surface (Train)", height=320)
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def make_montecarlo_figure(bootstrap: dict[str, Any]) -> go.Figure:
    """Plot 1000x Monte Carlo trade bootstrap return & max-drawdown distributions."""
    rets = [x * 100.0 for x in bootstrap.get("return_samples", [])]
    dds = [x * 100.0 for x in bootstrap.get("max_dd_samples", [])]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Bootstrap Total Return (%)", "Bootstrap Max Drawdown (%)"))
    if rets:
        fig.add_trace(
            go.Histogram(x=rets, nbinsx=35, marker_color=PALETTE["primary"], name="Return %"),
            row=1,
            col=1,
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color=PALETTE["red"], row=1, col=1)
    if dds:
        fig.add_trace(
            go.Histogram(x=dds, nbinsx=35, marker_color=PALETTE["amber"], name="Max DD %"),
            row=1,
            col=2,
        )
        p95 = float(bootstrap.get("p95_max_drawdown", 0.0)) * 100.0
        fig.add_vline(
            x=p95,
            line_dash="dash",
            line_color=PALETTE["red"],
            annotation_text=f"p95: {p95:.1f}%",
            row=1,
            col=2,
        )
    _apply_layout(fig, "Monte Carlo Trade Bootstrap (1,000 Paths)", height=340)
    return fig


def make_random_baseline_figure(rb: dict[str, Any]) -> go.Figure:
    """Plot 200x random-entry expectancy histogram with the real strategy marked."""
    exps = rb.get("baseline_expectancies", [])
    real_exp = float(rb.get("real_expectancy_r", 0.0))
    p_val = float(rb.get("p_value", 1.0))
    fig = go.Figure()
    if exps:
        fig.add_trace(
            go.Histogram(
                x=exps,
                nbinsx=30,
                marker_color=PALETTE["muted"],
                name="Random Entry Expectancy (R)",
            )
        )
        fig.add_vline(
            x=real_exp,
            line_width=3,
            line_color=PALETTE["cyan"],
            annotation_text=f"Real: {real_exp:+.3f}R (p={p_val:.3f})",
            annotation_position="top right",
        )
    _apply_layout(fig, "Random-Entry Permutation Baseline (200 Portfolios)", height=340)
    fig.update_xaxes(title_text="Mean Expectancy (R)")
    return fig


def make_cost_stress_figure(cost_res: dict[str, Any]) -> go.Figure:
    """Plot fee multiplier x slippage bps expectancy heatmap."""
    z = cost_res.get("expectancy_matrix", [])
    x_labels = [f"{s:.0f} bps" for s in cost_res.get("slippage_bps_grid", [])]
    y_labels = [f"{m:.0f}x Fee" for m in cost_res.get("fee_multipliers", [])]
    text = [[f"{val:+.3f}R" for val in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            text=text,
            texttemplate="%{text}",
            colorscale=[[0.0, "#B91C1C"], [0.5, "#1E293B"], [1.0, "#15803D"]],
            zmid=0.0,
            colorbar={"title": "Exp (R)"},
        )
    )
    _apply_layout(
        fig,
        f"Cost Stress Matrix (Break-Even: {cost_res.get('break_even_cost_bps', 0.0):.1f} bps/side)",
        height=320,
    )
    return fig


def make_regime_figure(regime_res: dict[str, Any]) -> go.Figure:
    """Plot net PnL across BTC bull/bear/chop regimes and calendar years."""
    regs = regime_res.get("regimes", {})
    yearly = regime_res.get("yearly", [])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Net PnL by BTC Regime ($)", "Annual Return (%)"))

    r_names = ["bull", "bear", "chop"]
    r_pnls = [float(regs.get(r, {}).get("net_pnl", 0.0)) for r in r_names]
    r_colors = [PALETTE["green"] if v >= 0 else PALETTE["red"] for v in r_pnls]
    fig.add_trace(go.Bar(x=[r.title() for r in r_names], y=r_pnls, marker_color=r_colors, name="Regime PnL"), row=1, col=1)

    yrs: list[str] = []
    if yearly:
        ordered_yearly = sorted(
            yearly,
            key=lambda y: (0, int(y["year"])) if str(y["year"]).isdigit() else (1, str(y["year"])),
        )
        yrs = [str(y["year"]) for y in ordered_yearly]
        rets = [float(y["return"]) * 100.0 for y in ordered_yearly]
        y_colors = [PALETTE["primary"] if v >= 0 else PALETTE["red"] for v in rets]
        fig.add_trace(go.Bar(x=yrs, y=rets, marker_color=y_colors, name="Yearly Return %"), row=1, col=2)

    _apply_layout(fig, "Market Regime & Calendar Year Consistency", height=340)
    fig.update_xaxes(type="category", row=1, col=1)
    if yrs:
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=yrs, row=1, col=2)
    else:
        fig.update_xaxes(type="category", row=1, col=2)
    return fig


def make_walkforward_figure(wf_res: dict[str, Any]) -> go.Figure:
    """Plot rolling walk-forward stitched OOS equity curve."""
    eq = wf_res.get("rolling_oos_equity", pd.Series(dtype=float))
    fig = go.Figure()
    if isinstance(eq, pd.Series) and not eq.empty:
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq.to_numpy(dtype=np.float64),
                name="Rolling 12m->3m OOS Equity",
                line={"color": PALETTE["cyan"], "width": 2.0},
            )
        )
    _apply_layout(fig, "Rolling Walk-Forward Stitched OOS Equity (Train+Val)", height=320)
    fig.update_yaxes(title_text="Equity ($)")
    return fig
