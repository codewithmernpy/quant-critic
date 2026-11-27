"""Tests for Plotly chart categorical year axes, price formatting, and shared UI CSS."""

from __future__ import annotations

import pandas as pd
import plotly.io as pio

from quantcritic.engine.metrics import compute_monthly_returns_table
from quantcritic.report.charts import (
    make_monthly_heatmap_figure,
    make_param_heatmap,
    make_regime_figure,
)
from quantcritic.ui import CUSTOM_CSS, format_price


def test_monthly_heatmap_year_axis_is_categorical_and_chronological() -> None:
    idx = pd.date_range("2023-01-01", "2024-06-30", freq="1D", tz="UTC")
    eq = pd.Series(10_000.0 * (1.0005 ** pd.RangeIndex(len(idx))), index=idx)
    monthly = compute_monthly_returns_table(eq, initial_capital=10_000.0)

    fig = make_monthly_heatmap_figure(monthly)
    assert fig.layout.yaxis.type == "category"
    assert list(fig.data[0].y) == ["2023", "2024"]
    assert list(fig.layout.yaxis.categoryarray) == ["2023", "2024"]
    assert fig.layout.yaxis.autorange == "reversed"

    # Unsorted or integer-keyed input still preserves chronological string order
    fig_unsorted = make_monthly_heatmap_figure(
        {2024: {"01": 0.02}, 2023: {"12": -0.01}}  # type: ignore[dict-item]
    )
    assert fig_unsorted.layout.yaxis.type == "category"
    assert list(fig_unsorted.data[0].y) == ["2023", "2024"]

    # HTML report embedding via plotly.io.to_html still works
    html_snippet = pio.to_html(fig, full_html=False, include_plotlyjs=False)
    assert '"type":"category"' in html_snippet.replace(" ", "")
    assert "2,024.5" not in html_snippet

    # Empty monthly table edge case
    fig_empty = make_monthly_heatmap_figure({})
    assert fig_empty.layout.yaxis.type == "category"


def test_regime_figure_and_param_heatmap_axes_are_categorical() -> None:
    regime_res = {
        "regimes": {
            "bull": {"net_pnl": 1200.0},
            "bear": {"net_pnl": -300.0},
            "chop": {"net_pnl": 150.0},
        },
        "yearly": [
            {"year": 2024, "return": 0.14},
            {"year": 2023, "return": 0.28},
        ],
    }
    fig = make_regime_figure(regime_res)
    assert fig.layout.xaxis.type == "category"
    assert fig.layout.xaxis2.type == "category"
    assert list(fig.data[1].x) == ["2023", "2024"]
    assert list(fig.layout.xaxis2.categoryarray) == ["2023", "2024"]

    p_fig = make_param_heatmap({"x_param": "fast", "y_param": "slow", "x_vals": [5, 10], "y_vals": [20, 40], "z_matrix": [[1.0, 1.1], [0.9, 1.2]]})
    assert p_fig.layout.xaxis.type == "category"
    assert p_fig.layout.yaxis.type == "category"


def test_format_price_thousands_and_small_coins() -> None:
    assert format_price(109557.3) == "109,557.30"
    assert format_price(3846.42) == "3,846.42"
    assert format_price(187.13) == "187.13"
    assert format_price(1.2) == "1.20"
    assert format_price(0.5234) == "0.5234"
    assert format_price(0.001234) == "0.001234"
    assert format_price(0.0001234) == "0.0001234"
    assert format_price(0.00001234) == "0.00001234"
    assert format_price(0.0) == "0.00"
    assert format_price(float("nan")) == "N/A"


def test_custom_css_prevents_metric_label_truncation() -> None:
    assert '[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *' in CUSTOM_CSS
    assert "white-space: normal !important;" in CUSTOM_CSS
    assert "overflow: visible !important;" in CUSTOM_CSS
    assert "text-overflow: clip !important;" in CUSTOM_CSS
    assert '[data-testid="stMetricValue"]' in CUSTOM_CSS
    assert "font-size: clamp(1.1rem, 2.2vw, 1.8rem);" in CUSTOM_CSS
