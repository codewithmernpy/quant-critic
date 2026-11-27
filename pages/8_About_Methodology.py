"""Page 8: About & Methodology."""

from __future__ import annotations

import streamlit as st

from quantcritic.ui import init_page, render_sidebar

init_page("About & Methodology", icon="📚")
ctx = render_sidebar()

st.title("8 · About & Methodology")
st.markdown(
    """
### Why Most Crypto Backtests Lie
1. **Lookahead Leakage** — centered rolling windows (`center=True`), full-sample mean/std normalization, or filling orders at the same bar close that generated the signal.
2. **Multiple-Testing Overfitting** — tuning dozens of indicator combinations on the full sample without adjusting for trial count or out-of-sample rank decay.
3. **Random-Timing Illusions** — positive drift in bull markets masquerading as entry timing skill.
4. **Cost & Perpetual Funding Neglect** — ignoring taker fees, adverse slippage, gap-through-stop fills, and 8h/4h/2h perpetual funding settlements.
5. **Regime & Outlier Fragility** — a single meme/bull rally driving >100% of total net profit.

### Mathematical Audits Inside QuantCritic
- **Causal Truncation & 1-Bar Delay (`audit/lookahead.py`)**: Evaluates `strategy.generate(df.iloc[:t+1])` at 25 random cut points `t` and verifies equality of every order field (`signal`, `stop_dist`, `tp_r`, `max_bars`, `exit_signal`) over the trailing 100 bars `<= t`, paired with an extra 1-bar entry delay stress test.
- **Chronological Split & Rolling Walk-Forward (`audit/walkforward.py`)**: 60% Train / 20% Validation / 20% **Locked Test** plus 12m->3m rolling walk-forward strictly inside Train+Validation. Validation, every walk-forward OOS window and the locked test use signals warmed up on all prior history; only trades entered inside the window count.
- **Deflated Sharpe Ratio & CSCV PBO (`audit/overfit.py`)**: Bailey & Lopez de Prado (2014) non-normal DSR adjusted for grid size, skewness, and kurtosis; Bailey et al. (2015) Combinatorially Symmetric Cross-Validation (`S = 10` blocks, 252 splits) for Probability of Backtest Overfitting; plus grid neighbor plateau stability.
- **Bootstrap & Matched Random-Entry Baseline (`audit/montecarlo.py`)**: 1,000x vectorized trade-R bootstrap and 200x random-entry portfolios matched by per-symbol trade count, holding-time distribution, long/short ratio, ATR stop rule, TP rule, fees, slippage and funding.
- **Cost Stress (`audit/costs.py`)**: 4x4 fee multiplier x slippage bps grid and a linear-fit break-even one-way cost (flagged when extrapolated beyond the grid).
- **Regimes & Prop-Firm Challenge (`audit/regime.py`, `audit/propfirm.py`)**: Calendar years, causal BTC bull/bear/chop classification (200d SMA, 30d vol vs expanding 80th percentile, lagged one day), top-5% outlier trade removal, and rolling two-step (+8%/+5%) prop challenge simulation.

### Scorecard & Verdict Strictness Modes
Seven weighted sub-scores (integrity, pre-test OOS edge, locked test, robustness, cost resilience, consistency, risk), followed by hard gates. In **all** modes, truncation/lookahead failure triggers **REJECT** (score <= 20), a losing locked test is at most **NEEDS WORK**, and real numbers are always reported:
- **`balanced` (default)**: test & pre-test OOS lost -> REJECT (<= 40); test lost -> at most NEEDS WORK (<= 55); pre-test OOS lost with profitable test -> at most NEEDS WORK only if test trades < 20; PROMISING >= 65 with test trades >= 10, base expectancy > 0, random-entry p <= 0.20, DSR >= 0.30.
- **`strict`**: original gates — any OOS loss caps at NEEDS WORK (<= 60); PROMISING >= 70 with test trades >= 10, base expectancy > 0, random-entry p <= 0.10, DSR >= 0.50.
- **`relaxed`**: exploratory — lookahead FAIL -> REJECT (<= 20), test lost -> at most NEEDS WORK (<= 55); PROMISING >= 60 with test trades >= 5 and base expectancy > 0.

### No-Code Builder & Python Plugins
- **Strategy Builder (`pages/7_Strategy_Builder.py`)**: Build causal closed-bar rule specs visually (or draft from plain English via Claude), preview signals, and save JSON specs to `user_strategies/` (`custom:<name>`).
- **Python Plugins (`quantcritic/strategies/custom/`)**: Subclass `Strategy` in a new `.py` file (copy `template_strategy.py`); every custom and plugin strategy goes through the exact same causal truncation test and audit suite.

### Crypto Quant Suite
Planned companion projects (not yet published): PropPilot, SqueezeRadar, RiskGuard.
"""
)
