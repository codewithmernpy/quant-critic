"""Single-file HTML (Plotly.js loaded from CDN) & JSON report generation for audits and leaderboard."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2
import plotly.io as pio

from quantcritic.agent.tools import AuditSession, CriticVerdict, to_jsonable
from quantcritic.config import REPORTS_DIR
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

# Autoescape all interpolated text (critic output may contain "<", "&", or markup);
# pre-rendered Plotly chart HTML is explicitly marked `| safe` in the templates.
_JINJA_ENV = jinja2.Environment(autoescape=True)

AUDIT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>QuantCritic Audit — {{ meta.strategy }}</title>
<style>
  :root {
    --bg: #0B1020; --panel: #131A2E; --border: #232E4D; --text: #E6E9F2; --muted: #94A3B8;
    --primary: #6366F1; --green: #22C55E; --amber: #F59E0B; --red: #EF4444; --cyan: #22D3EE;
  }
  [data-theme="light"] {
    --bg: #F8FAFC; --panel: #FFFFFF; --border: #E2E8F0; --text: #0F172A; --muted: #475569;
    --primary: #4F46E5; --green: #16A34A; --amber: #D97706; --red: #DC2626; --cyan: #0891B2;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.55;
  }
  .container { max-width: 1320px; margin: 0 auto; padding: 28px 24px 64px; }
  header {
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 16px; border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 24px;
  }
  .brand { font-size: 13px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--cyan); font-weight: 700; }
  h1 { margin: 4px 0 6px; font-size: 28px; }
  .subtitle { color: var(--muted); font-size: 14px; }
  .badge {
    display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700;
    font-size: 14px; letter-spacing: 0.05em;
  }
  .badge-PROMISING, .badge-pass { background: rgba(34,197,94,0.18); color: var(--green); border: 1px solid var(--green); }
  .badge-NEEDS-WORK, .badge-warn { background: rgba(245,158,11,0.18); color: var(--amber); border: 1px solid var(--amber); }
  .badge-REJECT, .badge-fail { background: rgba(239,68,68,0.18); color: var(--red); border: 1px solid var(--red); }
  .theme-btn {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600;
  }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; margin-bottom: 22px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 14px; margin-bottom: 24px; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-bottom: 20px;
  }
  .kpi-card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .kpi-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .kpi-value { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .kpi-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
  h2 { font-size: 18px; margin: 0 0 12px; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  h3 { font-size: 15px; margin: 14px 0 8px; color: var(--cyan); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }
  ul { margin: 6px 0 0 18px; padding: 0; }
  li { margin-bottom: 6px; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
  .callout { border-left: 3px solid var(--primary); padding: 10px 14px; background: rgba(99,102,241,0.08); border-radius: 6px; margin-bottom: 12px; }
  footer { margin-top: 40px; text-align: center; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <div class="brand">QuantCritic · Strategy Audit Report</div>
      <h1>{{ meta.strategy }}</h1>
      <div class="subtitle">
        Symbols: {{ meta.symbols | join(", ") }} · Interval: {{ meta.interval }} ·
        Range: {{ meta.date_range.start }} to {{ meta.date_range.end }} UTC ·
        Generated: {{ meta.generated_at }} · Critic Source: {{ verdict.source | upper }} ·
        Verdict mode: {{ meta.strictness or scorecard.strictness or "balanced" }}
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
      <span class="badge badge-{{ verdict.verdict | replace(' ', '-') }}">
        {{ verdict.verdict }} · {{ "%.1f"|format(verdict.score) }}/100
      </span>
      <button class="theme-btn" onclick="toggleTheme()">Toggle Light/Dark</button>
    </div>
  </header>

  <div class="grid-2">
    <div class="card">{{ charts.gauge | safe }}</div>
    <div class="card">{{ charts.subscores | safe }}</div>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Full Total Return</div>
      <div class="kpi-value">{{ "%+.1f%%"|format(full_metrics.total_return * 100) }}</div>
      <div class="kpi-sub">CAGR {{ "%+.1f%%"|format(full_metrics.cagr * 100) }} · {{ full_metrics.n_trades }} trades</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Validation Sharpe</div>
      <div class="kpi-value">{{ "%.2f"|format(wf.validation_metrics.sharpe) }}</div>
      <div class="kpi-sub">Train {{ "%.2f"|format(wf.train_metrics.sharpe) }} ({{ "%+.0f%%"|format(-wf.sharpe_degradation_pct) }})</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Locked Test Sharpe</div>
      <div class="kpi-value">{{ "%.2f"|format(final_test.test_metrics.sharpe) }}</div>
      <div class="kpi-sub">Test Ret {{ "%+.1f%%"|format(final_test.test_metrics.total_return * 100) }} · {{ final_test.test_metrics.n_trades }} trades</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Max Drawdown</div>
      <div class="kpi-value">{{ "%.1f%%"|format(full_metrics.max_drawdown * 100) }}</div>
      <div class="kpi-sub">Longest {{ "%.0f"|format(full_metrics.max_drawdown_duration_days) }}d · Calmar {{ "%.2f"|format(full_metrics.calmar) }}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Expectancy & Win Rate</div>
      <div class="kpi-value">{{ "%+.3fR"|format(full_metrics.expectancy_r) }}</div>
      <div class="kpi-sub">Win {{ "%.1f%%"|format(full_metrics.win_rate * 100) }} · PF {{ "%.2f"|format(full_metrics.profit_factor) }}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Fees & Funding</div>
      <div class="kpi-value">${{ "{:,.0f}".format(full_metrics.total_fees) }}</div>
      <div class="kpi-sub">Net Funding ${{ "{:+,.0f}".format(full_metrics.total_funding) }}</div>
    </div>
  </div>

  <div class="card">
    <h2>{{ "AI" if verdict.source == "ai" else "Offline" }} Critic Executive Assessment <span class="badge badge-{{ verdict.verdict | replace(' ', '-') }}">{{ verdict.verdict }}</span></h2>
    <div class="callout">{{ verdict.summary }}</div>
    <h3>Scorecard Rule</h3>
    <p class="subtitle">Verdict mode: {{ meta.strictness or scorecard.strictness or "balanced" }} · Weighted score {{ "%.1f"|format(scorecard.weighted_score) }}/100 &rarr; final {{ "%.1f"|format(scorecard.overall_score) }}/100 after hard gates (ceiling: {{ scorecard.verdict_ceiling }}).</p>
    {% if scorecard.gates %}
    <table>
      <tr><th>Gate</th><th>Score Cap</th><th>Verdict Ceiling</th><th>Reason</th></tr>
      {% for g in scorecard.gates %}
      <tr><td class="mono">{{ g.gate }}</td><td>{{ "%.1f"|format(g.cap) }}</td><td>{{ g.ceiling }}</td><td>{{ g.reason }}</td></tr>
      {% endfor %}
    </table>
    {% else %}
    <p class="subtitle">No hard gate triggered.</p>
    {% endif %}
    <div class="grid-2">
      <div>
        <h3>Strengths</h3>
        <ul>{% for s in verdict.strengths %}<li>{{ s }}</li>{% endfor %}</ul>
        <h3>Weaknesses</h3>
        <ul>{% for w in verdict.weaknesses %}<li>{{ w }}</li>{% endfor %}</ul>
      </div>
      <div>
        <h3>Red Flags</h3>
        <ul>
          {% if verdict.red_flags %}{% for r in verdict.red_flags %}<li>{{ r }}</li>{% endfor %}
          {% else %}<li>No critical red flags raised.</li>{% endif %}
        </ul>
        <h3>Recommended Improvements</h3>
        <ul>{% for imp in verdict.improvements %}<li>{{ imp }}</li>{% endfor %}</ul>
        <p class="subtitle"><strong>Recommended Risk / Trade:</strong> {{ "%.2f%%"|format(verdict.recommended_risk_per_trade * 100) }}</p>
      </div>
    </div>
  </div>

  <div class="card">{{ charts.equity | safe }}</div>
  <div class="card">{{ charts.monthly | safe }}</div>

  <div class="card">
    <h2>1. Integrity & Lookahead Audit <span class="badge badge-{{ lookahead.status }}">{{ lookahead.status | upper }}</span></h2>
    <p>{{ lookahead.explanation }}</p>
    <table>
      <tr><th>Check</th><th>Result</th><th>Details</th></tr>
      <tr>
        <td>Causal Truncation Test</td>
        <td>{{ "PASS" if lookahead.truncation.passed else "FAIL" }}</td>
        <td>{{ lookahead.truncation.cuts_checked }} random cuts checked · {{ lookahead.truncation.total_mismatches }} mismatches</td>
      </tr>
      <tr>
        <td>1-Bar Execution Delay Test</td>
        <td>{{ "WARN" if lookahead.delay.warn else "PASS" }}</td>
        <td>Base Exp {{ "%+.3fR"|format(lookahead.delay.base_expectancy_r) }} -> Delayed {{ "%+.3fR"|format(lookahead.delay.delayed_expectancy_r) }} ({{ "%+.1f%%"|format(-lookahead.delay.expectancy_collapse_pct) }})</td>
      </tr>
    </table>
    {% if lookahead.truncation.examples %}
    <h3>Sample Causality Mismatch Bars</h3>
    <table>
      <tr><th>Symbol</th><th>Cut Time</th><th>Bar Time</th><th>Field</th><th>Full Value</th><th>Truncated Value</th></tr>
      {% for ex in lookahead.truncation.examples[:10] %}
      <tr>
        <td>{{ ex.symbol }}</td><td class="mono">{{ ex.cut_time }}</td><td class="mono">{{ ex.bar_time }}</td>
        <td>{{ ex.column }}</td><td class="mono">{{ ex.full_value }}</td><td class="mono">{{ ex.truncated_value }}</td>
      </tr>
      {% endfor %}
    </table>
    {% endif %}
  </div>

  <div class="card">
    <h2>2. Walk-Forward & Out-of-Sample Split <span class="badge badge-{{ wf.status }}">{{ wf.status | upper }}</span></h2>
    <p>{{ wf.explanation }} · <strong>Effective Range:</strong> {{ meta.date_range.start }} to {{ meta.date_range.end }} UTC · <strong>Chosen Params:</strong> <span class="mono">{{ final_test.final_params }}</span></p>
    <table>
      <tr><th>Split Window</th><th>Window (UTC)</th><th>Trades</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Expectancy (R)</th></tr>
      <tr><td>Train (60%)</td><td class="mono">{{ meta.splits.train_start[:10] }} &rarr; {{ meta.splits.train_end[:10] }}</td><td>{{ wf.train_metrics.n_trades }}</td><td>{{ "%+.1f%%"|format(wf.train_metrics.total_return * 100) }}</td><td>{{ "%.2f"|format(wf.train_metrics.sharpe) }}</td><td>{{ "%.1f%%"|format(wf.train_metrics.max_drawdown * 100) }}</td><td>{{ "%+.3fR"|format(wf.train_metrics.expectancy_r) }}</td></tr>
      <tr><td>Validation (20%)</td><td class="mono">{{ meta.splits.val_start[:10] }} &rarr; {{ meta.splits.val_end[:10] }}</td><td>{{ wf.validation_metrics.n_trades }}</td><td>{{ "%+.1f%%"|format(wf.validation_metrics.total_return * 100) }}</td><td>{{ "%.2f"|format(wf.validation_metrics.sharpe) }}</td><td>{{ "%.1f%%"|format(wf.validation_metrics.max_drawdown * 100) }}</td><td>{{ "%+.3fR"|format(wf.validation_metrics.expectancy_r) }}</td></tr>
      <tr><td>Rolling WF OOS (12m->3m)</td><td class="mono">{{ meta.splits.train_start[:10] }} &rarr; {{ meta.splits.val_end[:10] }}</td><td>{{ wf.rolling_oos_metrics.n_trades }}</td><td>{{ "%+.1f%%"|format(wf.rolling_oos_metrics.total_return * 100) }}</td><td>{{ "%.2f"|format(wf.rolling_oos_metrics.sharpe) }}</td><td>{{ "%.1f%%"|format(wf.rolling_oos_metrics.max_drawdown * 100) }}</td><td>{{ "%+.3fR"|format(wf.rolling_oos_metrics.expectancy_r) }}</td></tr>
      <tr><td><strong>Locked Test (20%)</strong></td><td class="mono">{{ meta.splits.test_start[:10] }} &rarr; {{ meta.splits.test_end[:10] }}</td><td>{{ final_test.test_metrics.n_trades }}</td><td>{{ "%+.1f%%"|format(final_test.test_metrics.total_return * 100) }}</td><td>{{ "%.2f"|format(final_test.test_metrics.sharpe) }}</td><td>{{ "%.1f%%"|format(final_test.test_metrics.max_drawdown * 100) }}</td><td>{{ "%+.3fR"|format(final_test.test_metrics.expectancy_r) }}</td></tr>
    </table>
    {{ charts.walkforward | safe }}
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>3. Overfitting Diagnostics <span class="badge badge-{{ overfit.status }}">{{ overfit.status | upper }}</span></h2>
      <p>{{ overfit.explanation }}</p>
      {{ charts.param_heatmap | safe }}
    </div>
    <div class="card">
      <h2>4. Transaction Cost Stress <span class="badge badge-{{ costs.status }}">{{ costs.status | upper }}</span></h2>
      <p>{{ costs.explanation }}</p>
      {{ charts.cost_stress | safe }}
    </div>
  </div>

  <div class="grid-2">
    <div class="card">{{ charts.montecarlo | safe }}</div>
    <div class="card">{{ charts.random_baseline | safe }}</div>
  </div>

  <div class="card">
    <h2>5. Regimes, Long/Short & Outlier Sensitivity <span class="badge badge-{{ regime.status }}">{{ regime.status | upper }}</span></h2>
    <p>{{ regime.explanation }}</p>
    {{ charts.regime | safe }}
    <div class="grid-2">
      <table>
        <tr><th>Symbol</th><th>Trades</th><th>Win Rate</th><th>Avg R</th><th>Net PnL</th><th>Profit Share</th></tr>
        {% for s in regime.symbols %}
        <tr><td>{{ s.symbol }}</td><td>{{ s.n_trades }}</td><td>{{ "%.1f%%"|format(s.win_rate * 100) }}</td><td>{{ "%+.3fR"|format(s.avg_r) }}</td><td>${{ "{:+,.0f}".format(s.net_pnl) }}</td><td>{{ "%.1f%%"|format(s.profit_share_pct) }}</td></tr>
        {% endfor %}
      </table>
      <table>
        <tr><th>Direction / Slice</th><th>Trades</th><th>Win Rate</th><th>Avg R</th><th>Net PnL</th><th>PF</th></tr>
        <tr><td>LONG</td><td>{{ regime.long_vs_short.long.n_trades }}</td><td>{{ "%.1f%%"|format(regime.long_vs_short.long.win_rate * 100) }}</td><td>{{ "%+.3fR"|format(regime.long_vs_short.long.avg_r) }}</td><td>${{ "{:+,.0f}".format(regime.long_vs_short.long.net_pnl) }}</td><td>{{ "%.2f"|format(regime.long_vs_short.long.profit_factor) }}</td></tr>
        <tr><td>SHORT</td><td>{{ regime.long_vs_short.short.n_trades }}</td><td>{{ "%.1f%%"|format(regime.long_vs_short.short.win_rate * 100) }}</td><td>{{ "%+.3fR"|format(regime.long_vs_short.short.avg_r) }}</td><td>${{ "{:+,.0f}".format(regime.long_vs_short.short.net_pnl) }}</td><td>{{ "%.2f"|format(regime.long_vs_short.short.profit_factor) }}</td></tr>
        <tr><td>Top 5% Removed</td><td>-{{ regime.outlier_check.trades_removed }}</td><td>—</td><td>{{ "%+.3fR"|format(regime.outlier_check.trimmed_avg_r) }}</td><td>${{ "{:+,.0f}".format(regime.outlier_check.trimmed_net_pnl) }}</td><td>{{ "YES" if regime.outlier_check.still_profitable else "NO" }}</td></tr>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>6. Two-Step Prop-Firm Challenge Simulation <span class="badge badge-{{ propfirm.status }}">{{ propfirm.status | upper }}</span></h2>
    <p>{{ propfirm.explanation }}</p>
    <table>
      <tr><th>Start Weeks</th><th>Passed</th><th>Failed</th><th>Incomplete</th><th>Pass Rate</th><th>Median Days</th><th>p90 Days</th></tr>
      <tr>
        <td>{{ propfirm.n_attempts }}</td><td>{{ propfirm.n_passed }}</td><td>{{ propfirm.n_failed }}</td>
        <td>{{ propfirm.n_incomplete }}</td><td>{{ "%.1f%%"|format(propfirm.pass_rate * 100) }}</td>
        <td>{{ propfirm.median_days_to_pass if propfirm.median_days_to_pass is not none else "N/A" }}</td>
        <td>{{ propfirm.p90_days_to_pass if propfirm.p90_days_to_pass is not none else "N/A" }}</td>
      </tr>
    </table>
  </div>

  <div class="card">
    <h2>7. Agent Investigation Log</h2>
    <table>
      <tr><th>#</th><th>Tool</th><th>Arguments</th><th>Result Summary</th><th>Time (s)</th></tr>
      {% for entry in tool_log %}
      <tr>
        <td>{{ entry.step }}</td><td class="mono">{{ entry.tool }}</td>
        <td class="mono">{{ entry.args }}</td><td>{{ entry.summary }}</td><td>{{ entry.elapsed_s }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>

  <footer>QuantCritic · "Most crypto backtests lie. QuantCritic finds out how." · Part of the Crypto Quant Suite</footer>
</div>
<script>
  function toggleTheme() {
    const el = document.documentElement;
    el.setAttribute('data-theme', el.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  }
</script>
</body>
</html>
"""

LEADERBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>QuantCritic Strategy Leaderboard</title>
<style>
  body { margin: 0; font-family: Inter, system-ui, sans-serif; background: #0B1020; color: #E6E9F2; }
  .container { max-width: 1280px; margin: 0 auto; padding: 32px 24px; }
  h1 { margin: 0 0 8px; }
  .sub { color: #94A3B8; margin-bottom: 24px; }
  .card { background: #131A2E; border: 1px solid #232E4D; border-radius: 12px; padding: 18px; margin-bottom: 22px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 10px 10px; border-bottom: 1px solid #232E4D; }
  th { color: #94A3B8; text-transform: uppercase; font-size: 12px; }
  .badge { padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; }
  .PROMISING { background: rgba(34,197,94,0.18); color: #22C55E; border: 1px solid #22C55E; }
  .NEEDS-WORK { background: rgba(245,158,11,0.18); color: #F59E0B; border: 1px solid #F59E0B; }
  .REJECT { background: rgba(239,68,68,0.18); color: #EF4444; border: 1px solid #EF4444; }
</style>
</head>
<body>
<div class="container">
  <h1>QuantCritic Strategy Leaderboard</h1>
  <div class="sub">Symbols: {{ symbols | join(", ") }} · Interval: {{ interval }}{% if date_range %} · Range: {{ date_range }}{% endif %}{% if strictness %} · Verdict mode: {{ strictness }}{% endif %} · Generated: {{ generated_at }}</div>
  <div class="card">{{ chart | safe }}</div>
  <div class="card">
    <table>
      <tr>
        <th>Rank</th><th>Strategy</th><th>Verdict</th><th>Score</th><th>Lookahead</th>
        <th>Val Sharpe</th><th>Test Sharpe</th><th>DSR</th><th>PBO</th><th>p-val</th><th>Break-Even bps</th><th>Full Return</th>
      </tr>
      {% for row in rows %}
      <tr>
        <td>#{{ loop.index }}</td>
        <td><strong>{{ row.strategy }}</strong></td>
        <td><span class="badge {{ row.verdict | replace(' ', '-') }}">{{ row.verdict }}</span></td>
        <td><strong>{{ "%.1f"|format(row.score) }}</strong></td>
        <td>{{ row.lookahead_status | upper }}</td>
        <td>{{ "%.2f"|format(row.val_sharpe) }}</td>
        <td>{{ "%.2f"|format(row.test_sharpe) }}</td>
        <td>{{ "%.2f"|format(row.dsr) }}</td>
        <td>{{ "%.2f"|format(row.pbo) }}</td>
        <td>{{ "%.3f"|format(row.p_value) }}</td>
        <td>{{ "%.1f"|format(row.break_even_bps) }}</td>
        <td>{{ "%+.1f%%"|format(row.full_return * 100) }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
</div>
</body>
</html>
"""


def build_report_payload(session: AuditSession) -> dict[str, Any]:
    """Assemble a JSON-serializable dictionary of the entire strategy audit."""
    scorecard = session.ensure_complete_audit()
    if session.verdict is None:
        raise RuntimeError("AuditSession has no verdict recorded")

    full_res = session.cached_results["full_result"]
    wf = session.cached_results["walkforward"]
    ov = session.cached_results["overfit"]
    mc = session.cached_results["montecarlo"]
    la = session.cached_results["lookahead"]
    cs = session.cached_results["costs"]
    reg = session.cached_results["regime"]
    pf = session.cached_results["propfirm"]
    ft = session.cached_results["final_test"]

    train_start = session.splits.train_start
    if train_start is None:
        non_empty_tr = [df for df in session.splits.train.ohlcv.values() if not df.empty]
        train_start = min(df.index[0] for df in non_empty_tr) if non_empty_tr else session.splits.train_end
    splits_meta = {
        "train_start": str(train_start),
        "train_end": str(session.splits.train_end),
        "val_start": str(session.splits.val_start),
        "val_end": str(session.splits.val_end),
        "test_start": str(session.splits.test_start),
        "test_end": str(session.splits.test_end),
    }
    date_range_meta = {
        "start": str(train_start)[:10],
        "end": str(session.splits.test_end)[:10],
        "start_utc": str(train_start),
        "end_utc": str(session.splits.test_end),
    }

    return {
        "meta": {
            "strategy": session.strategy.name,
            "description": session.strategy.description,
            "symbols": session.data.symbols,
            "interval": session.data.interval,
            "strictness": session.strictness or scorecard.get("strictness", "balanced"),
            "date_range": date_range_meta,
            "splits": splits_meta,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "chosen_params": session.chosen_params,
            "config": session.config.model_dump(),
        },
        "verdict": session.verdict.model_dump(),
        "scorecard": scorecard,
        "full_metrics": full_res.metrics,
        "lookahead": la,
        "walkforward": {k: v for k, v in wf.items() if k not in ("rolling_oos_equity", "_combo_signals")},
        "overfit": {
            "status": ov["status"],
            "explanation": ov["explanation"],
            "n_trials": ov["n_trials"],
            "dsr": ov["dsr"],
            "pbo": {k: v for k, v in ov["pbo"].items() if k != "logits"},
            "stability": ov["stability"],
            "heatmap": ov["heatmap"],
        },
        "montecarlo": {
            "status": mc["status"],
            "explanation": mc["explanation"],
            "bootstrap": {k: v for k, v in mc["bootstrap"].items() if k not in ("return_samples", "max_dd_samples")},
            "random_baseline": {k: v for k, v in mc["random_baseline"].items() if k != "baseline_expectancies"},
        },
        "costs": cs,
        "regime": reg,
        "propfirm": pf,
        "final_test": ft,
        "tool_log": session.tool_log,
        "agent_thoughts": session.agent_thoughts,
    }


def render_audit_html_from_payload(
    payload: dict[str, Any],
    equity_curve: Any,
    wf_equity: Any = None,
    bootstrap_data: dict[str, Any] | None = None,
    random_baseline_data: dict[str, Any] | None = None,
) -> str:
    """Render a single-file HTML audit report string from a report payload and series."""
    verdict_d = payload["verdict"]
    sc = payload["scorecard"]
    full_m = payload["full_metrics"]
    wf = dict(payload["walkforward"])
    if wf_equity is not None:
        wf["rolling_oos_equity"] = wf_equity
    ov = payload["overfit"]
    cs = payload["costs"]
    reg = payload["regime"]
    bs = bootstrap_data or payload.get("_bootstrap_samples") or payload["montecarlo"]["bootstrap"]
    rb = random_baseline_data or payload.get("_random_baseline_samples") or payload["montecarlo"]["random_baseline"]

    def _to_html(fig: Any, include_js: bool = False) -> str:
        return pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn" if include_js else False,
            config={"displaylogo": False, "responsive": True},
        )

    charts_html = {
        "gauge": _to_html(
            make_scorecard_gauge(
                verdict_d["score"],
                verdict_d["verdict"],
                promising_threshold=float(sc.get("promising_threshold", 70.0)),
            ),
            include_js=True,
        ),
        "subscores": _to_html(make_subscores_bar(sc["sub_scores"])),
        "equity": _to_html(
            make_equity_drawdown_figure(
                equity_curve,
                payload["meta"]["config"]["initial_capital"],
                wf.get("splits_summary"),
            )
        ),
        "monthly": _to_html(make_monthly_heatmap_figure(full_m["monthly_returns"])),
        "walkforward": _to_html(make_walkforward_figure(wf)),
        "param_heatmap": _to_html(make_param_heatmap(ov["heatmap"])),
        "cost_stress": _to_html(make_cost_stress_figure(cs)),
        "montecarlo": _to_html(make_montecarlo_figure(bs)),
        "random_baseline": _to_html(make_random_baseline_figure(rb)),
        "regime": _to_html(make_regime_figure(reg)),
    }

    tmpl = _JINJA_ENV.from_string(AUDIT_TEMPLATE)
    return tmpl.render(
        meta=payload["meta"],
        verdict=verdict_d,
        scorecard=sc,
        full_metrics=full_m,
        lookahead=payload["lookahead"],
        wf=wf,
        overfit=ov,
        costs=cs,
        regime=reg,
        propfirm=payload["propfirm"],
        final_test=payload["final_test"],
        tool_log=payload["tool_log"],
        charts=charts_html,
    )


def generate_audit_reports(
    session: AuditSession,
    output_dir: Path | None = None,
    basename: str | None = None,
) -> tuple[Path, Path]:
    """Write a single-file HTML report (Plotly.js from CDN) and a JSON report to `output_dir`."""
    payload = build_report_payload(session)
    out_dir = output_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_stem = basename or f"{session.strategy.name}_{ts}"
    stem = re.sub(r'[^A-Za-z0-9._-]+', "_", raw_stem)

    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"

    json_path.write_text(json.dumps(to_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")

    full_res = session.cached_results["full_result"]
    wf = session.cached_results["walkforward"]
    mc = session.cached_results["montecarlo"]
    rendered = render_audit_html_from_payload(
        payload=payload,
        equity_curve=full_res.equity_curve,
        wf_equity=wf.get("rolling_oos_equity"),
        bootstrap_data=mc["bootstrap"],
        random_baseline_data=mc["random_baseline"],
    )
    html_path.write_text(rendered, encoding="utf-8")
    return html_path, json_path


def generate_leaderboard_report(
    rows: list[dict[str, Any]],
    symbols: list[str],
    interval: str,
    output_dir: Path | None = None,
    date_range: tuple[str, str] | None = None,
    strictness: str | None = None,
) -> tuple[Path, Path]:
    """Generate reports/leaderboard.html and reports/leaderboard.json from compared strategies."""
    import plotly.graph_objects as go

    out_dir = output_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)

    colors = [
        "#22C55E" if r["verdict"] == "PROMISING" else ("#F59E0B" if r["verdict"] == "NEEDS WORK" else "#EF4444")
        for r in sorted_rows
    ]
    fig = go.Figure(
        go.Bar(
            x=[r["strategy"] for r in sorted_rows],
            y=[r["score"] for r in sorted_rows],
            marker_color=colors,
            text=[f"{r['score']:.1f} ({r['verdict']})" for r in sorted_rows],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="QuantCritic Strategy Leaderboard (0–100 Robustness Score)",
        paper_bgcolor="#131A2E",
        plot_bgcolor="#0B1020",
        font={"color": "#E6E9F2"},
        yaxis={"range": [0, 112], "title": "Score"},
        height=360,
        margin={"l": 45, "r": 25, "t": 50, "b": 45},
    )
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config={"responsive": True})

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    range_str = f"{date_range[0]} to {date_range[1]} UTC" if date_range else None
    tmpl = _JINJA_ENV.from_string(LEADERBOARD_TEMPLATE)
    html_str = tmpl.render(
        rows=sorted_rows,
        symbols=symbols,
        interval=interval,
        date_range=range_str,
        strictness=strictness,
        generated_at=now_str,
        chart=chart_html,
    )
    html_path = out_dir / "leaderboard.html"
    json_path = out_dir / "leaderboard.json"
    html_path.write_text(html_str, encoding="utf-8")
    json_path.write_text(json.dumps(to_jsonable(sorted_rows), indent=2, allow_nan=False), encoding="utf-8")
    return html_path, json_path
