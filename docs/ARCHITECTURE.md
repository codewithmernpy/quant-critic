# QuantCritic Architecture

## System Overview

QuantCritic separates market ingestion, deterministic backtesting, statistical robustness auditing, AI tool-calling orchestration, and presentation into decoupled modules:

```
data/sample/*.csv.gz ──▶ quantcritic.data (loader, resample, binance, synthetic)
                                │
                                ▼
quantcritic.strategies ──▶ quantcritic.engine (event-driven backtest + metrics)
  (5 honest + leaky_demo)       ▲
                                │ (sizing via quantcritic.risk.sizing)
                                ▼
                        quantcritic.audit (8 statistical & integrity modules)
                                │
                                ▼
                        quantcritic.agent (AuditSession + Test-Set Lock)
                          ├── critic.py  (Claude tool_runner via anthropic 2.x)
                          └── offline.py (deterministic rule-based critic)
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
     quantcritic.report                 Streamlit Web App
   (HTML + JSON + Leaderboard)      (streamlit_app.py + 7 pages)
```

## Module Responsibilities

- **`quantcritic/config.py`**: Pydantic configuration schemas (`BacktestConfig`, `SplitConfig`, `PropFirmConfig`, `AuditConfig`), path constants, and `.env` loader.
- **`quantcritic/data/`**:
  - `loader.py`: Discovers available symbols across `data/sample/` and `data/cache/`, merges sample + cache without duplicates or gaps (`cache` wins on overlap, `data/sample/` is never modified), parses UTC timestamps normalized to `datetime64[ns, UTC]`, rounds funding jitter to nearest minute, resamples `1h -> 4h / 1d` using period-open labels while dropping exchange outage gaps, attaches causal `funding_rate` features, and prepares `[start, end]` range windows while retaining pre-start history for indicator warm-up (`prepare_range_data`, `validate_date_range`).
  - `binance.py`: Public downloader for `data.binance.vision` monthly USDT-M klines (`1h`) and `fundingRate` zip archives supporting symbol normalization/validation (`^[A-Z0-9]{2,20}USDT$`), incremental missing-month caching into `data/cache/`, later-listed coin handling, latest complete month updates, header/no-header CSVs, and millisecond/microsecond epoch detection.
  - `synthetic.py`: Seeded random-walk OHLCV, AR(1) funding, and explicit bar builders for deterministic testing.
- **`quantcritic/risk/sizing.py`**: Fixed-fractional sizing with portfolio leverage caps, volatility targeting, capped fractional Kelly, drawdown multipliers, and exponential-barrier risk of ruin. (Live liquidation/correlation monitoring is out of scope; a separate RiskGuard project is planned.)
- **`quantcritic/engine/`**:
  - `backtest.py`: Pre-aligns multi-symbol arrays via `PreparedMarket` onto a unified UTC timeline, applies next-bar open entries, worst-case stop-before-TP intraday checks, gap-through-open fills, fees, slippage, and per-bar funding accruals; `trade_start` evaluates a window with warm signals (validation, walk-forward OOS, locked test).
  - `metrics.py`: Computes 365-day annualized Sharpe/Sortino, max drawdown & duration, monthly returns matrix, and directional long/short stats.
- **`quantcritic/strategies/`**: Base `Strategy` interface, auto-discovering registry (`registry.py`), 5 strictly causal built-in strategies (`ema_trend`, `donchian_breakout`, `rsi_reversion`, `bollinger_squeeze`, `funding_contrarian`), `leaky_demo`, causal no-code `BuilderStrategy` / `StrategySpec` + Claude plain-English parser (`builder.py`), and `custom/` Python plugin auto-loader (with `template_strategy.py`).
- **`quantcritic/audit/`**: 8 modular diagnostics returning uniform `status` (`pass`/`warn`/`fail`), numeric metrics, and one-line explanations, combined by `scorecard.py` under configurable `strictness` modes (`strict`, `balanced` [default], `relaxed`).
- **`quantcritic/agent/`**:
  - `tools.py`: `AuditSession` enforces the strict test-set lock (the test slice is a `LockedSlice` whose `unlock()` is called only by `final_test()`, at most once, and counted in `test_access_count`; `run_backtest()` is locked afterward), drops cached audits when the working parameters change, applies the strictness-aware scorecard hard gates to any submitted verdict, and constructs the 12 `@beta_tool` closures returning compact, always-valid JSON (< 3000 chars).
  - `critic.py`: Invokes `anthropic.Anthropic().beta.messages.tool_runner(model, max_tokens=16000, max_iterations=24, system, output_config={"effort": "high"}, betas=["server-side-fallback-2026-07-01"], fallbacks="default", tools, messages)`; on `stop_reason == "refusal"` or any `anthropic.APIError` it falls back to `offline.py`.
- **`quantcritic/report/`**: Shared Plotly chart builders (`charts.py`) and Jinja2 (autoescaped) single-file HTML (Plotly.js from CDN) + JSON exporters (`html.py`).
- **`streamlit_app.py` + `pages/` + `quantcritic/ui.py`**: 7-page dark-themed interactive dashboard (Pages `1_Backtest`..`5_Lookahead_Lab`, `7_Strategy_Builder`, `8_About_Methodology`) with `@st.cache_data` caching, strictness selector, and session-only API key entry.
