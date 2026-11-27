"""Shared Streamlit UI helpers, sidebar controls, CSS styling, and cached runners."""

from __future__ import annotations

from datetime import date
import json
import math
import os
from typing import Any

import pandas as pd
import streamlit as st

from quantcritic.agent.offline import run_offline_critic
from quantcritic.agent.tools import AuditSession
from quantcritic.audit.lookahead import run_lookahead_audit
from quantcritic.audit.walkforward import run_window_backtest
from quantcritic.config import (
    DEFAULT_MODEL,
    DEFAULT_STRICTNESS,
    DEFAULT_SYMBOLS,
    STRICTNESS_MODES,
    SUPPORTED_INTERVALS,
    AuditConfig,
    BacktestConfig,
)
from quantcritic.data.binance import (
    ensure_symbol_data,
    latest_complete_month,
    parse_symbols_input,
    update_symbol_to_latest,
)
from quantcritic.data.loader import (
    MarketData,
    get_available_date_bounds,
    list_available_symbols,
    load_market_data,
    parse_utc_bound,
    prepare_range_data,
    validate_date_range,
)
from quantcritic.engine.backtest import BacktestResult, run_backtest
from quantcritic.report.html import build_report_payload, generate_audit_reports
from quantcritic.strategies.registry import get_strategy, list_strategies

CUSTOM_CSS = """
<style>
  .qc-badge {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-weight: 700; font-size: 0.85rem; letter-spacing: 0.04em;
  }
  .qc-PROMISING, .qc-pass {
    background: rgba(34, 197, 94, 0.18); color: #22C55E; border: 1px solid #22C55E;
  }
  .qc-NEEDS-WORK, .qc-warn {
    background: rgba(245, 158, 11, 0.18); color: #F59E0B; border: 1px solid #F59E0B;
  }
  .qc-REJECT, .qc-fail {
    background: rgba(239, 68, 68, 0.18); color: #EF4444; border: 1px solid #EF4444;
  }
  .qc-card {
    background: #131A2E; border: 1px solid #232E4D; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 12px;
  }
  .qc-footer {
    margin-top: 2rem; padding-top: 0.75rem; border-top: 1px solid #232E4D;
    color: #94A3B8; font-size: 0.78rem; line-height: 1.5;
  }
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
  }
  [data-testid="stMetricValue"] {
    font-size: clamp(1.1rem, 2.2vw, 1.8rem);
  }
</style>
"""


def format_price(price: float) -> str:
    """Format a price with thousands separators and adaptive decimal precision."""
    val = float(price)
    if not math.isfinite(val):
        return "N/A"
    ap = abs(val)
    if ap == 0.0 or ap >= 1.0:
        decimals = 2
    elif ap >= 0.01:
        decimals = 4
    else:
        decimals = min(12, max(4, -math.floor(math.log10(ap)) + 3))
    return f"{val:,.{decimals}f}"


def init_page(title: str, icon: str = "🔍") -> None:
    """Configure wide layout and inject dark Quant Suite CSS."""
    st.set_page_config(page_title=f"QuantCritic · {title}", page_icon=icon, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def badge_html(label: str) -> str:
    """Return a styled HTML status pill for verdicts or pass/warn/fail badges."""
    cls = label.strip().replace(" ", "-")
    if cls.lower() in ("pass", "warn", "fail"):
        cls = cls.lower()  # CSS classes are case-sensitive: .qc-pass / .qc-warn / .qc-fail
    return f'<span class="qc-badge qc-{cls}">{label}</span>'


def render_sidebar() -> dict[str, Any]:
    """Render shared QuantCritic sidebar controls and suite footer."""
    with st.sidebar:
        st.markdown("## QuantCritic")
        st.caption("Most crypto backtests lie. QuantCritic finds out how.")
        st.divider()

        strat_names = list_strategies(include_leaky=True)
        if "_qc_pending_strategy" in st.session_state:
            pending_strat = st.session_state.pop("_qc_pending_strategy")
            if pending_strat in strat_names:
                st.session_state["qc_strategy"] = pending_strat
                st.session_state["sb_strategy"] = pending_strat
        default_strat = st.session_state.get("qc_strategy", "donchian_breakout")
        if default_strat not in strat_names:
            default_strat = "donchian_breakout"
        # Re-assign EVERY run: Streamlit gives a keyed widget a new identity on another page and would otherwise show
        # its first option (browser-verified: a strategy picked on Backtest was lost on Leaderboard).
        cur_strat = st.session_state.get("sb_strategy", default_strat)
        st.session_state["sb_strategy"] = cur_strat if cur_strat in strat_names else default_strat
        strategy_name = st.selectbox(
            "Strategy",
            options=strat_names,
            key="sb_strategy",
        )
        st.session_state["qc_strategy"] = strategy_name

        available_symbols = list_available_symbols()
        if "_qc_pending_symbols" in st.session_state:
            pending_syms = st.session_state.pop("_qc_pending_symbols")
            for sym in pending_syms:
                if sym not in available_symbols:
                    available_symbols.append(sym)
            st.session_state["qc_symbols"] = list(pending_syms)
            st.session_state["sb_symbols"] = list(pending_syms)
        if st.session_state.pop("_qc_clear_add_input", False):
            st.session_state["sb_add_coins"] = ""

        default_syms = [
            s for s in st.session_state.get("qc_symbols", list(DEFAULT_SYMBOLS))
            if s in available_symbols
        ]
        if not default_syms:
            default_syms = [s for s in DEFAULT_SYMBOLS if s in available_symbols] or available_symbols[:1]
        if "sb_symbols" not in st.session_state:
            st.session_state["sb_symbols"] = list(default_syms)
        else:
            st.session_state["sb_symbols"] = [
                s for s in st.session_state["sb_symbols"] if s in available_symbols
            ]

        symbols = st.multiselect(
            "Symbols",
            options=available_symbols,
            key="sb_symbols",
        )
        if not symbols:
            fallback_sym = "BTCUSDT" if "BTCUSDT" in available_symbols else (available_symbols[0] if available_symbols else "BTCUSDT")
            symbols = [fallback_sym]
        st.session_state["qc_symbols"] = list(symbols)

        add_coins_raw = st.text_input(
            "Add coins",
            placeholder="e.g. XRPUSDT, DOGEUSDT or xrp",
            help="Comma-separated Binance USDT-M coins (e.g. 'XRPUSDT, DOGEUSDT' or 'xrp' -> 'XRPUSDT').",
            key="sb_add_coins",
        )
        for lvl, msg in st.session_state.pop("_qc_sidebar_notices", []):
            getattr(st, lvl)(msg)

        if add_coins_raw and add_coins_raw.strip() and add_coins_raw.strip() != st.session_state.get("_qc_last_add_raw"):
            st.session_state["_qc_last_add_raw"] = add_coins_raw.strip()
            valid_new, invalid_tokens = parse_symbols_input(add_coins_raw)
            notices: list[tuple[str, str]] = []
            if invalid_tokens:
                notices.append((
                    "error",
                    "Invalid symbol format: "
                    + ", ".join(repr(t) for t in invalid_tokens)
                    + ". Expected ^[A-Z0-9]{2,20}USDT$ (or base ticker like 'xrp').",
                ))

            added_any = False
            updated_syms = list(symbols)
            if valid_new:
                missing_locally = [s for s in valid_new if s not in available_symbols]
                already_local = [s for s in valid_new if s in available_symbols]
                for s in already_local:
                    if s not in updated_syms:
                        updated_syms.append(s)
                        added_any = True
                    notices.append(("info", f"{s} is already available locally; selected."))

                if missing_locally:
                    prog_bar = st.progress(0, text="Preparing Binance Vision download...")
                    status_txt = st.empty()
                    end_m = latest_complete_month()
                    n_missing = len(missing_locally)
                    for idx, sym in enumerate(missing_locally):
                        def _cb(step: int, total: int, label: str, _idx: int = idx) -> None:
                            frac = (_idx + (step / max(total, 1))) / n_missing
                            prog_bar.progress(min(99, max(0, int(frac * 100))), text=label)
                            status_txt.caption(label)

                        try:
                            res = ensure_symbol_data(
                                sym,
                                interval="1h",
                                start_month="2021-01",
                                end_month=end_m,
                                progress_callback=_cb,
                            )
                        except Exception as exc:
                            notices.append(("error", f"Failed to download {sym}: {exc}"))
                            continue

                        if res["status"] == "not_found":
                            notices.append((
                                "error",
                                f"{sym}: coin not found on Binance Vision (no archives in 2021-01..{end_m}).",
                            ))
                        else:
                            added_any = True
                            if sym not in updated_syms:
                                updated_syms.append(sym)
                            if res.get("listed_later"):
                                notices.append((
                                    "info",
                                    f"{sym}: listed later than 2021-01; using first available month {res['first_available_month']}.",
                                ))
                            else:
                                notices.append((
                                    "success",
                                    f"Downloaded {sym} ({res['first_available_month']}..{res['last_available_month']}).",
                                ))
                    prog_bar.progress(100, text="Download complete.")
                    status_txt.empty()

            if added_any:
                st.cache_data.clear()
                st.session_state["_qc_data_version"] = st.session_state.get("_qc_data_version", 0) + 1
                st.session_state["_qc_pending_symbols"] = updated_syms
                st.session_state["_qc_clear_add_input"] = True
                st.session_state["_qc_last_add_raw"] = ""
                st.session_state["_qc_sidebar_notices"] = notices
                st.rerun()
            else:
                for lvl, msg in notices:
                    getattr(st, lvl)(msg)

        if st.button(
            "Update data to latest month",
            key="sb_update_data",
            help="Download only missing monthly archives up to the latest complete month on data.binance.vision into data/cache.",
        ):
            prog_bar = st.progress(0, text="Checking Binance Vision for missing months...")
            status_txt = st.empty()
            end_m = latest_complete_month()
            upd_notices: list[tuple[str, str]] = []
            any_updated = False
            n_syms = len(symbols)
            for idx, sym in enumerate(symbols):
                def _ucb(step: int, total: int, label: str, _idx: int = idx) -> None:
                    frac = (_idx + (step / max(total, 1))) / max(n_syms, 1)
                    prog_bar.progress(min(99, max(0, int(frac * 100))), text=label)
                    status_txt.caption(label)

                try:
                    res = update_symbol_to_latest(sym, end_month=end_m, progress_callback=_ucb)
                except Exception as exc:
                    upd_notices.append(("error", f"Failed updating {sym}: {exc}"))
                    continue

                if res["downloaded_months"]:
                    any_updated = True
                    upd_notices.append((
                        "success",
                        f"{sym}: downloaded {len(res['downloaded_months'])} new month(s) through {res['last_available_month']}.",
                    ))
                else:
                    upd_notices.append((
                        "info",
                        f"{sym}: already up to date through {res.get('last_available_month', 'N/A')}.",
                    ))

            prog_bar.progress(100, text="Update check complete.")
            status_txt.empty()
            if any_updated:
                st.cache_data.clear()
                st.session_state["_qc_data_version"] = st.session_state.get("_qc_data_version", 0) + 1
                st.session_state["_qc_sidebar_notices"] = upd_notices
                st.rerun()
            else:
                for lvl, msg in upd_notices:
                    getattr(st, lvl)(msg)

        intervals = ["4h", "1h", "1d"]
        default_int = st.session_state.get("qc_interval", "4h")
        if default_int not in intervals:
            default_int = "4h"
        interval = st.selectbox(
            "Interval",
            options=intervals,
            index=intervals.index(default_int),
            key="sb_interval",
        )
        st.session_state["qc_interval"] = interval

        # Date range bounded by available data for selected symbols
        data_version = int(st.session_state.get("_qc_data_version", 0))
        avail_min, avail_max = get_available_date_bounds(cached_load_data(tuple(symbols), interval))
        prev_bounds = st.session_state.get("_qc_prev_avail_bounds")
        if "sb_start_date" not in st.session_state:
            st.session_state["sb_start_date"] = st.session_state.get("qc_start_date", avail_min)
        if "sb_end_date" not in st.session_state:
            st.session_state["sb_end_date"] = st.session_state.get("qc_end_date", avail_max)

        if prev_bounds is not None and prev_bounds != (avail_min, avail_max):
            if st.session_state["sb_start_date"] == prev_bounds[0]:
                st.session_state["sb_start_date"] = avail_min
            if st.session_state["sb_end_date"] == prev_bounds[1]:
                st.session_state["sb_end_date"] = avail_max

        st.session_state["sb_start_date"] = min(max(st.session_state["sb_start_date"], avail_min), avail_max)
        st.session_state["sb_end_date"] = min(max(st.session_state["sb_end_date"], avail_min), avail_max)
        st.session_state["_qc_prev_avail_bounds"] = (avail_min, avail_max)

        d_col1 = d_col2 = st                     # stacked: full-width date boxes in the narrow sidebar
        start_date = d_col1.date_input(
            "Start date",
            min_value=avail_min,
            max_value=avail_max,
            key="sb_start_date",
        )
        end_date = d_col2.date_input(
            "End date",
            min_value=avail_min,
            max_value=avail_max,
            key="sb_end_date",
        )
        st.session_state["qc_start_date"] = start_date
        st.session_state["qc_end_date"] = end_date
        st.caption(f"Available: `{avail_min.isoformat()}` to `{avail_max.isoformat()}`")
        date_error: str | None = None
        if start_date > end_date:
            date_error = f"Start date ({start_date.isoformat()}) must be on or before end date ({end_date.isoformat()})."
            st.error(date_error)

        strat = get_strategy(strategy_name)
        params: dict[str, Any] = dict(strat.default_params)
        with st.expander("Strategy Parameters", expanded=False):
            for k, val in strat.default_params.items():
                w_key = f"param_{strategy_name}_{k}"
                if isinstance(val, int) and not isinstance(val, bool):
                    min_v = 0 if (k == "max_bars" or int(val) == 0) else 1
                    params[k] = int(
                        st.number_input(k, value=int(st.session_state.get(f"qc_{w_key}", int(val))), min_value=min_v, step=1, key=w_key)
                    )
                    st.session_state[f"qc_{w_key}"] = params[k]
                elif isinstance(val, float):
                    params[k] = float(
                        st.number_input(k, value=float(st.session_state.get(f"qc_{w_key}", float(val))), step=0.25, format="%.3f", key=w_key)
                    )
                    st.session_state[f"qc_{w_key}"] = params[k]
                else:
                    params[k] = val

        with st.expander("Execution & Cost Settings", expanded=False):
            # value= comes from a non-widget mirror (qc_*) so choices survive page changes
            _g = st.session_state.get
            fee_bps = st.number_input("Taker Fee (bps/side)", value=float(_g("qc_fee_bps", 5.0)), min_value=0.0, max_value=50.0, step=0.5, key="sb_fee_bps")
            slippage_bps = st.number_input("Slippage (bps/side)", value=float(_g("qc_slip_bps", 2.0)), min_value=0.0, max_value=50.0, step=0.5, key="sb_slip_bps")
            risk_pct = st.number_input("Risk per Trade (%)", value=float(_g("qc_risk_pct", 0.5)), min_value=0.1, max_value=5.0, step=0.1, key="sb_risk_pct")
            max_lev = st.number_input("Max Portfolio Leverage (x)", value=float(_g("qc_max_lev", 3.0)), min_value=0.5, max_value=20.0, step=0.5, key="sb_max_lev")
            funding_on = st.checkbox("Include Perpetual Funding", value=bool(_g("qc_funding", True)), key="sb_funding")
            st.session_state.update(qc_fee_bps=fee_bps, qc_slip_bps=slippage_bps, qc_risk_pct=risk_pct, qc_max_lev=max_lev, qc_funding=funding_on)

        strictness_options = ["balanced", "strict", "relaxed"]
        default_strict = st.session_state.get("qc_strictness", DEFAULT_STRICTNESS)
        if default_strict not in strictness_options:
            default_strict = DEFAULT_STRICTNESS
        cur_strict = st.session_state.get("sb_strictness", default_strict)
        st.session_state["sb_strictness"] = cur_strict if cur_strict in strictness_options else default_strict
        strictness = st.selectbox(
            "Verdict Strictness",
            options=strictness_options,
            key="sb_strictness",
            help=(
                "strict: original gates (p<=0.10, DSR>=0.50, PROMISING>=70); "
                "balanced (default): p<=0.20, DSR>=0.30, PROMISING>=65; "
                "relaxed: lookahead + test trades>=5 + base expectancy>0, PROMISING>=60. "
                "Lookahead failure always REJECTS and losing test never gets PROMISING."
            ),
        )
        st.session_state["qc_strictness"] = strictness

        st.divider()
        api_key = st.text_input(
            "Anthropic API Key (session only)",
            type="password",
            value=st.session_state.get("qc_api_key", ""),
            help="Held in memory for this browser session only; never stored on disk.",
            key="sb_api_key",
        )
        st.session_state["qc_api_key"] = api_key
        model = st.text_input(
            "Claude Model",
            value=st.session_state.get("qc_model", os.getenv("QC_MODEL", DEFAULT_MODEL)),
            key="sb_model",
        )
        st.session_state["qc_model"] = model

        # Plain-text names only: no links until the companion repositories have real URLs
        st.markdown(
            '<div class="qc-footer">'
            "Crypto Quant Suite (planned companion projects):<br/>"
            "QuantCritic · PropPilot · SqueezeRadar · RiskGuard"
            "</div>",
            unsafe_allow_html=True,
        )

    config = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=float(risk_pct) / 100.0,
        max_leverage=float(max_lev),
        fee_rate=float(fee_bps) / 10_000.0,
        slippage_bps=float(slippage_bps),
        funding_enabled=bool(funding_on),
        interval=interval,  # type: ignore[arg-type]
    )
    return {
        "strategy_name": strategy_name,
        "symbols": list(symbols),
        "interval": interval,
        "params": params,
        "config": config,
        "strictness": strictness,
        "api_key": api_key,
        "model": model,
        "start_date": start_date,
        "end_date": end_date,
        "avail_min": avail_min,
        "avail_max": avail_max,
        "data_version": data_version,
        "date_error": date_error,
    }


def load_range_data(
    symbols: tuple[str, ...],
    interval: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    min_days: int | None = None,
    min_bars: int = 1,
    label: str = "Backtest",
) -> tuple[MarketData, MarketData, MarketData | None, pd.Timestamp | None]:
    """Load data for symbols and slice `[start_date, end_date]` while keeping pre-start warm-up."""
    full_data = cached_load_data(symbols, interval)
    history_to_end, window_data, warmup_data, trade_start = prepare_range_data(
        full_data, start=start_date, end=end_date
    )
    if min_days is not None:
        validate_date_range(
            window_data,
            start_ts=parse_utc_bound(start_date, is_end=False),
            end_ts=parse_utc_bound(end_date, is_end=True),
            min_days=min_days,
            min_bars=min_bars,
            label=label,
        )
    return history_to_end, window_data, warmup_data, trade_start


@st.cache_data(show_spinner=False)
def cached_load_data(symbols: tuple[str, ...], interval: str) -> MarketData:
    """Load and cache OHLCV + causal funding data."""
    return load_market_data(list(symbols), interval=interval)


@st.cache_data(show_spinner=False)
def cached_backtest(
    strategy_name: str,
    symbols: tuple[str, ...],
    interval: str,
    params_items: tuple[tuple[str, Any], ...],
    config_json: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BacktestResult:
    """Run and cache a single portfolio backtest (with indicator warm-up before start_date)."""
    history_to_end, _, _, trade_start = load_range_data(
        symbols, interval, start_date=start_date, end_date=end_date
    )
    strat = get_strategy(strategy_name)
    cfg = BacktestConfig.model_validate_json(config_json)
    if trade_start is not None:
        return run_window_backtest(history_to_end, strat, dict(params_items), cfg, trade_start)
    return run_backtest(history_to_end, strat, dict(params_items), cfg)


@st.cache_data(show_spinner=False)
def cached_offline_audit(
    strategy_name: str,
    symbols: tuple[str, ...],
    interval: str,
    config_json: str,
    start_date: str | None = None,
    end_date: str | None = None,
    strictness: str = DEFAULT_STRICTNESS,
) -> tuple[dict[str, Any], pd.Series, pd.Series, pd.DataFrame]:
    """Run and cache the complete offline robustness audit and report payload."""
    _, window_data, warmup_data, _ = load_range_data(
        symbols, interval, start_date=start_date, end_date=end_date
    )
    strat = get_strategy(strategy_name)
    cfg = BacktestConfig.model_validate_json(config_json)
    session = AuditSession(
        strategy=strat,
        data=window_data,
        config=cfg,
        audit_config=AuditConfig(strictness=strictness),  # type: ignore[arg-type]
        warmup_data=warmup_data,
    )
    run_offline_critic(session)
    payload = build_report_payload(session)
    full_res = session.cached_results["full_result"]
    wf_eq = session.cached_results["walkforward"].get("rolling_oos_equity", pd.Series(dtype=float))
    mc = session.cached_results["montecarlo"]
    payload["_bootstrap_samples"] = mc["bootstrap"]
    payload["_random_baseline_samples"] = mc["random_baseline"]
    return payload, full_res.equity_curve, wf_eq, full_res.trades


@st.cache_data(show_spinner=False)
def cached_leaderboard(
    symbols: tuple[str, ...],
    interval: str,
    config_json: str,
    include_leaky: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
    strictness: str = DEFAULT_STRICTNESS,
) -> list[dict[str, Any]]:
    """Audit all registered strategies (offline) and return sorted leaderboard rows."""
    rows: list[dict[str, Any]] = []
    for name in list_strategies(include_leaky=include_leaky):
        payload, _, _, _ = cached_offline_audit(
            name,
            symbols,
            interval,
            config_json,
            start_date=start_date,
            end_date=end_date,
            strictness=strictness,
        )
        rows.append(
            {
                "strategy": name,
                "verdict": payload["verdict"]["verdict"],
                "score": float(payload["verdict"]["score"]),
                "lookahead": payload["lookahead"]["status"].upper(),
                "val_sharpe": float(payload["walkforward"]["validation_metrics"]["sharpe"]),
                "test_sharpe": float(payload["final_test"]["test_metrics"]["sharpe"]),
                "dsr": float(payload["overfit"]["dsr"]["dsr"]),
                "pbo": float(payload["overfit"]["pbo"]["pbo"]),
                "p_value": float(payload["montecarlo"]["random_baseline"]["p_value"]),
                "break_even_bps": float(payload["costs"]["break_even_cost_bps"]),
                "full_return_pct": round(float(payload["full_metrics"]["total_return"]) * 100.0, 2),
                "max_dd_pct": round(float(payload["full_metrics"]["max_drawdown"]) * 100.0, 2),
                "summary": payload["verdict"]["summary"],
            }
        )
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    return rows


@st.cache_data(show_spinner=False)
def cached_lookahead_pair(
    honest_strategy: str,
    symbols: tuple[str, ...],
    interval: str,
    config_json: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Run raw backtests and truncation/delay tests for leaky_demo vs an honest strategy."""
    history_to_end, _, _, trade_start = load_range_data(
        symbols, interval, start_date=start_date, end_date=end_date
    )
    cfg = BacktestConfig.model_validate_json(config_json)
    out: dict[str, Any] = {}
    for name in ("leaky_demo", honest_strategy):
        strat = get_strategy(name)
        if trade_start is not None:
            res = run_window_backtest(history_to_end, strat, strat.default_params, cfg, trade_start)
        else:
            res = run_backtest(history_to_end, strat, strat.default_params, cfg)
        la = run_lookahead_audit(
            history_to_end,
            strat,
            strat.default_params,
            cfg,
            base_result=res,
            trade_start=trade_start,
        )
        out[name] = {
            "metrics": res.metrics,
            "equity_curve": res.equity_curve,
            "lookahead": la,
        }
    return out


def session_to_report_bytes(session: AuditSession) -> tuple[bytes, bytes]:
    """Render HTML and JSON report bytes from an AuditSession in memory."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        html_p, json_p = generate_audit_reports(session, output_dir=Path(tmp), basename="report")
        return html_p.read_bytes(), json_p.read_bytes()
