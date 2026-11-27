"""Streamlit AppTest verification across the entry page and all 6 multipage views.

Each test drives the real widgets (sidebar selectors, buttons, checkboxes) and checks that real
content renders - numbers that match a direct backtest, verdict badges, gates, download buttons,
leaderboard rows, and the leaky-vs-honest lookahead comparison.
"""

from __future__ import annotations

from datetime import date
import io
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.testing.v1 import AppTest
import tomllib

import quantcritic.data.binance as binance_mod
import quantcritic.data.loader as loader_mod
import quantcritic.strategies.builder as builder_mod
import quantcritic.strategies.registry as registry_mod
from quantcritic.config import BacktestConfig
from quantcritic.data.loader import load_market_data
from quantcritic.engine.backtest import run_backtest
from quantcritic.strategies.registry import get_strategy, list_strategies, reload_custom_strategies

ROOT = Path(__file__).resolve().parent.parent


def _narrow_sidebar_for_fast_test(at: AppTest, interval: str = "1d") -> None:
    """Narrow sidebar selection to 1 symbol (and optionally 1d) so AppTest finishes quickly."""
    at.sidebar.multiselect[0].set_value(["BTCUSDT"])
    at.sidebar.selectbox[1].set_value(interval)


def test_home_page_renders() -> None:
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("QuantCritic" in t.value for t in at.title)
    assert len(at.dataframe) >= 2
    sample_df = at.dataframe[0].value
    assert len(sample_df) == 3  # BTC, ETH, SOL loaded
    btc_close = sample_df.set_index("Symbol").loc["BTCUSDT", "Last Close ($)"]
    assert isinstance(btc_close, str) and "," in btc_close and btc_close == "109,557.30"
    assert set(at.dataframe[1].value["Strategy"]) == set(list_strategies(include_leaky=True))
    assert len(at.sidebar.text_input) >= 2
    footer = [m.value for m in at.sidebar.markdown if "qc-footer" in m.value][0]
    assert "href" not in footer and "RiskGuard" in footer  # plain-text names, no placeholder links


def test_page1_backtest_button_and_outputs() -> None:
    at = AppTest.from_file(str(ROOT / "pages/1_Backtest.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.sidebar.selectbox[0].set_value("ema_trend")
    _narrow_sidebar_for_fast_test(at, interval="4h")
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) >= 6
    assert len(at.dataframe) >= 1
    # Numbers on the page equal a direct engine run with the same (default) settings
    data = load_market_data(["BTCUSDT"], interval="4h")
    strat = get_strategy("ema_trend")
    direct = run_backtest(data, strat, dict(strat.default_params), BacktestConfig(interval="4h")).metrics
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Trades"] == str(direct["n_trades"])
    assert metrics["Total Return"] == f"{direct['total_return'] * 100:+.1f}%"
    assert len(at.dataframe[0].value) == direct["n_trades"]


def test_page2_audit_button_and_downloads() -> None:
    at = AppTest.from_file(str(ROOT / "pages/2_Audit.py"), default_timeout=120)
    at.run()
    assert not at.exception
    _narrow_sidebar_for_fast_test(at, interval="1d")
    at.button[0].click().run()
    assert not at.exception
    verdict_md = [m.value for m in at.markdown if "Verdict" in m.value]
    assert verdict_md and any(v in verdict_md[0] for v in ("PROMISING", "NEEDS WORK", "REJECT"))
    # donchian 1d loses on the locked test -> gate shown and verdict is not PROMISING
    assert "PROMISING" not in verdict_md[0]
    assert any("test_fail" in w.value for w in at.warning)
    assert [d.label for d in at.get("download_button")] == ["Download HTML Report", "Download JSON Report"]
    assert len(at.tabs) == 6
    badges = [m.value for m in at.markdown if "qc-badge qc-" in m.value]
    assert any("qc-pass" in b or "qc-warn" in b or "qc-fail" in b for b in badges)


def test_page3_ai_critic_offline_execution() -> None:
    at = AppTest.from_file(str(ROOT / "pages/3_AI_Critic.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("offline critic" in i.value for i in at.info)  # no key -> offline path announced
    _narrow_sidebar_for_fast_test(at, interval="1d")
    at.button[0].click().run()
    assert not at.exception
    assert any("Critic Verdict" in m.value and "OFFLINE" in m.value for m in at.markdown)
    log = at.dataframe[0].value
    tools = list(log["tool"])
    assert {"walk_forward", "lookahead_test", "final_test", "submit_verdict"} <= set(tools)
    assert tools.count("final_test") == 1
    assert tools.count("walk_forward") == 1
    assert len(tools) == len(set(tools))
    assert all(a != b for a, b in zip(tools, tools[1:]))


def test_page4_leaderboard_renders() -> None:
    at = AppTest.from_file(str(ROOT / "pages/4_Leaderboard.py"), default_timeout=300)
    at.run()
    assert not at.exception
    _narrow_sidebar_for_fast_test(at, interval="1d")
    at.button[0].click().run()
    assert not at.exception
    df = at.dataframe[0].value
    assert set(df["strategy"]) == set(list_strategies(include_leaky=True))
    assert df.set_index("strategy").loc["leaky_demo", "verdict"] == "REJECT"
    assert list(df["score"]) == sorted(df["score"], reverse=True)
    next(c for c in at.checkbox if c.label.startswith("Include leaky_demo")).uncheck().run()
    at.button[0].click().run()
    assert not at.exception
    assert "leaky_demo" not in set(at.dataframe[0].value["strategy"])


def test_page5_lookahead_lab_catches_leaky_demo() -> None:
    at = AppTest.from_file(str(ROOT / "pages/5_Lookahead_Lab.py"), default_timeout=120)
    at.run()
    assert not at.exception
    _narrow_sidebar_for_fast_test(at, interval="1d")
    next(sb for sb in at.selectbox if sb.label == "Honest Comparison Strategy").set_value("ema_trend")
    at.button[0].click().run()
    assert not at.exception
    assert any("FAIL" in m.value for m in at.markdown)
    assert any("PASS" in m.value for m in at.markdown)
    assert any("ema_trend" in s.value for s in at.subheader)
    assert len(at.error) >= 1
    assert len(at.success) >= 1


def test_page8_about_methodology_renders() -> None:
    at = AppTest.from_file(str(ROOT / "pages/8_About_Methodology.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("About & Methodology" in t.value for t in at.title)
    assert any("Strictness" in m.value for m in at.markdown)


def _mock_zip_kline_month(year: int, month: int, seed: int = 19) -> bytes:
    start = pd.Timestamp(f"{year:04d}-{month:02d}-01 00:00:00+00:00")
    next_m = start + pd.DateOffset(months=1)
    idx = pd.date_range(start, next_m - pd.Timedelta(hours=1), freq="1h", tz="UTC")
    rng = np.random.default_rng(seed + year * 100 + month)
    close = 0.5 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, len(idx))))
    open_p = np.roll(close, 1)
    open_p[0] = 0.5
    high = np.maximum(open_p, close) * 1.008
    low = np.minimum(open_p, close) * 0.992
    ms = (idx.asi8 // 1_000_000).astype(np.int64)
    lines = [
        f"{ms[i]},{open_p[i]:.5f},{high[i]:.5f},{low[i]:.5f},{close[i]:.5f},10000.0,{ms[i] + 3599999},{close[i] * 10000:.2f},80,5000.0,{close[i] * 5000:.2f},0"
        for i in range(len(idx))
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"XRPUSDT-1h-{year}-{month:02d}.csv", ("\n".join(lines) + "\n").encode("utf-8"))
    return buf.getvalue()


def _mock_zip_funding_month(year: int, month: int) -> bytes:
    start = pd.Timestamp(f"{year:04d}-{month:02d}-01 00:00:00+00:00")
    next_m = start + pd.DateOffset(months=1)
    idx = pd.date_range(start, next_m - pd.Timedelta(hours=8), freq="8h", tz="UTC")
    ms = (idx.asi8 // 1_000_000).astype(np.int64)
    lines = ["calc_time,funding_interval_hours,last_funding_rate"]
    lines.extend(f"{ms[i]},8,0.0001" for i in range(len(idx)))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"XRPUSDT-fundingRate-{year}-{month:02d}.csv", ("\n".join(lines) + "\n").encode("utf-8"))
    return buf.getvalue()


def test_apptest_add_mocked_coin_date_range_backtest_and_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(binance_mod, "CACHE_DATA_DIR", cache_dir)
    monkeypatch.setattr(loader_mod, "CACHE_DATA_DIR", cache_dir)
    monkeypatch.setattr(binance_mod, "latest_complete_month", lambda reference_date=None: "2024-12")

    avail_months = {(2024, m) for m in range(1, 13)}

    class _Resp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def _fake_urlopen(req: urllib.request.Request | str, timeout: float = 30.0) -> _Resp:
        url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
        parts = url.rsplit("/", 1)[-1].replace(".zip", "").split("-")
        sym, year, month = parts[0], int(parts[-2]), int(parts[-1])
        if sym != "XRPUSDT" or (year, month) not in avail_months:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        if "/fundingRate/" in url:
            return _Resp(_mock_zip_funding_month(year, month))
        return _Resp(_mock_zip_kline_month(year, month))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    st.cache_data.clear()
    try:
        # 1. Page 1 Backtest: add mocked coin via text input, check unknown coin error, set range, run
        at1 = AppTest.from_file(str(ROOT / "pages/1_Backtest.py"), default_timeout=120)
        at1.run()
        assert not at1.exception

        # Unknown coin yields clean error and does not crash
        at1.sidebar.text_input(key="sb_add_coins").set_value("NOPEUSDT").run()
        assert not at1.exception
        assert any("NOPEUSDT" in e.value for e in at1.sidebar.error)

        # Add mocked coin using lowercase ticker -> normalized to XRPUSDT
        at1.sidebar.text_input(key="sb_add_coins").set_value("xrp").run()
        assert not at1.exception
        assert "XRPUSDT" in at1.sidebar.multiselect[0].value
        assert "XRPUSDT" in at1.sidebar.multiselect[0].options
        assert (cache_dir / "XRPUSDT_1h.csv.gz").is_file()

        # Select only XRPUSDT + ema_trend + 4h + custom range [2024-02-01, 2024-11-30]
        at1.sidebar.multiselect[0].set_value(["XRPUSDT"])
        at1.sidebar.selectbox[0].set_value("ema_trend")
        at1.sidebar.selectbox[1].set_value("4h")
        at1.run()
        at1.sidebar.date_input(key="sb_start_date").set_value(date(2024, 2, 1))
        at1.sidebar.date_input(key="sb_end_date").set_value(date(2024, 11, 30))
        at1.button[0].click().run()
        assert not at1.exception
        assert any("2024-02-01" in c.value and "2024-11-30" in c.value for c in at1.caption)
        assert len(at1.metric) >= 6
        assert len(at1.dataframe) >= 1
        trades_df = at1.dataframe[0].value
        assert not trades_df.empty
        entry_times = pd.to_datetime(trades_df["entry_time"], utc=True)
        assert (entry_times >= pd.Timestamp("2024-02-01 00:00+00:00")).all()
        assert (entry_times <= pd.Timestamp("2024-11-30 23:59:59+00:00")).all()

        # 2. Page 2 Audit: select cached XRPUSDT, set range, run Full Audit, assert range & split dates shown
        at2 = AppTest.from_file(str(ROOT / "pages/2_Audit.py"), default_timeout=180)
        at2.run()
        assert not at2.exception
        at2.sidebar.multiselect[0].set_value(["XRPUSDT"])
        at2.sidebar.selectbox[0].set_value("ema_trend")
        at2.sidebar.selectbox[1].set_value("4h")
        at2.run()
        at2.sidebar.date_input(key="sb_start_date").set_value(date(2024, 2, 1))
        at2.sidebar.date_input(key="sb_end_date").set_value(date(2024, 11, 30))
        at2.button[0].click().run()
        assert not at2.exception
        assert any("Effective Range" in c.value and "2024-02-01" in c.value and "2024-11-30" in c.value for c in at2.caption)
        assert any("Locked Test (20%)" in c.value for c in at2.caption)
        verdict_md = [m.value for m in at2.markdown if "Verdict" in m.value]
        assert verdict_md and any(v in verdict_md[0] for v in ("PROMISING", "NEEDS WORK", "REJECT"))
        wf_table = at2.dataframe[0].value
        assert "Window (UTC)" in wf_table.columns
    finally:
        st.cache_data.clear()


def test_apptest_builder_save_and_run_in_backtest_and_switch_strictness_on_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    user_dir = tmp_path / "user_strategies"
    user_dir.mkdir()
    monkeypatch.setattr(builder_mod, "USER_STRATEGIES_DIR", user_dir)
    monkeypatch.setattr(registry_mod, "USER_STRATEGIES_DIR", user_dir)
    reload_custom_strategies()
    st.cache_data.clear()
    try:
        # 1. Open Strategy Builder page, test validation error on bad ATR stop, then fix & save
        at_b = AppTest.from_file(str(ROOT / "pages/7_Strategy_Builder.py"), default_timeout=120)
        at_b.run()
        assert not at_b.exception
        _narrow_sidebar_for_fast_test(at_b, interval="4h")
        at_b.run()

        # Bad ATR stop multiplier (0.0) triggers clear validation error, no crash
        at_b.number_input(key="builder_atr_stop_mult").set_value(0.0).run()
        assert not at_b.exception
        assert any("validation" in e.value.lower() or "greater than 0" in e.value.lower() for e in at_b.error)

        # Fix ATR stop multiplier, give custom name, verify live preview, click Save
        at_b.number_input(key="builder_atr_stop_mult").set_value(2.0)
        at_b.text_input(key="builder_name").set_value("apptest_pullback")
        at_b.run()
        assert not at_b.exception
        assert any(m.label == "Total Entry Signals" for m in at_b.metric)

        save_btn = next(b for b in at_b.button if b.label == "Save Strategy")
        save_btn.click().run()
        assert not at_b.exception
        assert (user_dir / "apptest_pullback.json").is_file()
        assert "custom:apptest_pullback" in list_strategies()

        # 2. Open Page 1 Backtest: verify custom:apptest_pullback is in selector and run it
        at1 = AppTest.from_file(str(ROOT / "pages/1_Backtest.py"), default_timeout=120)
        at1.run()
        assert not at1.exception
        assert "custom:apptest_pullback" in at1.sidebar.selectbox(key="sb_strategy").options
        at1.sidebar.selectbox(key="sb_strategy").set_value("custom:apptest_pullback")
        _narrow_sidebar_for_fast_test(at1, interval="4h")
        at1.button[0].click().run()
        assert not at1.exception
        assert len(at1.metric) >= 6
        assert any(m.label == "Trades" and int(m.value) > 0 for m in at1.metric)

        # 3. Open Page 2 Audit: run in default balanced mode, then switch strictness in sidebar
        at2 = AppTest.from_file(str(ROOT / "pages/2_Audit.py"), default_timeout=120)
        at2.run()
        assert not at2.exception
        at2.sidebar.selectbox(key="sb_strategy").set_value("donchian_breakout")
        _narrow_sidebar_for_fast_test(at2, interval="1d")
        assert at2.sidebar.selectbox(key="sb_strictness").value == "balanced"
        at2.button[0].click().run()
        assert not at2.exception
        assert any("Verdict mode: balanced" in m.value for m in at2.markdown)

        at2.sidebar.selectbox(key="sb_strictness").set_value("strict").run()
        assert not at2.exception
        assert any("Verdict mode: strict" in m.value for m in at2.markdown)

        at2.sidebar.selectbox(key="sb_strictness").set_value("relaxed").run()
        assert not at2.exception
        assert any("Verdict mode: relaxed" in m.value for m in at2.markdown)
    finally:
        monkeypatch.undo()
        reload_custom_strategies()
        st.cache_data.clear()


def test_entrypoint_navigation_and_titles() -> None:
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["client"]["toolbarMode"] == "viewer"

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "QuantCritic"

    expected_titles = [
        "Home",
        "Backtest",
        "Audit",
        "AI Critic",
        "Leaderboard",
        "Lookahead Lab",
        "Strategy Builder",
        "About & Methodology",
    ]
    assert [p["page_name"] for p in at._registered_pages.values()] == expected_titles
    assert at._registered_pages[at._finished_page_script_hash]["page_name"] == "Home"

    def _has_metric_css(app: AppTest) -> bool:
        return any(
            '[data-testid="stMetricLabel"]' in m.value
            and "white-space: normal !important" in m.value
            and '[data-testid="stMetricValue"]' in m.value
            and "clamp(1.1rem, 2.2vw, 1.8rem)" in m.value
            for m in app.markdown
        )

    assert _has_metric_css(at)

    for page_path in (
        "pages/1_Backtest.py",
        "pages/2_Audit.py",
        "pages/3_AI_Critic.py",
        "pages/4_Leaderboard.py",
        "pages/5_Lookahead_Lab.py",
        "pages/7_Strategy_Builder.py",
        "pages/8_About_Methodology.py",
    ):
        at.switch_page(page_path).run()
        assert not at.exception
        assert _has_metric_css(at), f"Missing metric label CSS on {page_path}"


def test_backtest_metric_delta_colors() -> None:
    at = AppTest.from_file(str(ROOT / "pages/1_Backtest.py"), default_timeout=60).run()
    assert not at.exception
    at.sidebar.selectbox[0].set_value("ema_trend")
    _narrow_sidebar_for_fast_test(at, interval="4h")
    at.button[0].click().run()
    assert not at.exception

    by_label = {m.label: m for m in at.metric}
    for label in ("Sharpe Ratio", "Max Drawdown", "Trades", "Expectancy"):
        assert by_label[label].proto.color == MetricProto.MetricColor.GRAY
        assert by_label[label].proto.direction == MetricProto.MetricDirection.NONE

    assert by_label["Total Return"].proto.color == MetricProto.MetricColor.GREEN
    assert by_label["Total Return"].proto.direction == MetricProto.MetricDirection.UP
    # Funding for ema_trend on BTCUSDT 4h is negative -> never green/up
    assert by_label["Fees & Funding"].proto.color != MetricProto.MetricColor.GREEN
    assert by_label["Fees & Funding"].proto.direction != MetricProto.MetricDirection.UP


def test_builder_switch_page_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    user_dir = tmp_path / "user_strategies"
    user_dir.mkdir()
    monkeypatch.setattr(builder_mod, "USER_STRATEGIES_DIR", user_dir)
    monkeypatch.setattr(registry_mod, "USER_STRATEGIES_DIR", user_dir)
    reload_custom_strategies()
    st.cache_data.clear()
    try:
        at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180).run()
        assert not at.exception

        at.switch_page("pages/7_Strategy_Builder.py").run()
        assert not at.exception
        _narrow_sidebar_for_fast_test(at, interval="1d")
        at.text_input(key="builder_name").set_value("switch_test")
        at.run()
        next(b for b in at.button if b.label == "Save & Open Backtest").click().run()
        assert not at.exception
        assert at._registered_pages[at._finished_page_script_hash]["page_name"] == "Backtest"

        at.switch_page("pages/7_Strategy_Builder.py").run()
        assert not at.exception
        next(b for b in at.button if b.label == "Save & Open Audit").click().run()
        assert not at.exception
        assert at._registered_pages[at._finished_page_script_hash]["page_name"] == "Audit"
    finally:
        monkeypatch.undo()
        reload_custom_strategies()
        st.cache_data.clear()
