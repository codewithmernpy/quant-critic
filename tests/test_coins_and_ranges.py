"""Offline tests for custom coin downloads, sample+cache merge, date-range warm-up, and CLI."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantcritic.data.binance as binance_mod
import quantcritic.data.loader as loader_mod
from quantcritic.__main__ import main as cli_main
from quantcritic.agent.offline import run_offline_critic
from quantcritic.agent.tools import AuditSession
from quantcritic.audit.walkforward import run_window_backtest
from quantcritic.config import AuditConfig, BacktestConfig
from quantcritic.data.binance import (
    download_symbol_data,
    ensure_symbol_data,
    latest_complete_month,
    normalize_symbol,
    parse_symbols_input,
    update_symbol_to_latest,
)
from quantcritic.data.loader import (
    list_available_symbols,
    load_market_data,
    parse_utc_bound,
    prepare_range_data,
    validate_date_range,
)
from quantcritic.engine.backtest import run_backtest
from quantcritic.report.html import generate_audit_reports
from quantcritic.strategies.registry import get_strategy


def _make_monthly_kline_zip(year: int, month: int, seed: int = 7, base_price: float = 100.0) -> bytes:
    start = pd.Timestamp(f"{year:04d}-{month:02d}-01 00:00:00+00:00")
    next_m = start + pd.DateOffset(months=1)
    idx = pd.date_range(start, next_m - pd.Timedelta(hours=1), freq="1h", tz="UTC")
    rng = np.random.default_rng(seed + year * 100 + month)
    close = base_price * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))))
    open_p = np.roll(close, 1)
    open_p[0] = base_price
    high = np.maximum(open_p, close) * 1.005
    low = np.minimum(open_p, close) * 0.995
    ms = (idx.asi8 // 1_000_000).astype(np.int64)
    lines = [
        f"{ms[i]},{open_p[i]:.4f},{high[i]:.4f},{low[i]:.4f},{close[i]:.4f},1000.0,{ms[i] + 3599999},{close[i] * 1000:.2f},50,500.0,{close[i] * 500:.2f},0"
        for i in range(len(idx))
    ]
    csv_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"MOCK-{year}-{month:02d}.csv", csv_bytes)
    return buf.getvalue()


def _make_monthly_funding_zip(year: int, month: int, rate: float = 0.0001) -> bytes:
    start = pd.Timestamp(f"{year:04d}-{month:02d}-01 00:00:00+00:00")
    next_m = start + pd.DateOffset(months=1)
    idx = pd.date_range(start, next_m - pd.Timedelta(hours=8), freq="8h", tz="UTC")
    ms = (idx.asi8 // 1_000_000).astype(np.int64)
    lines = ["calc_time,funding_interval_hours,last_funding_rate"]
    lines.extend(f"{ms[i]},8,{rate}" for i in range(len(idx)))
    csv_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"MOCK-funding-{year}-{month:02d}.csv", csv_bytes)
    return buf.getvalue()


class _FakeHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _install_mock_binance(
    monkeypatch: pytest.MonkeyPatch,
    available: dict[str, set[tuple[int, int]]],
    calls: list[str] | None = None,
) -> list[str]:
    requested = calls if calls is not None else []

    def _fake_urlopen(req: urllib.request.Request | str, timeout: float = 30.0) -> _FakeHTTPResponse:
        url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
        requested.append(url)
        parts = url.rsplit("/", 1)[-1].replace(".zip", "").split("-")
        sym = parts[0]
        year, month = int(parts[-2]), int(parts[-1])
        months_for_sym = available.get(sym, set())
        if (year, month) not in months_for_sym:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        if "/fundingRate/" in url:
            return _FakeHTTPResponse(_make_monthly_funding_zip(year, month))
        return _FakeHTTPResponse(_make_monthly_kline_zip(year, month))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return requested


def test_symbol_normalization_and_validation() -> None:
    assert normalize_symbol("xrp") == "XRPUSDT"
    assert normalize_symbol("  dogeusdt ") == "DOGEUSDT"
    assert normalize_symbol("1000pepe") == "1000PEPEUSDT"

    valid, invalid = parse_symbols_input("xrp, DOGEUSDT, xrp, bad-coin, a, 1000SHIB")
    assert valid == ["XRPUSDT", "DOGEUSDT", "1000SHIBUSDT"]
    assert invalid == ["bad-coin", "a"]

    for bad in ("", "A", "USDT", "BTC-USDT", "$ETH", "X" * 21 + "USDT"):
        with pytest.raises(ValueError):
            normalize_symbol(bad)


def test_unknown_symbol_returns_clean_error_no_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_mock_binance(monkeypatch, available={})
    monkeypatch.setattr(binance_mod, "CACHE_DATA_DIR", tmp_path)
    monkeypatch.setattr(loader_mod, "CACHE_DATA_DIR", tmp_path)

    res = ensure_symbol_data("NOPEUSDT", start_month="2024-01", end_month="2024-03", output_dir=tmp_path)
    assert res["status"] == "not_found"
    assert res["saved"] == {}
    assert "404" in res["message"]
    assert len(calls) == 3  # Only kline months checked; funding skipped when no klines found
    assert not list(tmp_path.glob("NOPEUSDT*"))

    rc = cli_main(["backtest", "--strategy", "ema_trend", "--symbols", "NOPEUSDT", "--start", "2024-01-01", "--end", "2024-03-31"])
    assert rc == 1


def test_later_listed_coin_and_incremental_cache_update(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    avail = {"SUIUSDT": {(2024, 3), (2024, 4), (2024, 5)}}
    calls = _install_mock_binance(monkeypatch, available=avail)
    monkeypatch.setattr(binance_mod, "CACHE_DATA_DIR", tmp_path)
    monkeypatch.setattr(loader_mod, "CACHE_DATA_DIR", tmp_path)

    res = ensure_symbol_data("SUIUSDT", start_month="2024-01", end_month="2024-05", output_dir=tmp_path)
    assert res["listed_later"] is True
    assert res["first_available_month"] == "2024-03"
    assert res["last_available_month"] == "2024-05"
    assert res["downloaded_months"] == ["2024-03", "2024-04", "2024-05"]
    # 5 kline requests (Jan-May) + 3 funding requests (Mar-May, skipping pre-listing Jan-Feb)
    assert len(calls) == 8

    # Second call for the same range must hit zero HTTP endpoints (cached + pre-listing meta)
    calls.clear()
    res2 = ensure_symbol_data("SUIUSDT", start_month="2024-01", end_month="2024-05", output_dir=tmp_path)
    assert res2["status"] == "cached"
    assert res2["downloaded_months"] == []
    assert len(calls) == 0

    # Now publish June & July on mock Binance Vision and run incremental update
    avail["SUIUSDT"].update({(2024, 6), (2024, 7)})
    calls.clear()
    upd = update_symbol_to_latest("SUIUSDT", end_month="2024-07", output_dir=tmp_path)
    assert upd["downloaded_months"] == ["2024-06", "2024-07"]
    # Only June and July klines + funding requested (4 requests total)
    assert len(calls) == 4
    assert all("2024-06" in u or "2024-07" in u for u in calls)

    md = load_market_data(["SUIUSDT"], interval="1h")
    df = md.ohlcv["SUIUSDT"]
    assert df.index[0] == pd.Timestamp("2024-03-01 00:00:00+00:00")
    assert df.index[-1] == pd.Timestamp("2024-07-31 23:00:00+00:00")


def test_sample_and_cache_merge_no_duplicates_cache_wins_no_gap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample_dir = tmp_path / "sample"
    cache_dir = tmp_path / "cache"
    sample_dir.mkdir()
    cache_dir.mkdir()

    # Sample: Jan 1 .. Jan 10 (close = 100.0, funding = 0.0001)
    idx_s = pd.date_range("2024-01-01 00:00+00:00", "2024-01-10 23:00+00:00", freq="1h")
    df_s = pd.DataFrame(
        {
            "open_time": idx_s,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "trades": 5,
            "taker_buy_base": 5.0,
        }
    )
    sample_kline_p = sample_dir / "BTCUSDT_1h.csv.gz"
    df_s.to_csv(sample_kline_p, index=False, compression="gzip")
    sample_bytes_before = sample_kline_p.read_bytes()

    f_idx_s = pd.date_range("2024-01-01 00:00+00:00", "2024-01-10 16:00+00:00", freq="8h")
    pd.DataFrame({"funding_time": f_idx_s, "funding_rate": 0.0001}).to_csv(
        sample_dir / "BTCUSDT_funding.csv.gz", index=False, compression="gzip"
    )

    # Cache: Jan 8 .. Jan 20 (overlap Jan 8..10 with close = 200.0 and funding = 0.0009)
    idx_c = pd.date_range("2024-01-08 00:00+00:00", "2024-01-20 23:00+00:00", freq="1h")
    df_c = pd.DataFrame(
        {
            "open_time": idx_c,
            "open": 200.0,
            "high": 201.0,
            "low": 199.0,
            "close": 200.0,
            "volume": 20.0,
            "quote_volume": 4000.0,
            "trades": 8,
            "taker_buy_base": 10.0,
        }
    )
    df_c.to_csv(cache_dir / "BTCUSDT_1h.csv.gz", index=False, compression="gzip")

    f_idx_c = pd.date_range("2024-01-08 00:00+00:00", "2024-01-20 16:00+00:00", freq="8h")
    pd.DataFrame({"funding_time": f_idx_c, "funding_rate": 0.0009}).to_csv(
        cache_dir / "BTCUSDT_funding.csv.gz", index=False, compression="gzip"
    )

    # Also add an extra cached coin to test available symbol discovery
    df_c.to_csv(cache_dir / "XRPUSDT_1h.csv.gz", index=False, compression="gzip")

    monkeypatch.setattr(loader_mod, "SAMPLE_DATA_DIR", sample_dir)
    monkeypatch.setattr(loader_mod, "CACHE_DATA_DIR", cache_dir)

    assert list_available_symbols() == ["BTCUSDT", "XRPUSDT"]

    md_1h = load_market_data(["BTCUSDT"], interval="1h")
    df_1h = md_1h.ohlcv["BTCUSDT"]
    expected_idx = pd.date_range("2024-01-01 00:00+00:00", "2024-01-20 23:00+00:00", freq="1h")
    assert len(df_1h) == len(expected_idx) == 20 * 24
    assert not df_1h.index.duplicated().any()
    assert df_1h.index.is_monotonic_increasing
    assert (df_1h.index.diff().dropna() == pd.Timedelta(hours=1)).all()  # zero gaps

    # Cache wins on overlap (Jan 8..10)
    assert df_1h.loc[pd.Timestamp("2024-01-07 23:00+00:00"), "close"] == pytest.approx(100.0)
    assert df_1h.loc[pd.Timestamp("2024-01-08 00:00+00:00"), "close"] == pytest.approx(200.0)
    assert df_1h.loc[pd.Timestamp("2024-01-09 00:00+00:00"), "funding_rate"] == pytest.approx(0.0009)
    assert df_1h.loc[pd.Timestamp("2024-01-05 00:00+00:00"), "funding_rate"] == pytest.approx(0.0001)
    # Sample file was never modified
    assert sample_kline_p.read_bytes() == sample_bytes_before


def test_range_slicing_warmup_correctness_and_audit_splits_inside_range(tmp_path: Path) -> None:
    full_data = load_market_data(["BTCUSDT"], interval="4h")
    strat = get_strategy("rsi_reversion")  # 200-bar EMA warmup
    cfg = BacktestConfig(interval="4h")

    start_str, end_str = "2023-02-01", "2024-12-31"
    start_ts = parse_utc_bound(start_str, is_end=False)
    end_ts = parse_utc_bound(end_str, is_end=True)

    history_to_end, window_data, warmup_data, trade_start = prepare_range_data(
        full_data, start=start_str, end=end_str
    )
    assert warmup_data is not None and trade_start is not None

    win_res = run_window_backtest(history_to_end, strat, strat.default_params, cfg, trade_start)
    # Compare against generating signals on the full history up to end_ts and gating entries at trade_start
    full_sigs = {sym: strat.generate(df, strat.default_params) for sym, df in history_to_end.ohlcv.items()}
    ref_res = run_backtest(history_to_end, full_sigs, strat.default_params, cfg, trade_start=start_ts)

    pd.testing.assert_frame_equal(win_res.trades, ref_res.trades)
    pd.testing.assert_series_equal(win_res.equity_curve, ref_res.equity_curve)
    assert (pd.to_datetime(win_res.trades["entry_time"], utc=True) >= start_ts).all()
    assert (pd.to_datetime(win_res.trades["exit_time"], utc=True) <= end_ts).all()

    # Naive run without warm-up loses the 200-bar EMA early signals and produces different trades
    naive_res = run_backtest(window_data, strat, strat.default_params, cfg)
    assert len(naive_res.trades) < len(win_res.trades)
    assert not naive_res.trades["entry_time"].equals(win_res.trades["entry_time"])

    # Also verify exact match to a full-history run filtered to [start, end] when flat at start
    full_run = run_backtest(history_to_end, strat, strat.default_params, cfg)
    entry_t = pd.to_datetime(full_run.trades["entry_time"], utc=True)
    open_at_start = full_run.trades.loc[
        (entry_t < start_ts) & (pd.to_datetime(full_run.trades["exit_time"], utc=True) >= start_ts)
    ]
    assert open_at_start.empty
    filtered = full_run.trades.loc[entry_t >= start_ts].reset_index(drop=True)
    assert list(win_res.trades["entry_time"]) == list(filtered["entry_time"])
    assert list(win_res.trades["side"]) == list(filtered["side"])
    assert list(win_res.trades["entry_price"]) == pytest.approx(list(filtered["entry_price"]))

    # Minimum range checks
    short_end = "2023-02-15"
    _, short_win, _, _ = prepare_range_data(full_data, start=start_str, end=short_end)
    with pytest.raises(ValueError, match="at least 30 days"):
        validate_date_range(
            short_win,
            start_ts=parse_utc_bound(start_str, False),
            end_ts=parse_utc_bound(short_end, True),
            min_days=30,
            label="Backtest",
        )
    mid_end = "2023-06-01"
    _, mid_win, _, _ = prepare_range_data(full_data, start=start_str, end=mid_end)
    with pytest.raises(ValueError, match="at least 180 days"):
        validate_date_range(
            mid_win,
            start_ts=parse_utc_bound(start_str, False),
            end_ts=parse_utc_bound(mid_end, True),
            min_days=180,
            label="Audit",
        )

    # Audit splits strictly inside [2023-02-01, 2024-12-31] + reported in HTML/JSON
    session = AuditSession(
        strategy=get_strategy("ema_trend"),
        data=window_data,
        config=cfg,
        audit_config=AuditConfig(lookahead_cuts=5, mc_simulations=50, random_baseline_runs=20),
        warmup_data=warmup_data,
    )
    assert session.test_access_count == 0
    sp = session.splits
    assert sp.train_start >= start_ts
    assert sp.train_start < sp.train_end < sp.val_start < sp.val_end < sp.test_start < sp.test_end <= end_ts

    run_offline_critic(session)
    assert session.test_access_count == 1
    html_p, json_p = generate_audit_reports(session, output_dir=tmp_path, basename="range_audit")
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    assert payload["meta"]["date_range"]["start"] == "2023-02-01"
    assert payload["meta"]["date_range"]["end"] == "2024-12-31"
    assert payload["meta"]["splits"]["train_start"].startswith("2023-02-01")
    assert payload["meta"]["splits"]["test_end"].startswith("2024-12-31")
    html_text = html_p.read_text(encoding="utf-8")
    assert "2023-02-01 to 2024-12-31" in html_text
    assert str(sp.test_start)[:10] in html_text


def test_cli_range_auto_download_and_no_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache"
    reports_dir = tmp_path / "reports"
    cache_dir.mkdir()
    monkeypatch.setattr(binance_mod, "CACHE_DATA_DIR", cache_dir)
    monkeypatch.setattr(loader_mod, "CACHE_DATA_DIR", cache_dir)

    avail = {"NEWUSDT": {(2024, m) for m in range(1, 8)}}
    calls = _install_mock_binance(monkeypatch, available=avail)

    # --no-download forbids network for missing symbol
    rc_nodl = cli_main([
        "backtest", "--strategy", "ema_trend", "--symbols", "new",
        "--interval", "4h", "--start", "2024-01-01", "--end", "2024-06-30", "--no-download",
    ])
    assert rc_nodl == 1
    assert len(calls) == 0

    # Auto-download missing coin with normalized symbol + date range
    rc_bt = cli_main([
        "backtest", "--strategy", "ema_trend", "--symbols", "new",
        "--interval", "4h", "--start", "2024-02-01", "--end", "2024-06-30",
    ])
    assert rc_bt == 0
    out_bt = capsys.readouterr().out
    assert "NEWUSDT" in out_bt
    assert "2024-02-01 to 2024-06-30" in out_bt
    assert len(calls) > 0

    # Re-run with --no-download now succeeds from cache with 0 network calls
    calls.clear()
    rc_cached = cli_main([
        "backtest", "--strategy", "ema_trend", "--symbols", "NEWUSDT",
        "--interval", "4h", "--start", "2024-02-01", "--end", "2024-06-30", "--no-download",
    ])
    assert rc_cached == 0
    assert len(calls) == 0

    # Too-short range fails cleanly
    rc_short = cli_main([
        "backtest", "--strategy", "ema_trend", "--symbols", "BTCUSDT",
        "--interval", "4h", "--start", "2024-01-01", "--end", "2024-01-15", "--no-download",
    ])
    assert rc_short == 1

    # Audit with --start/--end on bundled sample reports split dates
    rc_au = cli_main([
        "audit", "--strategy", "donchian_breakout", "--symbols", "BTCUSDT",
        "--interval", "1d", "--start", "2022-01-01", "--end", "2024-12-31",
        "--no-download", "--output-dir", str(reports_dir),
    ])
    assert rc_au == 0
    out_au = capsys.readouterr().out
    assert "Range        : 2022-01-01 to 2024-12-31" in out_au
    assert "Locked Test" in out_au

    # Compare with --start/--end
    rc_cmp = cli_main([
        "compare", "--symbols", "BTCUSDT", "--interval", "1d",
        "--start", "2023-01-01", "--end", "2024-12-31",
        "--no-download", "--exclude-leaky", "--output-dir", str(reports_dir),
    ])
    assert rc_cmp == 0
    assert (reports_dir / "leaderboard.html").is_file()

    # Download subcommand keeps working
    dl_dir = tmp_path / "dl"
    rc_dl = cli_main([
        "download", "--symbols", "NEWUSDT", "--interval", "1h",
        "--start", "2024-01", "--end", "2024-02", "--output-dir", str(dl_dir),
    ])
    assert rc_dl == 0
    assert (dl_dir / "NEWUSDT_1h.csv.gz").is_file()
    assert latest_complete_month(pd.Timestamp("2026-11-25").date()) == "2026-10"
