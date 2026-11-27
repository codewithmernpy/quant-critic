"""Tests for data loading, resampling, synthetic generation, and Binance Vision parsing."""

from __future__ import annotations

import pandas as pd
import pytest

from quantcritic.data.binance import parse_binance_funding_csv, parse_binance_kline_csv
from quantcritic.data.loader import load_funding_rates, load_klines, load_market_data, resample_ohlcv
from quantcritic.data.synthetic import generate_funding_series, generate_random_walk_ohlcv


def test_load_sample_klines_and_funding() -> None:
    df = load_klines("BTCUSDT", interval="1h")
    assert not df.empty
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)

    funding = load_funding_rates("BTCUSDT")
    assert not funding.empty
    assert str(funding.index.tz) == "UTC"
    # Verify jitter was rounded to nearest minute
    assert (funding.index.second == 0).all()
    assert (funding.index.microsecond == 0).all()


def test_resample_4h_and_1d_ohlc_and_gap_handling() -> None:
    df_1h = load_klines("SOLUSDT", interval="1h")
    df_4h = resample_ohlcv(df_1h, "4h")
    df_1d = resample_ohlcv(df_1h, "1d")

    assert not df_4h.isna().any().any()
    assert not df_1d.isna().any().any()
    # Period-open labels: 4h timestamps fall on multiples of 4 hours, 1d on midnight UTC
    assert (df_4h.index.hour % 4 == 0).all()
    assert (df_1d.index.hour == 0).all()

    first_4h_ts = df_4h.index[0]
    chunk = df_1h.loc[first_4h_ts : first_4h_ts + pd.Timedelta(hours=3)]
    assert df_4h.loc[first_4h_ts, "open"] == pytest.approx(chunk["open"].iloc[0])
    assert df_4h.loc[first_4h_ts, "high"] == pytest.approx(chunk["high"].max())
    assert df_4h.loc[first_4h_ts, "low"] == pytest.approx(chunk["low"].min())
    assert df_4h.loc[first_4h_ts, "close"] == pytest.approx(chunk["close"].iloc[-1])
    assert df_4h.loc[first_4h_ts, "volume"] == pytest.approx(chunk["volume"].sum())


def test_parse_binance_kline_csv_ms_us_header_and_no_header() -> None:
    # No-header milliseconds
    no_hdr_ms = (
        b"1609459200000,29000.0,29100.0,28900.0,29050.0,100.0,1609462799999,2905000.0,500,50.0,1450000.0,0\n"
        b"1609462800000,29050.0,29200.0,29000.0,29150.0,120.0,1609466399999,3490000.0,600,60.0,1740000.0,0\n"
    )
    df_ms = parse_binance_kline_csv(no_hdr_ms)
    assert len(df_ms) == 2
    assert pd.Timestamp(df_ms["open_time"].iloc[0]) == pd.Timestamp("2021-01-01 00:00:00+00:00")

    # Header + microseconds (2025+ Binance Vision format)
    hdr_us = (
        b"open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        b"1735689600000000,93000.0,93500.0,92800.0,93200.0,55.0,1735693199999999,5126000.0,400,28.0,2609600.0,0\n"
    )
    df_us = parse_binance_kline_csv(hdr_us)
    assert len(df_us) == 1
    assert pd.Timestamp(df_us["open_time"].iloc[0]) == pd.Timestamp("2025-01-01 00:00:00+00:00")
    assert df_us["trades"].iloc[0] == 400


def test_parse_binance_funding_csv_header_and_no_header() -> None:
    hdr_csv = b"calc_time,funding_interval_hours,last_funding_rate\n1704067200005,8,0.00015\n"
    f1 = parse_binance_funding_csv(hdr_csv)
    assert pd.Timestamp(f1["funding_time"].iloc[0]) == pd.Timestamp("2024-01-01 00:00:00+00:00")
    assert f1["funding_rate"].iloc[0] == pytest.approx(0.00015)

    no_hdr_us = b"1704096000004000,8,-0.0002\n"
    f2 = parse_binance_funding_csv(no_hdr_us)
    assert pd.Timestamp(f2["funding_time"].iloc[0]) == pd.Timestamp("2024-01-01 08:00:00+00:00")
    assert f2["funding_rate"].iloc[0] == pytest.approx(-0.0002)


def test_synthetic_ohlcv_invariants_and_causal_funding_attachment() -> None:
    df = generate_random_walk_ohlcv(n_bars=200, interval="4h", seed=7)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    f = generate_funding_series(df.index[0], df.index[-1], seed=7)
    assert not f.empty

    md = load_market_data(["BTCUSDT"], interval="4h", start="2024-01-01", end="2024-03-01")
    assert "funding_rate" in md.ohlcv["BTCUSDT"].columns
