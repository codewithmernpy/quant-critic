"""Deterministic synthetic OHLCV and funding generators for testing."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from quantcritic.config import INTERVAL_TIMEDELTA_HOURS


def generate_random_walk_ohlcv(
    n_bars: int = 1000,
    start_price: float = 100.0,
    drift_per_bar: float = 0.0,
    vol_per_bar: float = 0.015,
    interval: str = "4h",
    start_time: str = "2022-01-01 00:00:00+00:00",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a geometric random-walk OHLCV DataFrame with valid high/low envelopes."""
    rng = np.random.default_rng(seed)
    hours = INTERVAL_TIMEDELTA_HOURS[interval]
    index = pd.date_range(start=start_time, periods=n_bars, freq=f"{hours}h", tz="UTC").as_unit("ns")

    log_rets = rng.normal(loc=drift_per_bar, scale=vol_per_bar, size=n_bars)
    close = start_price * np.exp(np.cumsum(log_rets))
    open_px = np.empty(n_bars, dtype=np.float64)
    open_px[0] = start_price
    open_px[1:] = close[:-1]

    wick_up = np.abs(rng.normal(0.0, vol_per_bar * 0.5, size=n_bars))
    wick_dn = np.abs(rng.normal(0.0, vol_per_bar * 0.5, size=n_bars))
    high = np.maximum(open_px, close) * (1.0 + wick_up)
    low = np.minimum(open_px, close) * np.maximum(1.0 - wick_dn, 0.01)

    volume = rng.lognormal(mean=7.0, sigma=0.6, size=n_bars)
    quote_volume = volume * close
    trades = rng.integers(500, 10_000, size=n_bars)
    taker_buy_base = volume * rng.uniform(0.35, 0.65, size=n_bars)

    return pd.DataFrame(
        {
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
            "trades": trades,
            "taker_buy_base": taker_buy_base,
            "funding_rate": np.zeros(n_bars, dtype=np.float64),
        },
        index=pd.DatetimeIndex(index, name="open_time"),
    )


def generate_funding_series(
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
    base_rate: float = 0.0001,
    vol: float = 0.0002,
    freq_hours: int = 8,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic 8h funding rate series with AR(1) persistence."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start_time, end=end_time, freq=f"{freq_hours}h", tz="UTC").as_unit("ns")
    n = len(idx)
    rates = np.empty(n, dtype=np.float64)
    val = base_rate
    for i in range(n):
        val = 0.7 * val + 0.3 * base_rate + rng.normal(0.0, vol)
        rates[i] = float(np.clip(val, -0.0075, 0.0075))
    return pd.Series(rates, index=pd.DatetimeIndex(idx, name="funding_time"), name="funding_rate")


def make_ohlcv_from_bars(
    bars: Sequence[tuple[float, float, float, float]],
    interval: str = "1h",
    start_time: str = "2024-01-01 00:00:00+00:00",
    volume: float = 1000.0,
) -> pd.DataFrame:
    """Build an OHLCV DataFrame from explicit (open, high, low, close) bar tuples for tests."""
    hours = INTERVAL_TIMEDELTA_HOURS[interval]
    index = pd.date_range(start=start_time, periods=len(bars), freq=f"{hours}h", tz="UTC").as_unit("ns")
    arr = np.asarray(bars, dtype=np.float64)
    return pd.DataFrame(
        {
            "open": arr[:, 0],
            "high": arr[:, 1],
            "low": arr[:, 2],
            "close": arr[:, 3],
            "volume": np.full(len(bars), volume, dtype=np.float64),
            "quote_volume": arr[:, 3] * volume,
            "trades": np.full(len(bars), 100, dtype=np.int64),
            "taker_buy_base": np.full(len(bars), volume * 0.5, dtype=np.float64),
            "funding_rate": np.zeros(len(bars), dtype=np.float64),
        },
        index=pd.DatetimeIndex(index, name="open_time"),
    )
