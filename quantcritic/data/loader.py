"""Market data loader and resampler for Binance USDT-M perpetuals."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quantcritic.config import (
    CACHE_DATA_DIR,
    DEFAULT_SYMBOLS,
    INTERVAL_TIMEDELTA_HOURS,
    PANDAS_RESAMPLE_MAP,
    SAMPLE_DATA_DIR,
    SUPPORTED_INTERVALS,
)

logger = logging.getLogger(__name__)

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def to_utc_ns_index(obj: Any) -> pd.DatetimeIndex:
    """Normalize any datetime-like index or array to nanosecond-resolution UTC DatetimeIndex."""
    dt = pd.DatetimeIndex(pd.to_datetime(obj, utc=True, format="ISO8601"))
    return dt.as_unit("ns")


def parse_utc_bound(
    value: str | date | datetime | pd.Timestamp | None,
    is_end: bool = False,
) -> pd.Timestamp | None:
    """Parse a date/timestamp bound into UTC nanosecond Timestamp.

    Pure dates (`datetime.date` or `YYYY-MM-DD` strings) with `is_end=True` expand to the
    inclusive end of that UTC day (`23:59:59.999999999`).
    """
    if value is None:
        return None
    is_pure_date = (isinstance(value, date) and not isinstance(value, (datetime, pd.Timestamp))) or (
        isinstance(value, str) and bool(_DATE_ONLY_RE.match(value.strip()))
    )
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if is_pure_date and is_end:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts.as_unit("ns")


def list_available_symbols(
    sample_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> list[str]:
    """Return locally available symbols across `data/sample/` and `data/cache/`."""
    s_dir = sample_dir or SAMPLE_DATA_DIR
    c_dir = cache_dir or CACHE_DATA_DIR
    found: set[str] = set()
    suffixes = tuple(f"_{iv}.csv.gz" for iv in SUPPORTED_INTERVALS)
    for root in (s_dir, c_dir):
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if not p.is_file():
                continue
            name = p.name
            for suf in suffixes:
                if name.endswith(suf):
                    sym = name[: -len(suf)].upper()
                    if sym:
                        found.add(sym)
                    break

    ordered = [s for s in DEFAULT_SYMBOLS if s in found]
    extras = sorted(s for s in found if s not in DEFAULT_SYMBOLS)
    return ordered + extras


@dataclass
class MarketData:
    """Container holding aligned OHLCV DataFrames and raw funding Series per symbol."""

    ohlcv: dict[str, pd.DataFrame]
    funding: dict[str, pd.Series]
    interval: str

    @property
    def symbols(self) -> list[str]:
        return list(self.ohlcv.keys())

    def slice_index_bounds(
        self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> "MarketData":
        """Return a new MarketData sliced inclusively by UTC timestamp bounds."""
        sliced_ohlcv: dict[str, pd.DataFrame] = {}
        sliced_funding: dict[str, pd.Series] = {}
        for sym, df in self.ohlcv.items():
            mask = pd.Series(True, index=df.index)
            if start is not None:
                mask &= df.index >= start
            if end is not None:
                mask &= df.index <= end
            sliced_ohlcv[sym] = df.loc[mask].copy()

            f_ser = self.funding.get(sym, pd.Series(dtype=float))
            if not f_ser.empty:
                f_mask = pd.Series(True, index=f_ser.index)
                if start is not None:
                    f_mask &= f_ser.index >= start
                if end is not None:
                    # Include funding at end_bar close (end + bar_delta) for positions open on last bar
                    bar_delta = pd.Timedelta(hours=INTERVAL_TIMEDELTA_HOURS[self.interval])
                    f_mask &= f_ser.index <= end + bar_delta
                sliced_funding[sym] = f_ser.loc[f_mask].copy()
            else:
                sliced_funding[sym] = f_ser.copy()
        return MarketData(ohlcv=sliced_ohlcv, funding=sliced_funding, interval=self.interval)


def _resolve_kline_sources(
    symbol: str, interval: str, data_dir: Path | None = None
) -> list[tuple[Path, str]]:
    """Locate kline files in precedence order (sample first, cache second so cache wins)."""
    sym = symbol.upper()
    if data_dir is not None:
        exact = data_dir / f"{sym}_{interval}.csv.gz"
        if exact.is_file():
            return [(exact, interval)]
        base_1h = data_dir / f"{sym}_1h.csv.gz"
        if base_1h.is_file():
            return [(base_1h, "1h")]
        raise FileNotFoundError(f"No kline file found for {sym} in {data_dir}")

    sources: list[tuple[Path, str]] = []
    for root in (SAMPLE_DATA_DIR, CACHE_DATA_DIR):
        exact = root / f"{sym}_{interval}.csv.gz"
        base_1h = root / f"{sym}_1h.csv.gz"
        if exact.is_file():
            sources.append((exact, interval))
        elif base_1h.is_file():
            sources.append((base_1h, "1h"))

    if not sources:
        raise FileNotFoundError(
            f"No data file found for {sym} ({interval}) in {SAMPLE_DATA_DIR} or {CACHE_DATA_DIR}"
        )
    return sources


def _resolve_funding_paths(symbol: str, data_dir: Path | None = None) -> list[Path]:
    """Locate funding CSV files in precedence order (sample first, cache second)."""
    sym = symbol.upper()
    if data_dir is not None:
        path = data_dir / f"{sym}_funding.csv.gz"
        return [path] if path.is_file() else []
    return [
        p
        for root in (SAMPLE_DATA_DIR, CACHE_DATA_DIR)
        if (p := root / f"{sym}_funding.csv.gz").is_file()
    ]


def _read_kline_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df["open_time"] = to_utc_ns_index(df["open_time"])
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    df = df.set_index("open_time")
    df.index = to_utc_ns_index(df.index)

    numeric_cols = [
        c for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base")
        if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample 1h OHLCV klines to 1h / 4h / 1d with period-open labels.

    Empty bins (e.g. exchange downtime gaps) are dropped rather than filled with NaNs.
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval {interval!r}; expected one of {SUPPORTED_INTERVALS}")
    if interval == "1h":
        return df.copy()

    rule = PANDAS_RESAMPLE_MAP[interval]
    agg_map: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for optional_col in ("quote_volume", "trades", "taker_buy_base"):
        if optional_col in df.columns:
            agg_map[optional_col] = "sum"

    resample_kwargs: dict[str, Any] = {"label": "left", "closed": "left"}
    if interval == "4h":
        resample_kwargs["origin"] = "epoch"
    resampled = (
        df.resample(rule, **resample_kwargs)
        .agg(agg_map)
        .dropna(subset=["open", "high", "low", "close"])
    )
    return resampled


def load_klines(
    symbol: str,
    interval: str = "4h",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load klines for a symbol (merging sample + cache; cache wins), parse UTC, and resample."""
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval {interval!r}; expected {SUPPORTED_INTERVALS}")

    sources = _resolve_kline_sources(symbol, interval, data_dir=data_dir)
    all_1h = all(src_iv == "1h" for _, src_iv in sources)

    if all_1h:
        frames = [_read_kline_csv(p) for p, _ in sources]
        frames = [f for f in frames if not f.empty]
        if not frames:
            raise FileNotFoundError(f"Kline data is empty for {symbol.upper()}")
        df = pd.concat(frames)
        df = df.sort_index(kind="mergesort")
        df = df[~df.index.duplicated(keep="last")]
        df.index = to_utc_ns_index(df.index)
        if interval != "1h":
            df = resample_ohlcv(df, interval)
        return df

    # At least one source is already at `interval` (or non-1h)
    frames = []
    for path, src_iv in sources:
        logger.debug("Loading %s (%s) from %s", symbol, src_iv, path)
        sub = _read_kline_csv(path)
        if sub.empty:
            continue
        if src_iv != interval:
            if src_iv != "1h":
                raise ValueError(
                    f"Cannot resample from {src_iv} to {interval}; provide 1h or {interval} data."
                )
            sub = resample_ohlcv(sub, interval)
        frames.append(sub)

    if not frames:
        raise FileNotFoundError(f"Kline data is empty for {symbol.upper()}")
    df = pd.concat(frames)
    df = df.sort_index(kind="mergesort")
    df = df[~df.index.duplicated(keep="last")]
    df.index = to_utc_ns_index(df.index)
    return df


def load_funding_rates(symbol: str, data_dir: Path | None = None) -> pd.Series:
    """Load UTC funding rates (merging sample + cache; cache wins), rounding jitter to minute."""
    paths = _resolve_funding_paths(symbol, data_dir=data_dir)
    if not paths:
        logger.warning("No funding file found for %s; returning empty Series", symbol)
        return pd.Series(dtype=float, name="funding_rate")

    frames: list[pd.DataFrame] = []
    for path in paths:
        raw = pd.read_csv(path)
        if raw.empty:
            continue
        ft = to_utc_ns_index(raw["funding_time"]).round("min").as_unit("ns")
        raw = raw.assign(funding_time=ft)
        frames.append(raw[["funding_time", "funding_rate"]])

    if not frames:
        return pd.Series(dtype=float, name="funding_rate")

    df = pd.concat(frames, ignore_index=True).sort_values("funding_time", kind="mergesort")
    df = df.drop_duplicates(subset=["funding_time"], keep="last")
    rates = pd.to_numeric(df["funding_rate"], errors="coerce").fillna(0.0)
    series = pd.Series(
        rates.to_numpy(dtype=np.float64),
        index=to_utc_ns_index(df["funding_time"]),
        name="funding_rate",
    )
    return series


def attach_causal_funding(
    df: pd.DataFrame, funding: pd.Series, interval: str
) -> pd.DataFrame:
    """Attach strictly causal funding features to an OHLCV DataFrame.

    At the close of bar t (`bar_close = open_time + bar_delta`), all funding settlements
    with `funding_time < bar_close` are causally known.
    """
    out = df.copy()
    if funding.empty or out.empty:
        out["funding_rate"] = 0.0
        return out

    bar_delta = pd.Timedelta(hours=INTERVAL_TIMEDELTA_HOURS[interval])
    idx_ns = to_utc_ns_index(out.index)
    bar_close_ns = (idx_ns + bar_delta - pd.Timedelta(nanoseconds=1)).as_unit("ns").asi8
    fund_ns = to_utc_ns_index(funding.index).asi8
    fund_vals = funding.to_numpy(dtype=np.float64)

    # searchsorted(..., side="right") - 1 gives last index where fund_ns <= bar_close - 1ns (< bar_close)
    idx = np.searchsorted(fund_ns, bar_close_ns, side="right") - 1
    aligned = np.where(idx >= 0, fund_vals[np.maximum(idx, 0)], 0.0)
    out["funding_rate"] = aligned
    return out


def concat_market_data(prefix: MarketData | None, suffix: MarketData) -> MarketData:
    """Concatenate pre-window warm-up data with window data without duplicates."""
    if prefix is None:
        return suffix
    ohlcv_out: dict[str, pd.DataFrame] = {}
    funding_out: dict[str, pd.Series] = {}
    all_syms = list(dict.fromkeys([*suffix.ohlcv.keys(), *prefix.ohlcv.keys()]))
    for sym in all_syms:
        p_df = prefix.ohlcv.get(sym, pd.DataFrame())
        s_df = suffix.ohlcv.get(sym, pd.DataFrame())
        parts = [d for d in (p_df, s_df) if not d.empty]
        if parts:
            df = pd.concat(parts).sort_index(kind="mergesort")
            df = df[~df.index.duplicated(keep="last")]
            df.index = to_utc_ns_index(df.index)
            ohlcv_out[sym] = df
        else:
            ohlcv_out[sym] = s_df.copy()

        p_f = prefix.funding.get(sym, pd.Series(dtype=float))
        s_f = suffix.funding.get(sym, pd.Series(dtype=float))
        f_parts = [f for f in (p_f, s_f) if not f.empty]
        if f_parts:
            f_ser = pd.concat(f_parts).sort_index(kind="mergesort")
            f_ser = f_ser[~f_ser.index.duplicated(keep="last")]
            f_ser.index = to_utc_ns_index(f_ser.index)
            f_ser.name = "funding_rate"
            funding_out[sym] = f_ser
        else:
            funding_out[sym] = pd.Series(dtype=float, name="funding_rate")
    return MarketData(ohlcv=ohlcv_out, funding=funding_out, interval=suffix.interval)


def prepare_range_data(
    full_data: MarketData,
    start: str | date | datetime | pd.Timestamp | None = None,
    end: str | date | datetime | pd.Timestamp | None = None,
) -> tuple[MarketData, MarketData, MarketData | None, pd.Timestamp | None]:
    """Slice `full_data` to [start, end] while preserving pre-start warm-up history up to `end`.

    Returns `(history_to_end, window_data, warmup_data, trade_start_ts)`.
    """
    start_ts = parse_utc_bound(start, is_end=False)
    end_ts = parse_utc_bound(end, is_end=True)

    history_to_end = full_data.slice_index_bounds(end=end_ts) if end_ts is not None else full_data
    window_data = history_to_end.slice_index_bounds(start=start_ts) if start_ts is not None else history_to_end

    non_empty_win = [df for df in window_data.ohlcv.values() if not df.empty]
    if not non_empty_win:
        return history_to_end, window_data, None, start_ts

    first_win_ts = min(pd.Timestamp(df.index[0]) for df in non_empty_win)
    has_pre_history = any(
        (not df.empty and pd.Timestamp(df.index[0]) < first_win_ts)
        for df in history_to_end.ohlcv.values()
    )
    if has_pre_history:
        warmup_data = history_to_end.slice_index_bounds(end=first_win_ts - pd.Timedelta(nanoseconds=1))
        return history_to_end, window_data, warmup_data, first_win_ts
    return history_to_end, window_data, None, None


def get_available_date_bounds(data: MarketData) -> tuple[date, date]:
    """Return inclusive UTC `(min_date, max_date)` covered across all non-empty symbols."""
    non_empty = [df for df in data.ohlcv.values() if not df.empty]
    if not non_empty:
        raise ValueError("No market data bars available to determine date bounds.")
    min_ts = min(pd.Timestamp(df.index[0]) for df in non_empty)
    max_ts = max(pd.Timestamp(df.index[-1]) for df in non_empty)
    return min_ts.date(), max_ts.date()


def validate_date_range(
    window_data: MarketData,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    min_days: int = 30,
    min_bars: int = 1,
    label: str = "Backtest",
) -> float:
    """Enforce minimum date span and bar count for a backtest or audit window."""
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(
            f"{label}: start date ({start_ts.date()}) must be on or before end date ({end_ts.date()})."
        )
    non_empty = [df for df in window_data.ohlcv.values() if not df.empty]
    if not non_empty:
        raise ValueError(f"{label}: no market bars available in the selected date range.")

    min_bar = min(pd.Timestamp(df.index[0]) for df in non_empty)
    max_bar = max(pd.Timestamp(df.index[-1]) for df in non_empty)
    bar_delta = pd.Timedelta(hours=INTERVAL_TIMEDELTA_HOURS[window_data.interval])
    effective_days = (max_bar + bar_delta - min_bar).total_seconds() / 86400.0
    if start_ts is not None and end_ts is not None:
        req_days = (end_ts - start_ts).total_seconds() / 86400.0
        if req_days < min_days - 1e-4:
            raise ValueError(
                f"{label} requires at least {min_days} days (selected range is {req_days:.1f} days)."
            )
    if effective_days < min_days - 1e-4:
        raise ValueError(
            f"{label} requires at least {min_days} days of available data in range "
            f"(available span is {effective_days:.1f} days from {min_bar.date()} to {max_bar.date()})."
        )
    max_sym_bars = max(len(df) for df in non_empty)
    if max_sym_bars < min_bars:
        raise ValueError(
            f"{label} requires at least {min_bars} bars in range (found {max_sym_bars})."
        )
    return effective_days


def load_market_data(
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    interval: str = "4h",
    data_dir: Path | None = None,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
) -> MarketData:
    """Load OHLCV and funding rates for multiple symbols with causal funding features."""
    ohlcv_map: dict[str, pd.DataFrame] = {}
    funding_map: dict[str, pd.Series] = {}

    start_ts = parse_utc_bound(start, is_end=False)
    end_ts = parse_utc_bound(end, is_end=True)

    for sym in symbols:
        sym_u = sym.upper()
        df = load_klines(sym_u, interval=interval, data_dir=data_dir)
        f_ser = load_funding_rates(sym_u, data_dir=data_dir)
        df = attach_causal_funding(df, f_ser, interval=interval)
        ohlcv_map[sym_u] = df
        funding_map[sym_u] = f_ser

    md = MarketData(ohlcv=ohlcv_map, funding=funding_map, interval=interval)
    if start_ts is not None or end_ts is not None:
        md = md.slice_index_bounds(start=start_ts, end=end_ts)
    return md
