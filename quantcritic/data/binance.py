"""Downloader for public Binance Vision USDT-M futures monthly klines and funding rates."""

from __future__ import annotations

import io
import json
import logging
import re
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantcritic.config import CACHE_DATA_DIR, SAMPLE_DATA_DIR, SUPPORTED_INTERVALS

logger = logging.getLogger(__name__)

BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/monthly"
SYMBOL_REGEX = re.compile(r"^[A-Z0-9]{2,20}USDT$")

RAW_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

OUTPUT_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
]


def parse_epoch_timestamps(series: pd.Series) -> pd.DatetimeIndex:
    """Parse timestamps supporting seconds, milliseconds, microseconds, nanoseconds, or ISO strings."""
    if pd.api.types.is_numeric_dtype(series):
        vals = series.astype(np.int64)
    else:
        first = str(series.iloc[0]).strip()
        if not first.lstrip("-").isdigit():
            return pd.DatetimeIndex(pd.to_datetime(series, utc=True, format="ISO8601")).as_unit("ns")
        vals = pd.to_numeric(series, errors="raise").astype(np.int64)

    med = float(np.median(np.abs(vals)))
    if med >= 1e17:
        unit = "ns"
    elif med >= 1e14:
        unit = "us"
    elif med >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.DatetimeIndex(pd.to_datetime(vals, unit=unit, utc=True)).as_unit("ns")


def parse_binance_kline_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse a Binance Vision kline CSV (with or without header, ms or us timestamps)."""
    text_sample = raw_bytes[:512].decode("utf-8", errors="ignore").lstrip()
    first_token = text_sample.split(",", 1)[0].strip().lower()
    has_header = first_token == "open_time" or not first_token.lstrip("-").isdigit()

    buf = io.BytesIO(raw_bytes)
    if has_header:
        df = pd.read_csv(buf)
        rename_map = {
            "count": "trades",
            "taker_buy_volume": "taker_buy_base",
            "taker_buy_quote_volume": "taker_buy_quote",
        }
        df = df.rename(columns=rename_map)
        if list(df.columns[:6]) != RAW_KLINE_COLUMNS[:6] and len(df.columns) >= 12:
            df.columns = RAW_KLINE_COLUMNS[: len(df.columns)]
    else:
        df = pd.read_csv(buf, header=None, names=RAW_KLINE_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_KLINE_COLUMNS)

    df["open_time"] = parse_epoch_timestamps(df["open_time"])
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(np.int64)

    out = df[OUTPUT_KLINE_COLUMNS].dropna(subset=["open_time", "open", "high", "low", "close"])
    return out


def parse_binance_funding_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse a Binance Vision fundingRate CSV (with or without header, ms or us timestamps)."""
    text_sample = raw_bytes[:512].decode("utf-8", errors="ignore").lstrip()
    first_line = text_sample.splitlines()[0] if text_sample else ""
    first_token = first_line.split(",", 1)[0].strip().lower()
    has_header = any(ch.isalpha() for ch in first_token)

    buf = io.BytesIO(raw_bytes)
    if has_header:
        df = pd.read_csv(buf)
        cols_lower = {c.lower(): c for c in df.columns}
        time_col = (
            cols_lower.get("calc_time")
            or cols_lower.get("funding_time")
            or cols_lower.get("time")
            or df.columns[0]
        )
        rate_col = (
            cols_lower.get("last_funding_rate")
            or cols_lower.get("funding_rate")
            or cols_lower.get("fundingrate")
            or df.columns[-1]
        )
        out = pd.DataFrame(
            {
                "funding_time": parse_epoch_timestamps(df[time_col]).round("min"),
                "funding_rate": pd.to_numeric(df[rate_col], errors="coerce"),
            }
        )
    else:
        df = pd.read_csv(buf, header=None)
        if df.shape[1] >= 3:
            # Typical schema: calc_time, funding_interval_hours, last_funding_rate
            time_col_idx, rate_col_idx = 0, 2
        else:
            time_col_idx, rate_col_idx = 0, 1
        out = pd.DataFrame(
            {
                "funding_time": parse_epoch_timestamps(df.iloc[:, time_col_idx]).round("min"),
                "funding_rate": pd.to_numeric(df.iloc[:, rate_col_idx], errors="coerce"),
            }
        )

    return out.dropna(subset=["funding_time", "funding_rate"])


def normalize_symbol(raw: str) -> str:
    """Normalize and validate a Binance USDT-M symbol (e.g. 'xrp' -> 'XRPUSDT')."""
    sym = raw.strip().upper()
    if not sym:
        raise ValueError("Symbol cannot be empty.")
    if not sym.endswith("USDT"):
        sym = f"{sym}USDT"
    if not SYMBOL_REGEX.match(sym):
        raise ValueError(
            f"Invalid Binance USDT-M symbol {raw!r} (normalized {sym!r}); "
            "expected format ^[A-Z0-9]{2,20}USDT$."
        )
    return sym


def parse_symbols_input(raw: str) -> tuple[list[str], list[str]]:
    """Parse comma/whitespace-separated symbols into (valid_unique_symbols, invalid_tokens)."""
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        try:
            norm = normalize_symbol(token)
        except ValueError:
            invalid.append(token)
            continue
        if norm not in seen:
            seen.add(norm)
            valid.append(norm)
    return valid, invalid


def latest_complete_month(reference_date: date | None = None) -> str:
    """Return YYYY-MM for the most recently completed UTC calendar month."""
    ref = reference_date or datetime.now(timezone.utc).date()
    prev_last = date(ref.year, ref.month, 1) - timedelta(days=1)
    return f"{prev_last.year:04d}-{prev_last.month:02d}"


def _norm_month_str(month_str: str) -> str:
    s = month_str.strip()
    if len(s) >= 7 and s[4] == "-":
        s = s[:7]
    parts = s.split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected YYYY-MM month string, got {month_str!r}")
    y, m = int(parts[0]), int(parts[1])
    if not (1 <= m <= 12):
        raise ValueError(f"Invalid month in {month_str!r}")
    return f"{y:04d}-{m:02d}"


def _month_range(start_month: str, end_month: str) -> list[tuple[int, int]]:
    """Generate (year, month) tuples inclusively from YYYY-MM strings."""
    sy, sm = map(int, _norm_month_str(start_month).split("-"))
    ey, em = map(int, _norm_month_str(end_month).split("-"))
    cur = date(sy, sm, 1)
    end_d = date(ey, em, 1)
    months: list[tuple[int, int]] = []
    while cur <= end_d:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return months


def _read_months_from_csv(path: Path, time_col: str) -> set[tuple[int, int]]:
    """Return the set of (year, month) pairs already stored in a gzip CSV."""
    if not path.is_file():
        return set()
    try:
        df = pd.read_csv(path, usecols=[time_col])
    except Exception:
        return set()
    if df.empty:
        return set()
    ts = pd.to_datetime(df[time_col], utc=True, format="ISO8601", errors="coerce").dropna()
    if ts.empty:
        return set()
    return set(zip(ts.dt.year.astype(int), ts.dt.month.astype(int)))


def _meta_path(out_dir: Path, symbol: str) -> Path:
    return out_dir / f"{symbol.upper()}_meta.json"


def _load_symbol_meta(out_dir: Path, symbol: str) -> dict[str, Any]:
    path = _meta_path(out_dir, symbol)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_symbol_meta(out_dir: Path, symbol: str, meta: dict[str, Any]) -> None:
    path = _meta_path(out_dir, symbol)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _month_tuple(ym: str) -> tuple[int, int]:
    y, m = map(int, _norm_month_str(ym).split("-"))
    return (y, m)


def _fmt_month(ym: tuple[int, int]) -> str:
    return f"{ym[0]:04d}-{ym[1]:02d}"


def get_stored_months(
    symbol: str,
    interval: str = "1h",
    cache_dir: Path | None = None,
    sample_dir: Path | None = None,
    check_sample: bool = True,
) -> dict[str, Any]:
    """Inspect locally stored kline/funding months across cache (and sample if enabled)."""
    sym = normalize_symbol(symbol)
    c_dir = cache_dir or CACHE_DATA_DIR
    s_dir = sample_dir or SAMPLE_DATA_DIR

    kline_months: set[tuple[int, int]] = _read_months_from_csv(c_dir / f"{sym}_{interval}.csv.gz", "open_time")
    funding_months: set[tuple[int, int]] = _read_months_from_csv(c_dir / f"{sym}_funding.csv.gz", "funding_time")

    if check_sample:
        kline_months |= _read_months_from_csv(s_dir / f"{sym}_{interval}.csv.gz", "open_time")
        funding_months |= _read_months_from_csv(s_dir / f"{sym}_funding.csv.gz", "funding_time")

    meta = _load_symbol_meta(c_dir, sym)
    known_unavailable: set[tuple[int, int]] = set()
    probed_from = meta.get("probed_from")
    first_avail = meta.get("first_available_month")
    if probed_from and first_avail:
        pf = _month_tuple(probed_from)
        fa = _month_tuple(first_avail)
        if pf < fa:
            for m in _month_range(probed_from, first_avail):
                if m < fa:
                    known_unavailable.add(m)

    return {
        "klines": kline_months,
        "funding": funding_months,
        "known_unavailable": known_unavailable,
        "meta": meta,
    }


def _fetch_zip_csv_bytes(url: str, timeout: float = 30.0) -> bytes | None:
    """Download a single zip archive from Binance Vision and extract the first CSV file."""
    req = urllib.request.Request(url, headers={"User-Agent": "QuantCritic/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.info("Archive not available (404): %s", url)
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return None
        return zf.read(csv_names[0])


def _merge_csv_frames(
    existing_path: Path,
    new_frames: list[pd.DataFrame],
    time_col: str,
    round_min: bool = False,
) -> pd.DataFrame:
    """Merge existing cached gzip CSV with newly downloaded frames, avoiding duplicates."""
    frames: list[pd.DataFrame] = []
    if existing_path.is_file():
        prev = pd.read_csv(existing_path)
        if not prev.empty:
            ts = pd.to_datetime(prev[time_col], utc=True, format="ISO8601")
            if round_min:
                ts = ts.dt.round("min")
            prev[time_col] = ts.dt.as_unit("ns")
            frames.append(prev)
    frames.extend(new_frames)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(time_col, kind="mergesort").drop_duplicates(subset=[time_col], keep="last")
    return merged


def ensure_symbol_data(
    symbol: str,
    interval: str = "1h",
    start_month: str = "2021-01",
    end_month: str | None = None,
    output_dir: Path | None = None,
    include_funding: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    skip_existing: bool = True,
    check_sample: bool | None = None,
) -> dict[str, Any]:
    """Ensure monthly klines and fundingRate archives exist locally, downloading only missing months.

    Never modifies `data/sample/` files; new/missing months are saved and merged into `data/cache/`.
    Returns status metadata including downloaded months and whether the coin listed later.
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval {interval!r}; expected {SUPPORTED_INTERVALS}")

    sym = normalize_symbol(symbol)
    out_dir = output_dir or CACHE_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    s_month = _norm_month_str(start_month)
    e_month = _norm_month_str(end_month or latest_complete_month())
    months = _month_range(s_month, e_month)

    should_check_sample = (output_dir is None or output_dir == CACHE_DATA_DIR) if check_sample is None else check_sample
    stored = get_stored_months(
        sym,
        interval=interval,
        cache_dir=out_dir,
        check_sample=should_check_sample,
    )
    stored_kline_months: set[tuple[int, int]] = set(stored["klines"]) if skip_existing else set()
    stored_funding_months: set[tuple[int, int]] = set(stored["funding"]) if skip_existing else set()
    known_unavail: set[tuple[int, int]] = set(stored["known_unavailable"]) if skip_existing else set()

    missing_kline_months = [m for m in months if m not in stored_kline_months and m not in known_unavail]

    saved: dict[str, Path] = {}
    kline_frames: list[pd.DataFrame] = []
    downloaded_kline_months: list[str] = []
    est_total = max(len(missing_kline_months) * (2 if include_funding else 1), 1)
    done_steps = 0

    for year, month in missing_kline_months:
        done_steps += 1
        if progress_callback is not None:
            progress_callback(done_steps, est_total, f"Downloading {sym} klines {year}-{month:02d}...")
        url = (
            f"{BINANCE_VISION_BASE}/klines/{sym}/{interval}/"
            f"{sym}-{interval}-{year}-{month:02d}.zip"
        )
        raw = _fetch_zip_csv_bytes(url)
        if raw is not None:
            df_k = parse_binance_kline_csv(raw)
            if not df_k.empty:
                kline_frames.append(df_k)
                downloaded_kline_months.append(f"{year:04d}-{month:02d}")
                stored_kline_months.add((year, month))

    kline_path = out_dir / f"{sym}_{interval}.csv.gz"
    if kline_frames:
        klines = _merge_csv_frames(kline_path, kline_frames, time_col="open_time")
        klines.to_csv(kline_path, index=False, compression="gzip")
        saved["klines"] = kline_path
        logger.info("Saved %d kline rows to %s", len(klines), kline_path)
    elif kline_path.is_file():
        saved["klines"] = kline_path

    if not stored_kline_months:
        return {
            "symbol": sym,
            "status": "not_found",
            "saved": {},
            "downloaded_months": [],
            "first_available_month": None,
            "last_available_month": None,
            "listed_later": False,
            "message": f"{sym}: no data found on Binance Vision for {s_month}..{e_month} (404).",
        }

    earliest_kline_ym = min(stored_kline_months)
    latest_kline_ym = max(stored_kline_months)
    first_avail_str = _fmt_month(earliest_kline_ym)
    last_avail_str = _fmt_month(latest_kline_ym)

    # Record probed start & earliest month found so pre-listing 404 months aren't re-queried
    meta = dict(stored["meta"])
    old_probed = meta.get("probed_from")
    new_probed = s_month if not old_probed else min(old_probed, s_month)
    meta["probed_from"] = new_probed
    meta["first_available_month"] = first_avail_str
    meta["last_available_month"] = last_avail_str
    _save_symbol_meta(out_dir, sym, meta)

    if include_funding:
        missing_funding_months = [
            m for m in months
            if m >= earliest_kline_ym and m not in stored_funding_months
        ]
        total_steps = max(len(missing_kline_months) + len(missing_funding_months), 1)
        funding_frames: list[pd.DataFrame] = []
        for year, month in missing_funding_months:
            done_steps += 1
            if progress_callback is not None:
                progress_callback(done_steps, total_steps, f"Downloading {sym} funding {year}-{month:02d}...")
            url = (
                f"{BINANCE_VISION_BASE}/fundingRate/{sym}/"
                f"{sym}-fundingRate-{year}-{month:02d}.zip"
            )
            raw = _fetch_zip_csv_bytes(url)
            if raw is not None:
                df_f = parse_binance_funding_csv(raw)
                if not df_f.empty:
                    funding_frames.append(df_f)
        funding_path = out_dir / f"{sym}_funding.csv.gz"
        if funding_frames:
            funding = _merge_csv_frames(funding_path, funding_frames, time_col="funding_time", round_min=True)
            funding.to_csv(funding_path, index=False, compression="gzip")
            saved["funding"] = funding_path
            logger.info("Saved %d funding rows to %s", len(funding), funding_path)
        elif funding_path.is_file():
            saved["funding"] = funding_path

    listed_later = earliest_kline_ym > _month_tuple(s_month)
    status = "downloaded" if downloaded_kline_months else "cached"
    if listed_later:
        msg = f"{sym}: listed later than requested start {s_month}; using first available month {first_avail_str}."
    elif downloaded_kline_months:
        msg = f"{sym}: downloaded {len(downloaded_kline_months)} month(s) ({downloaded_kline_months[0]}..{downloaded_kline_months[-1]})."
    else:
        msg = f"{sym}: using cached data ({first_avail_str}..{last_avail_str})."

    return {
        "symbol": sym,
        "status": status,
        "saved": saved,
        "downloaded_months": downloaded_kline_months,
        "first_available_month": first_avail_str,
        "last_available_month": last_avail_str,
        "listed_later": listed_later,
        "message": msg,
    }


def update_symbol_to_latest(
    symbol: str,
    interval: str = "1h",
    end_month: str | None = None,
    output_dir: Path | None = None,
    include_funding: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Download only missing months from earliest stored month through latest complete month."""
    sym = normalize_symbol(symbol)
    out_dir = output_dir or CACHE_DATA_DIR
    should_check = output_dir is None or output_dir == CACHE_DATA_DIR
    stored = get_stored_months(sym, interval=interval, cache_dir=out_dir, check_sample=should_check)
    start_m = _fmt_month(min(stored["klines"])) if stored["klines"] else "2021-01"
    target_end = _norm_month_str(end_month or latest_complete_month())
    return ensure_symbol_data(
        symbol=sym,
        interval=interval,
        start_month=start_m,
        end_month=target_end,
        output_dir=out_dir,
        include_funding=include_funding,
        progress_callback=progress_callback,
        skip_existing=True,
        check_sample=should_check,
    )


def download_symbol_data(
    symbol: str,
    interval: str = "1h",
    start_month: str = "2021-01",
    end_month: str = "2025-10",
    output_dir: Path | None = None,
    include_funding: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Download monthly klines and fundingRate archives from data.binance.vision.

    Saves gzip-compressed CSV files matching the data/sample schema into data/cache/.
    """
    res = ensure_symbol_data(
        symbol=symbol,
        interval=interval,
        start_month=start_month,
        end_month=end_month,
        output_dir=output_dir,
        include_funding=include_funding,
        progress_callback=progress_callback,
        skip_existing=skip_existing,
    )
    return dict(res["saved"])
