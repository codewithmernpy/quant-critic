"""Base Strategy interface and shared causal indicator primitives."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute causal Wilder Average True Range (ATR)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["close"].astype(float).shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / max(1, period), adjust=False, min_periods=period).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute strictly causal Exponential Moving Average."""
    return series.astype(float).ewm(span=max(1, span), adjust=False, min_periods=span).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute causal Wilder Relative Strength Index (0..100)."""
    delta = series.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / max(1, period), adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / max(1, period), adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Where avg_loss is 0 and avg_gain > 0 -> RSI = 100; both 0 -> RSI = 50
    rsi = rsi.where(avg_loss > 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    return rsi


def build_signal_frame(
    index: pd.Index,
    signal: pd.Series | np.ndarray,
    stop_dist: pd.Series | np.ndarray,
    tp_r: pd.Series | np.ndarray | float = np.nan,
    max_bars: pd.Series | np.ndarray | int = 0,
    exit_signal: pd.Series | np.ndarray | bool = False,
) -> pd.DataFrame:
    """Construct and validate a standardized strategy output DataFrame aligned to `index`."""
    sig_arr = np.asarray(signal, dtype=np.int8)
    sd_arr = np.asarray(stop_dist, dtype=np.float64)
    # Enforce valid positive stop distance where signal != 0
    invalid = (sig_arr != 0) & (~np.isfinite(sd_arr) | (sd_arr <= 0.0))
    if np.any(invalid):
        sig_arr = np.where(invalid, 0, sig_arr)

    n = len(index)
    if np.ndim(tp_r) == 0:
        tp_val = float(tp_r) if tp_r is not None else np.nan
        tp_arr = np.full(n, tp_val if tp_val > 0 else np.nan, dtype=np.float64)
    else:
        tp_arr = np.asarray(tp_r, dtype=np.float64)

    if np.ndim(max_bars) == 0:
        mb_val = int(max_bars) if max_bars is not None else 0
        mb_arr = np.full(n, max(0, mb_val), dtype=np.int32)
    else:
        mb_arr = np.asarray(max_bars, dtype=np.int32)

    if np.ndim(exit_signal) == 0:
        ex_arr = np.full(n, bool(exit_signal), dtype=bool)
    else:
        ex_arr = np.asarray(exit_signal, dtype=bool)

    return pd.DataFrame(
        {
            "signal": sig_arr,
            "stop_dist": sd_arr,
            "tp_r": tp_arr,
            "max_bars": mb_arr,
            "exit_signal": ex_arr,
        },
        index=index,
    )


class Strategy(ABC):
    """Abstract base class for QuantCritic trading strategies."""

    name: str = ""
    description: str = ""
    default_params: dict[str, Any] = {}
    param_grid: dict[str, list[Any]] = {}

    def merge_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(self.default_params)
        if params:
            merged.update(params)
        return merged

    def expand_param_grid(self) -> list[dict[str, Any]]:
        """Expand param_grid into a deterministic list of parameter combinations (<= 36)."""
        if not self.param_grid:
            return [dict(self.default_params)]
        keys = sorted(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        combos: list[dict[str, Any]] = []
        for prod in itertools.product(*values):
            p = dict(self.default_params)
            for k, v in zip(keys, prod):
                p[k] = v
            if self.is_valid_params(p):
                combos.append(p)
        return combos[:36]

    def is_valid_params(self, params: dict[str, Any]) -> bool:
        """Optional hook to prune invalid combinations from grid expansion."""
        return True

    @abstractmethod
    def generate(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Generate causal bar-close signals aligned to `df.index`."""
