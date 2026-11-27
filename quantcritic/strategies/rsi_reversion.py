"""RSI mean-reversion strategy aligned with a 200-EMA macro trend filter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.strategies.base import (
    Strategy,
    build_signal_frame,
    compute_atr,
    compute_ema,
    compute_rsi,
)


class RsiReversionStrategy(Strategy):
    """Enters pullbacks at RSI extremes in the direction of the 200-EMA trend with time stop."""

    name = "rsi_reversion"
    description = (
        "Counter-trend pullback strategy: buys oversold RSI extremes above the 200-EMA "
        "and shorts overbought RSI extremes below the 200-EMA, protected by ATR and time stops."
    )
    default_params = {
        "rsi_period": 14,
        "rsi_low": 30.0,
        "ema_trend_period": 200,
        "atr_period": 14,
        "atr_stop_mult": 2.0,
        "tp_r": 2.0,
        "max_bars": 18,
    }
    param_grid = {
        "rsi_period": [10, 14],
        "rsi_low": [25.0, 32.0],
        "atr_stop_mult": [1.5, 2.5],
        "max_bars": [12, 24],
        "tp_r": [1.5, 2.5],
    }

    def generate(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        p = self.merge_params(params)
        rsi_period = int(p["rsi_period"])
        rsi_low = float(p["rsi_low"])
        rsi_high = 100.0 - rsi_low
        ema_period = int(p["ema_trend_period"])
        atr_period = int(p["atr_period"])
        atr_mult = float(p["atr_stop_mult"])
        tp_r = float(p["tp_r"])
        max_bars = int(p["max_bars"])

        close = df["close"].astype(float)
        ema_trend = compute_ema(close, ema_period)
        rsi = compute_rsi(close, rsi_period)
        atr = compute_atr(df, atr_period)

        prev_rsi = rsi.shift(1)
        long_entry = (close > ema_trend) & (rsi < rsi_low) & (prev_rsi >= rsi_low)
        short_entry = (close < ema_trend) & (rsi > rsi_high) & (prev_rsi <= rsi_high)

        sig = np.zeros(len(df), dtype=np.int8)
        sig[long_entry.fillna(False).to_numpy()] = 1
        sig[short_entry.fillna(False).to_numpy()] = -1

        warmup = max(ema_period, rsi_period, atr_period) + 2
        if len(sig) > 0:
            sig[: min(warmup, len(sig))] = 0

        stop_dist = (atr * atr_mult).to_numpy(dtype=np.float64)
        return build_signal_frame(
            index=df.index,
            signal=sig,
            stop_dist=stop_dist,
            tp_r=tp_r,
            max_bars=max_bars,
        )
