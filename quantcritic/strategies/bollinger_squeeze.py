"""Bollinger bandwidth percentile squeeze breakout strategy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.strategies.base import Strategy, build_signal_frame, compute_atr


class BollingerSqueezeStrategy(Strategy):
    """Trades volatility expansion breakouts following a causal bandwidth percentile squeeze."""

    name = "bollinger_squeeze"
    description = (
        "Identifies volatility compression via causal rolling percentile rank of Bollinger "
        "bandwidth, then enters breakouts outside the upper/lower band with ATR stops."
    )
    default_params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "pct_lookback": 120,
        "squeeze_pct": 0.20,
        "atr_period": 14,
        "atr_stop_mult": 2.0,
        "tp_r": 2.5,
        "max_bars": 24,
    }
    param_grid = {
        "bb_period": [20, 30],
        "squeeze_pct": [0.15, 0.25],
        "atr_stop_mult": [2.0, 3.0],
        "tp_r": [2.0, 3.0],
    }

    def generate(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        p = self.merge_params(params)
        bb_period = int(p["bb_period"])
        bb_std = float(p["bb_std"])
        pct_lookback = int(p["pct_lookback"])
        squeeze_pct = float(p["squeeze_pct"])
        atr_period = int(p["atr_period"])
        atr_mult = float(p["atr_stop_mult"])
        tp_r = float(p["tp_r"])
        max_bars = int(p["max_bars"])

        close = df["close"].astype(float)
        mid = close.rolling(bb_period, min_periods=bb_period).mean()
        sd = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
        upper = mid + bb_std * sd
        lower = mid - bb_std * sd
        bandwidth = (upper - lower) / mid.replace(0.0, np.nan)

        bw_rank = bandwidth.rolling(pct_lookback, min_periods=max(20, pct_lookback // 2)).rank(pct=True)
        # Squeeze active in prior 3 bars before breakout expansion
        recent_squeeze = (bw_rank <= squeeze_pct).rolling(3, min_periods=1).max().shift(1) > 0

        long_break = recent_squeeze & (close > upper) & (close.shift(1) <= upper.shift(1))
        short_break = recent_squeeze & (close < lower) & (close.shift(1) >= lower.shift(1))

        sig = np.zeros(len(df), dtype=np.int8)
        sig[long_break.fillna(False).to_numpy()] = 1
        sig[short_break.fillna(False).to_numpy()] = -1

        warmup = max(bb_period + pct_lookback // 2, atr_period) + 2
        if len(sig) > 0:
            sig[: min(warmup, len(sig))] = 0

        atr = compute_atr(df, atr_period)
        stop_dist = (atr * atr_mult).to_numpy(dtype=np.float64)
        return build_signal_frame(
            index=df.index,
            signal=sig,
            stop_dist=stop_dist,
            tp_r=tp_r,
            max_bars=max_bars,
        )
