"""EMA trend-following strategy with ATR protective stops."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.strategies.base import Strategy, build_signal_frame, compute_atr, compute_ema


class EmaTrendStrategy(Strategy):
    """Dual exponential moving average crossover with ATR volatility stop."""

    name = "ema_trend"
    description = (
        "Classic trend-following crossover: enters long/short when fast EMA crosses "
        "above/below slow EMA, sized via causal ATR protective stops."
    )
    default_params = {
        "fast_span": 20,
        "slow_span": 60,
        "atr_period": 14,
        "atr_stop_mult": 2.5,
        "tp_r": 0.0,
    }
    param_grid = {
        "fast_span": [10, 20, 30],
        "slow_span": [50, 80, 120],
        "atr_stop_mult": [2.0, 3.0],
    }

    def is_valid_params(self, params: dict[str, Any]) -> bool:
        return int(params.get("fast_span", 0)) < int(params.get("slow_span", 1))

    def generate(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        p = self.merge_params(params)
        fast_span = int(p["fast_span"])
        slow_span = int(p["slow_span"])
        atr_period = int(p["atr_period"])
        atr_mult = float(p["atr_stop_mult"])
        tp_r = float(p.get("tp_r", 0.0) or 0.0)

        close = df["close"].astype(float)
        ema_fast = compute_ema(close, fast_span)
        ema_slow = compute_ema(close, slow_span)
        atr = compute_atr(df, atr_period)

        prev_fast = ema_fast.shift(1)
        prev_slow = ema_slow.shift(1)
        bull_cross = (ema_fast > ema_slow) & (prev_fast <= prev_slow)
        bear_cross = (ema_fast < ema_slow) & (prev_fast >= prev_slow)

        sig = np.zeros(len(df), dtype=np.int8)
        sig[bull_cross.fillna(False).to_numpy()] = 1
        sig[bear_cross.fillna(False).to_numpy()] = -1

        warmup = max(slow_span, atr_period) + 1
        if len(sig) > 0:
            sig[: min(warmup, len(sig))] = 0

        stop_dist = (atr * atr_mult).to_numpy(dtype=np.float64)
        return build_signal_frame(
            index=df.index,
            signal=sig,
            stop_dist=stop_dist,
            tp_r=tp_r if tp_r > 0 else np.nan,
        )
