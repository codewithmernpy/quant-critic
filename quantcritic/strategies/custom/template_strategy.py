"""Template Python strategy plugin for QuantCritic.

NOTE: This file (`template_strategy.py`) is intentionally IGNORED by auto-discovery so it
is NOT registered by default. To activate your own Python strategy plugin:

1. Copy or rename this file inside `quantcritic/strategies/custom/` (for example:
   `my_momentum.py`). Files starting with `_` and `template_strategy.py` are skipped.
2. Subclass `quantcritic.strategies.base.Strategy` with a unique lowercase `name`.
3. Implement `generate(self, df, params)` using ONLY strictly causal operations on bars <= t
   (no `rolling(..., center=True)`, no negative shifts, no full-sample statistics).
4. Every plugin automatically passes through the same causal truncation & audit checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.strategies.base import Strategy, build_signal_frame, compute_atr, compute_ema


class TemplateCustomStrategy(Strategy):
    """Example causal EMA crossover + ATR stop plugin template (rename file to enable)."""

    name = "template_custom"
    description = "Template causal plugin strategy (rename template_strategy.py to register)."
    default_params = {
        "fast_span": 12,
        "slow_span": 48,
        "atr_period": 14,
        "atr_stop_mult": 2.0,
        "tp_r": 2.0,
    }
    param_grid = {
        "fast_span": [10, 16],
        "slow_span": [40, 60],
        "atr_stop_mult": [2.0, 2.5],
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

        bull = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        bear = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        sig = np.zeros(len(df), dtype=np.int8)
        sig[bull.fillna(False).to_numpy()] = 1
        sig[bear.fillna(False).to_numpy()] = -1

        warmup = max(slow_span, atr_period) + 2
        if len(sig) > 0:
            sig[: min(warmup, len(sig))] = 0

        stop_dist = (atr * atr_mult).to_numpy(dtype=np.float64)
        return build_signal_frame(
            index=df.index,
            signal=sig,
            stop_dist=stop_dist,
            tp_r=tp_r if tp_r > 0 else np.nan,
        )
