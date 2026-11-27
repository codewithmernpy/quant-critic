"""Donchian channel breakout strategy with ATR stop and trailing channel exit."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantcritic.strategies.base import Strategy, build_signal_frame, compute_atr


class DonchianBreakoutStrategy(Strategy):
    """Turtle-style N-bar Donchian breakout with causal ATR stop and M-bar trailing exit."""

    name = "donchian_breakout"
    description = (
        "Enters on breakouts beyond the prior N-bar high/low channel with an ATR "
        "protective stop, optional R-multiple take profit, and M-bar opposite channel exit."
    )
    default_params = {
        "entry_period": 36,
        "exit_period": 18,
        "atr_period": 14,
        "atr_stop_mult": 2.5,
        "tp_r": 0.0,
    }
    param_grid = {
        "entry_period": [20, 36, 55],
        "exit_period": [10, 18],
        "atr_stop_mult": [2.0, 3.0],
        "tp_r": [0.0, 2.5],
    }

    def is_valid_params(self, params: dict[str, Any]) -> bool:
        return int(params.get("exit_period", 0)) < int(params.get("entry_period", 1))

    def generate(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        p = self.merge_params(params)
        entry_n = int(p["entry_period"])
        exit_m = int(p["exit_period"])
        atr_period = int(p["atr_period"])
        atr_mult = float(p["atr_stop_mult"])
        tp_r = float(p.get("tp_r", 0.0) or 0.0)

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        upper = high.shift(1).rolling(entry_n, min_periods=entry_n).max().to_numpy(dtype=np.float64)
        lower = low.shift(1).rolling(entry_n, min_periods=entry_n).min().to_numpy(dtype=np.float64)
        exit_long = low.shift(1).rolling(exit_m, min_periods=exit_m).min().to_numpy(dtype=np.float64)
        exit_short = high.shift(1).rolling(exit_m, min_periods=exit_m).max().to_numpy(dtype=np.float64)

        c_arr = close.to_numpy(dtype=np.float64)
        atr_arr = compute_atr(df, atr_period).to_numpy(dtype=np.float64)

        n = len(df)
        sig = np.zeros(n, dtype=np.int8)
        ex_sig = np.zeros(n, dtype=bool)
        warmup = max(entry_n, exit_m, atr_period) + 1

        # Purely causal state machine tracking directional breakout regime
        # (plain Python floats: ~5x faster than numpy scalar indexing in this loop)
        c_l, u_l, l_l = c_arr.tolist(), upper.tolist(), lower.tolist()
        el_l, es_l, atr_l = exit_long.tolist(), exit_short.tolist(), atr_arr.tolist()
        isfinite = math.isfinite
        regime = 0
        for t in range(warmup, n):
            c = c_l[t]
            u = u_l[t]
            l = l_l[t]
            el = el_l[t]
            es = es_l[t]
            if not (isfinite(c) and isfinite(u) and isfinite(l) and isfinite(atr_l[t])):
                continue

            if c > u:
                if regime != 1:
                    sig[t] = 1
                    regime = 1
            elif c < l:
                if regime != -1:
                    sig[t] = -1
                    regime = -1
            elif regime == 1 and isfinite(el) and c < el:
                ex_sig[t] = True
                regime = 0
            elif regime == -1 and isfinite(es) and c > es:
                ex_sig[t] = True
                regime = 0

        stop_dist = atr_arr * atr_mult
        return build_signal_frame(
            index=df.index,
            signal=sig,
            stop_dist=stop_dist,
            tp_r=tp_r if tp_r > 0 else np.nan,
            exit_signal=ex_sig,
        )
