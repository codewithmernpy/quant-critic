"""Deliberately flawed lookahead strategy demonstrating what QuantCritic catches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.strategies.base import Strategy, build_signal_frame, compute_atr


class LeakyDemoStrategy(Strategy):
    """DEMO OF LOOKAHEAD BIAS: centered rolling smoothing + full-sample z-score normalization.

    WARNING: This strategy intentionally cheats to showcase QuantCritic's integrity checks:
      1. Uses `rolling(smooth_window, center=True).mean()`, peeking future bars.
      2. Normalizes signals using full-sample `.mean()` and `.std()` across the entire DataFrame.
    It produces an unrealistically strong raw backtest and MUST be rejected by lookahead audits.
    """

    name = "leaky_demo"
    description = (
        "[DELIBERATE CHEAT DEMO] Smoothed momentum z-score with subtle lookahead bugs: "
        "centered rolling average (peeks future bars) and full-sample z-score normalization."
    )
    default_params = {
        "smooth_window": 5,
        "z_thresh": 0.75,
        "atr_period": 14,
        "atr_stop_mult": 2.0,
        "max_bars": 4,
    }
    param_grid = {
        "smooth_window": [5, 7, 9],
        "z_thresh": [0.5, 1.0],
        "atr_stop_mult": [2.0, 3.0],
    }

    def generate(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        p = self.merge_params(params)
        smooth_win = int(p["smooth_window"])
        z_thresh = float(p["z_thresh"])
        atr_period = int(p["atr_period"])
        atr_mult = float(p["atr_stop_mult"])
        max_bars = int(p["max_bars"])

        close = df["close"].astype(float)
        atr = compute_atr(df, atr_period)

        # BUG 1 (Lookahead): center=True uses bars t+1..t+(w//2) to estimate smoothed fair value at t
        centered_fair = close.rolling(smooth_win, center=True).mean()
        raw_edge = (centered_fair - close) / atr.replace(0.0, np.nan)

        # BUG 2 (Lookahead): full-sample mean/std across future and past data
        edge_mean = float(raw_edge.mean()) if np.isfinite(raw_edge.mean()) else 0.0
        edge_std = float(raw_edge.std(ddof=1)) if np.isfinite(raw_edge.std(ddof=1)) else 1.0
        edge_z = (raw_edge - edge_mean) / max(edge_std, 1e-8)

        sig = np.zeros(len(df), dtype=np.int8)
        sig[(edge_z > z_thresh).fillna(False).to_numpy()] = 1
        sig[(edge_z < -z_thresh).fillna(False).to_numpy()] = -1

        warmup = atr_period + 2
        if len(sig) > 0:
            sig[: min(warmup, len(sig))] = 0

        stop_dist = (atr * atr_mult).to_numpy(dtype=np.float64)
        return build_signal_frame(
            index=df.index,
            signal=sig,
            stop_dist=stop_dist,
            max_bars=max_bars,
        )
