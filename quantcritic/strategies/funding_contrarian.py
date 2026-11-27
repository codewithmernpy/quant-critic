"""Perpetual futures contrarian strategy fading crowded funding extremes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantcritic.strategies.base import Strategy, build_signal_frame, compute_atr, compute_ema


class FundingContrarianStrategy(Strategy):
    """Fades extreme causal funding-rate z-scores when price is stretched away from EMA."""

    name = "funding_contrarian"
    description = (
        "Crypto-derivatives contrarian model: shorts over-leveraged long crowding when "
        "causal funding z-score and price extension hit positive extremes (collecting funding), "
        "and buys capitulation when both hit negative extremes."
    )
    default_params = {
        "fund_window": 90,
        "fund_z_thresh": 2.0,
        "ema_period": 30,
        "price_stretch_atr": 1.25,
        "atr_period": 14,
        "atr_stop_mult": 2.5,
        "tp_r": 2.0,
        "max_bars": 18,
    }
    param_grid = {
        "fund_window": [60, 120],
        "fund_z_thresh": [1.5, 2.0, 2.5],
        "atr_stop_mult": [2.0, 3.0],
        "max_bars": [12, 24],
    }

    def generate(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        p = self.merge_params(params)
        fund_win = int(p["fund_window"])
        z_thresh = float(p["fund_z_thresh"])
        ema_period = int(p["ema_period"])
        stretch_atr = float(p["price_stretch_atr"])
        atr_period = int(p["atr_period"])
        atr_mult = float(p["atr_stop_mult"])
        tp_r = float(p["tp_r"])
        max_bars = int(p["max_bars"])

        close = df["close"].astype(float)
        atr = compute_atr(df, atr_period)
        ema = compute_ema(close, ema_period)

        if "funding_rate" in df.columns:
            funding = df["funding_rate"].astype(float).fillna(0.0)
        else:
            funding = pd.Series(0.0, index=df.index, dtype=float)

        min_p = max(15, fund_win // 3)
        f_mean = funding.rolling(fund_win, min_periods=min_p).mean()
        f_std = funding.rolling(fund_win, min_periods=min_p).std(ddof=1)
        # Guard flat/synthetic zero-funding regimes
        fund_z = ((funding - f_mean) / f_std.where(f_std > 1e-8, np.nan)).fillna(0.0)

        stretch = ((close - ema) / atr.where(atr > 0.0, np.nan)).fillna(0.0)

        # Enter on transition into crowded extreme to avoid repeated firing every bar
        short_cond = (fund_z >= z_thresh) & (stretch >= stretch_atr)
        long_cond = (fund_z <= -z_thresh) & (stretch <= -stretch_atr)

        short_entry = short_cond & (~short_cond.shift(1, fill_value=False))
        long_entry = long_cond & (~long_cond.shift(1, fill_value=False))

        sig = np.zeros(len(df), dtype=np.int8)
        sig[long_entry.to_numpy()] = 1
        sig[short_entry.to_numpy()] = -1

        warmup = max(min_p, ema_period, atr_period) + 2
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
