"""Strict unit tests for the event-driven crypto-perpetuals backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcritic.config import BacktestConfig
from quantcritic.data.loader import MarketData
from quantcritic.data.synthetic import make_ohlcv_from_bars
from quantcritic.engine.backtest import run_backtest


def _sig_df(
    df: pd.DataFrame,
    signals: list[int],
    stop_dists: list[float],
    tp_rs: list[float] | None = None,
    max_bars: list[int] | None = None,
    exit_signals: list[bool] | None = None,
) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame(
        {
            "signal": np.asarray(signals, dtype=np.int8),
            "stop_dist": np.asarray(stop_dists, dtype=np.float64),
            "tp_r": np.asarray(tp_rs if tp_rs is not None else [np.nan] * n, dtype=np.float64),
            "max_bars": np.asarray(max_bars if max_bars is not None else [0] * n, dtype=np.int32),
            "exit_signal": np.asarray(exit_signals if exit_signals is not None else [False] * n, dtype=bool),
        },
        index=df.index,
    )


def test_exact_entry_next_open_plus_slippage_and_stop_fill() -> None:
    # Bar 0: signal +1 at close, stop_dist=10
    # Bar 1: open=100 -> entry=100*(1+0.001)=100.1, stop=90.1, low=95 (stays open)
    # Bar 2: open=98, high=102, low=89 -> hits stop at 90.1 -> exit=90.1*(1-0.001)
    bars = [
        (99.0, 101.0, 98.0, 100.0),
        (100.0, 103.0, 95.0, 101.0),
        (98.0, 102.0, 89.0, 91.0),
        (91.0, 92.0, 90.0, 91.5),
    ]
    df = make_ohlcv_from_bars(bars, interval="1h")
    sigs = _sig_df(df, [1, 0, 0, 0], [10.0, 10.0, 10.0, 10.0])
    md = MarketData(ohlcv={"TESTUSDT": df}, funding={}, interval="1h")
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        max_leverage=3.0,
        fee_rate=0.001,
        slippage_bps=10.0,
        funding_enabled=False,
        interval="1h",
    )
    res = run_backtest(md, {"TESTUSDT": sigs}, config=cfg)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]

    expected_entry = 100.0 * 1.001
    expected_stop = expected_entry - 10.0
    expected_exit = expected_stop * 0.999
    expected_qty = (10_000.0 * 0.01) / 10.0  # 10.0 units (notional 1001 <= 30000)
    expected_fees = expected_qty * expected_entry * 0.001 + expected_qty * expected_exit * 0.001
    expected_gross = expected_qty * (expected_exit - expected_entry)
    expected_net = expected_gross - expected_fees

    assert tr["entry_time"] == df.index[1]
    assert tr["exit_time"] == df.index[2]
    assert tr["exit_reason"] == "stop"
    assert tr["entry_price"] == pytest.approx(expected_entry)
    assert tr["exit_price"] == pytest.approx(expected_exit)
    assert tr["fees"] == pytest.approx(expected_fees)
    assert tr["net_pnl"] == pytest.approx(expected_net)
    assert tr["R"] == pytest.approx(expected_net / 100.0)
    assert res.equity_curve.iloc[-1] == pytest.approx(10_000.0 + expected_net)


def test_gap_through_stop_fills_at_open_and_stop_beats_tp_same_bar() -> None:
    # Trade 1: Gap through stop at bar 2 open (open=85 < stop=90) -> fills at 85
    # Trade 2: Bar 4 touches BOTH stop (low=85 <= 90) and TP (high=125 >= 120) -> stop wins
    bars = [
        (100.0, 101.0, 99.0, 100.0),  # 0: signal +1, stop_dist=10, tp_r=2 -> stop=90, tp=120
        (100.0, 102.0, 95.0, 101.0),  # 1: entry at 100
        (85.0, 92.0, 84.0, 88.0),     # 2: gaps below stop=90 -> fills at open 85
        (99.0, 100.0, 98.0, 100.0),   # 3: signal +1
        (100.0, 125.0, 85.0, 110.0),  # 4: entry at 100, touches BOTH stop=90 and tp=120 -> stop=90
    ]
    df = make_ohlcv_from_bars(bars, interval="1h")
    sigs = _sig_df(df, [1, 0, 0, 1, 0], [10.0] * 5, tp_rs=[2.0] * 5)
    md = MarketData(ohlcv={"TESTUSDT": df}, funding={}, interval="1h")
    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0, funding_enabled=False, interval="1h")

    res = run_backtest(md, {"TESTUSDT": sigs}, config=cfg)
    assert len(res.trades) == 2
    assert res.trades.iloc[0]["exit_price"] == pytest.approx(85.0)
    assert res.trades.iloc[0]["exit_reason"] == "stop"
    assert res.trades.iloc[1]["exit_price"] == pytest.approx(90.0)
    assert res.trades.iloc[1]["exit_reason"] == "stop"


def test_funding_sign_long_pays_short_receives() -> None:
    # Two identical price flat scenarios with positive funding at bar 2 open (02:00)
    bars = [
        (100.0, 100.5, 99.5, 100.0),  # 00:00 signal
        (100.0, 100.5, 99.5, 100.0),  # 01:00 entry
        (100.0, 100.5, 99.5, 100.0),  # 02:00 exit at open due to exit_signal on bar 1
        (100.0, 100.5, 99.5, 100.0),  # 03:00
    ]
    df = make_ohlcv_from_bars(bars, interval="1h", start_time="2024-01-01 00:00:00+00:00")
    funding = pd.Series(
        [0.01],  # +1% funding at 02:00 (end of bar 1)
        index=pd.DatetimeIndex(["2024-01-01 02:00:00+00:00"]).as_unit("ns"),
    )
    md = MarketData(ohlcv={"TESTUSDT": df}, funding={"TESTUSDT": funding}, interval="1h")
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        risk_per_trade=0.01,
        fee_rate=0.0,
        slippage_bps=0.0,
        funding_enabled=True,
        interval="1h",
    )

    # Long trade: pays positive funding -> funding < 0
    long_sig = _sig_df(df, [1, 0, 0, 0], [10.0] * 4, exit_signals=[False, True, False, False])
    res_long = run_backtest(md, {"TESTUSDT": long_sig}, config=cfg)
    assert len(res_long.trades) == 1
    assert res_long.trades.iloc[0]["funding"] < 0.0
    assert res_long.trades.iloc[0]["funding"] == pytest.approx(-10.0 * 100.0 * 0.01)

    # Short trade: receives positive funding -> funding > 0
    short_sig = _sig_df(df, [-1, 0, 0, 0], [10.0] * 4, exit_signals=[False, True, False, False])
    res_short = run_backtest(md, {"TESTUSDT": short_sig}, config=cfg)
    assert len(res_short.trades) == 1
    assert res_short.trades.iloc[0]["funding"] > 0.0
    assert res_short.trades.iloc[0]["funding"] == pytest.approx(10.0 * 100.0 * 0.01)


def test_no_entry_on_last_bar_and_one_position_per_symbol() -> None:
    bars = [
        (100.0, 101.0, 99.0, 100.0),  # 0: signal +1 -> enters bar 1
        (100.0, 101.0, 99.0, 100.0),  # 1: signal +1 (must NOT open a second position!)
        (100.0, 101.0, 99.0, 100.0),  # 2: signal 0
        (100.0, 101.0, 99.0, 100.0),  # 3: signal +1 on LAST bar (must NOT enter)
    ]
    df = make_ohlcv_from_bars(bars, interval="1h")
    # First test: signal ONLY on last bar -> 0 trades
    last_only = _sig_df(df, [0, 0, 0, 1], [10.0] * 4)
    md = MarketData(ohlcv={"TESTUSDT": df}, funding={}, interval="1h")
    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0, funding_enabled=False, interval="1h")
    assert len(run_backtest(md, {"TESTUSDT": last_only}, config=cfg).trades) == 0

    # Second test: repeated +1 signals -> still only 1 position opened
    repeated = _sig_df(df, [1, 1, 0, 0], [10.0] * 4)
    res_rep = run_backtest(md, {"TESTUSDT": repeated}, config=cfg)
    assert len(res_rep.trades) == 1
