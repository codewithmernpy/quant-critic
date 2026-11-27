"""Independent verification suite with HAND-CALCULATED expected values.

Every expected number below is derived in the comments from first principles, not from the code
under test. Covers engine fills/fees/funding/R, resampling, causal funding attachment, truncation
lookahead on REAL sample data, the locked-test access counter, DSR, PBO, scorecard gating, tool
JSON, and the Claude tool-runner path (mocked transport, no network).
"""

from __future__ import annotations

import json
import math

import anthropic
import httpx2
import numpy as np
import pandas as pd
import pytest

import quantcritic.audit.costs as costs_mod
import quantcritic.audit.overfit as overfit_mod
import quantcritic.audit.walkforward as wf_mod
import quantcritic.engine.backtest as bt_mod
from quantcritic.agent.critic import run_ai_critic
from quantcritic.agent.offline import run_offline_critic
from quantcritic.agent.tools import MAX_TOOL_RESPONSE_CHARS, AuditSession, _clip_json, build_beta_tools
from quantcritic.audit.lookahead import run_truncation_test
from quantcritic.audit.montecarlo import run_random_entry_baseline
from quantcritic.audit.overfit import compute_cscv_pbo, compute_deflated_sharpe_ratio
from quantcritic.audit.scorecard import compute_scorecard
from quantcritic.config import AuditConfig, BacktestConfig
from quantcritic.data.loader import MarketData, attach_causal_funding, load_market_data, resample_ohlcv
from quantcritic.data.synthetic import generate_random_walk_ohlcv, make_ohlcv_from_bars
from quantcritic.engine.backtest import _precompute_bar_funding, run_backtest
from quantcritic.strategies.base import Strategy
from quantcritic.strategies.registry import get_strategy, list_strategies

UTC0 = "2024-01-01 00:00:00+00:00"


def _sigs(df: pd.DataFrame, signal, stop, tp=None, max_bars=None, exit_sig=None) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame(
        {
            "signal": np.asarray(signal, dtype=np.int8),
            "stop_dist": np.asarray(stop if np.ndim(stop) else [stop] * n, dtype=np.float64),
            "tp_r": np.asarray(tp if tp is not None else [np.nan] * n, dtype=np.float64),
            "max_bars": np.asarray(max_bars if max_bars is not None else [0] * n, dtype=np.int32),
            "exit_signal": np.asarray(exit_sig if exit_sig is not None else [False] * n, dtype=bool),
        },
        index=df.index,
    )


def _md(df: pd.DataFrame, funding: pd.Series | None = None, sym: str = "TESTUSDT") -> MarketData:
    return MarketData(ohlcv={sym: df}, funding={sym: funding} if funding is not None else {}, interval="1h")


# ----------------------------------------------------------------------------------------------
# 1. Engine: entry timing, slippage direction, fees, stop fill, R, equity MTM
# ----------------------------------------------------------------------------------------------
def test_engine_long_entry_next_open_slippage_fees_stop_and_R() -> None:
    bars = [
        (100.0, 101.0, 99.0, 100.0),  # b0: signal +1 at close, stop_dist 5
        (100.0, 102.0, 97.0, 101.0),  # b1: entry at OPEN of b1 (not b0 close)
        (101.0, 103.0, 94.0, 96.0),   # b2: low 94 <= stop 95.1 -> stopped at 95.1
        (96.0, 97.0, 95.0, 96.0),
    ]
    df = make_ohlcv_from_bars(bars, start_time=UTC0)
    cfg = BacktestConfig(initial_capital=10_000.0, risk_per_trade=0.01, max_leverage=3.0,
                         fee_rate=0.0004, slippage_bps=10.0, funding_enabled=False, interval="1h")
    res = run_backtest(_md(df), {"TESTUSDT": _sigs(df, [1, 0, 0, 0], 5.0)}, config=cfg)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    # entry = 100 * (1 + 0.0010) = 100.1 ; stop = 100.1 - 5 = 95.1
    # qty = 10_000 * 0.01 / 5 = 20 (notional 2_002 < 3x cap 30_000)
    # exit = 95.1 * (1 - 0.0010) = 95.0049 (adverse slippage on the sell)
    # gross = 20 * (95.0049 - 100.1) = -101.902
    # fees = 20*100.1*0.0004 + 20*95.0049*0.0004 = 0.8008 + 0.7600392 = 1.5608392
    # net = -101.902 - 1.5608392 = -103.4628392 ; R = net / (20*5) = -1.034628392
    assert t["entry_time"] == df.index[1] and t["exit_time"] == df.index[2]
    assert t["entry_price"] == pytest.approx(100.1)
    assert t["exit_price"] == pytest.approx(95.0049)
    assert t["qty"] == pytest.approx(20.0)
    assert t["risk_amount"] == pytest.approx(100.0)
    assert t["gross_pnl"] == pytest.approx(-101.902)
    assert t["fees"] == pytest.approx(1.5608392)
    assert t["net_pnl"] == pytest.approx(-103.4628392)
    assert t["R"] == pytest.approx(-1.034628392)
    assert t["exit_reason"] == "stop" and t["bars_held"] == 2
    # MTM at b1 close: cash 10_000 - 0.8008 = 9_999.1992 ; + 20*(101 - 100.1) = 18 -> 10_017.1992
    assert res.equity_curve.iloc[0] == pytest.approx(10_000.0)
    assert res.equity_curve.iloc[1] == pytest.approx(10_017.1992)
    assert res.equity_curve.iloc[-1] == pytest.approx(10_000.0 - 103.4628392)


def test_engine_gap_through_stop_same_bar_stop_wins_and_take_profit() -> None:
    bars = [
        (100.0, 101.0, 99.0, 100.0),  # b0 signal +1, stop 5, tp 2R -> stop 95, tp 110
        (100.0, 104.0, 96.0, 103.0),  # b1 entry 100, survives
        (90.0, 92.0, 88.0, 91.0),     # b2 opens BELOW stop -> fill at open 90 -> R = (90-100)/5 = -2
        (100.0, 101.0, 99.0, 100.0),  # b3 signal +1
        (100.0, 111.0, 94.0, 105.0),  # b4 entry 100; low<=95 AND high>=110 -> worst case: stop -> R = -1
        (100.0, 101.0, 99.0, 100.0),  # b5 signal +1
        (100.0, 108.0, 97.0, 107.0),  # b6 entry 100, survives
        (107.0, 112.0, 105.0, 111.0),  # b7 high 112 >= tp 110 -> fill at 110 -> R = +2
        (111.0, 112.0, 110.0, 111.0),
    ]
    df = make_ohlcv_from_bars(bars, start_time=UTC0)
    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0, funding_enabled=False, interval="1h")
    sig = _sigs(df, [1, 0, 0, 1, 0, 1, 0, 0, 0], 5.0, tp=[2.0] * 9)
    tr = run_backtest(_md(df), {"TESTUSDT": sig}, config=cfg).trades
    assert list(tr["exit_reason"]) == ["stop", "stop", "take_profit"]
    assert list(tr["exit_price"]) == pytest.approx([90.0, 95.0, 110.0])
    assert list(tr["R"]) == pytest.approx([-2.0, -1.0, 2.0])
    assert list(tr["entry_time"]) == [df.index[1], df.index[4], df.index[6]]


def test_engine_funding_window_sign_and_short_slippage() -> None:
    bars = [(100.0, 100.5, 99.5, 100.0)] * 4  # 00:00..03:00, flat
    df = make_ohlcv_from_bars(bars, start_time=UTC0)
    funding = pd.Series(
        [0.05, 0.0001, 0.07],
        index=pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 03:00"], tz="UTC").as_unit("ns"),
    )
    cfg = BacktestConfig(initial_capital=10_000.0, risk_per_trade=0.01, fee_rate=0.0,
                         slippage_bps=10.0, funding_enabled=True, interval="1h")
    exit_sig = [False, True, False, False]  # decided at b1 close -> exit at b2 open (02:00)
    # Entry 01:00 open, exit 02:00 open. Held over the settlement at 02:00 only:
    #  - 01:00 settlement (0.05) is AT the entry instant -> not charged
    #  - 03:00 settlement (0.07) is after the exit -> not charged
    # qty = 100 / 5 = 20 ; funding cash = -dir * qty * close(b1)=100 * 0.0001 = -dir * 0.2
    long_t = run_backtest(_md(df, funding), {"TESTUSDT": _sigs(df, [1, 0, 0, 0], 5.0, exit_sig=exit_sig)}, config=cfg).trades.iloc[0]
    # long: entry 100*1.001 = 100.1 ; exit 100*0.999 = 99.9 ; gross 20*(99.9-100.1) = -4.0
    assert long_t["entry_price"] == pytest.approx(100.1) and long_t["exit_price"] == pytest.approx(99.9)
    assert long_t["funding"] == pytest.approx(-0.2)
    assert long_t["net_pnl"] == pytest.approx(-4.2) and long_t["R"] == pytest.approx(-0.042)
    short_t = run_backtest(_md(df, funding), {"TESTUSDT": _sigs(df, [-1, 0, 0, 0], 5.0, exit_sig=exit_sig)}, config=cfg).trades.iloc[0]
    # short: entry 100*(1-0.001) = 99.9 (sell lower) ; exit 100*(1+0.001) = 100.1 (buy higher)
    # gross = -1 * 20 * (100.1 - 99.9) = -4.0 ; funding +0.2 (shorts receive positive funding)
    assert short_t["entry_price"] == pytest.approx(99.9) and short_t["exit_price"] == pytest.approx(100.1)
    assert short_t["funding"] == pytest.approx(0.2)
    assert short_t["net_pnl"] == pytest.approx(-3.8) and short_t["R"] == pytest.approx(-0.038)
    assert short_t["exit_reason"] == "exit_signal" and short_t["exit_time"] == df.index[2]


def test_engine_leverage_cap_time_stop_and_end_of_data() -> None:
    flat = [(100.0, 100.05, 99.95, 100.0)] * 6  # range never reaches a 0.1 stop (99.9)
    a = make_ohlcv_from_bars(flat, start_time=UTC0)
    b = make_ohlcv_from_bars(flat, start_time=UTC0)
    md = MarketData(ohlcv={"AAAUSDT": a, "BBBUSDT": b}, funding={}, interval="1h")
    cfg = BacktestConfig(initial_capital=10_000.0, risk_per_trade=0.01, max_leverage=3.0,
                         fee_rate=0.0, slippage_bps=0.0, funding_enabled=False, interval="1h")
    sig = [1, 0, 0, 0, 0, 0]
    res = run_backtest(md, {"AAAUSDT": _sigs(a, sig, 0.1, max_bars=[2] * 6), "BBBUSDT": _sigs(b, sig, 0.1)}, config=cfg)
    # raw qty = 100 / 0.1 = 1_000 -> notional 100_000 > 3 x 10_000 -> qty = 30_000/100 = 300, risk 300*0.1 = 30
    # BBB on the same bar: available notional = 30_000 - 30_000 = 0 -> no position
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["symbol"] == "AAAUSDT" and t["qty"] == pytest.approx(300.0) and t["risk_amount"] == pytest.approx(30.0)
    # max_bars=2: entry b1, survives b1 and b2 -> time stop at b3 OPEN
    assert t["exit_reason"] == "time_stop" and t["exit_time"] == a.index[3]

    # end of data: open position closed at the final close with exit slippage
    cfg2 = cfg.model_copy(update={"slippage_bps": 10.0})
    res2 = run_backtest(_md(a), {"TESTUSDT": _sigs(a, sig, 5.0)}, config=cfg2)
    t2 = res2.trades.iloc[0]
    # exit = 100 * (1 - 0.001) = 99.9 at the last bar timestamp
    assert t2["exit_reason"] == "end_of_data" and t2["exit_price"] == pytest.approx(99.9)
    assert t2["exit_time"] == a.index[-1]
    assert res2.equity_curve.iloc[-1] == pytest.approx(10_000.0 + t2["net_pnl"])


def test_engine_trade_start_window_uses_warm_signal_and_slices_equity() -> None:
    bars = [(100.0, 100.5, 99.5, 100.0)] * 6
    df = make_ohlcv_from_bars(bars, start_time=UTC0)
    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0, funding_enabled=False, interval="1h")
    # b0 signal would enter at b1 (before the window) -> must be ignored;
    # b2 signal (last bar BEFORE the window) enters at b3 = first window bar.
    sig = _sigs(df, [1, 0, 1, 0, 0, 0], 5.0)
    res = run_backtest(_md(df), {"TESTUSDT": sig}, config=cfg, trade_start=df.index[3])
    assert len(res.trades) == 1 and res.trades.iloc[0]["entry_time"] == df.index[3]
    assert list(res.equity_curve.index) == list(df.index[3:])
    assert res.metrics["n_trades"] == 1


# ----------------------------------------------------------------------------------------------
# 2. Resampling and causal funding attachment
# ----------------------------------------------------------------------------------------------
def test_resample_hand_built_bars_with_gap() -> None:
    hours = [0, 1, 2, 3, 4, 6, 7, 8]  # 05:00 missing (exchange gap)
    idx = pd.DatetimeIndex([pd.Timestamp(UTC0) + pd.Timedelta(hours=h) for h in hours]).as_unit("ns")
    o = np.array([10.0 + h for h in hours])
    df = pd.DataFrame({"open": o, "high": o + 2.0, "low": o - 1.0, "close": o + 0.5,
                       "volume": np.array([h + 1.0 for h in hours])}, index=idx)
    r4 = resample_ohlcv(df, "4h")
    # 00:00 bar = hours 0-3: O=10, H=max(12..15)=15, L=min(9..12)=9, C=13.5, V=1+2+3+4=10
    # 04:00 bar = hours 4,6,7: O=14, H=max(16,18,19)=19, L=min(13,15,16)=13, C=17.5, V=5+7+8=20
    # 08:00 bar = hour 8 only (never merged into the 04:00 bar): O=18, H=20, L=17, C=18.5, V=9
    assert list(r4.index.hour) == [0, 4, 8]
    assert r4.iloc[0][["open", "high", "low", "close", "volume"]].tolist() == [10.0, 15.0, 9.0, 13.5, 10.0]
    assert r4.iloc[1][["open", "high", "low", "close", "volume"]].tolist() == [14.0, 19.0, 13.0, 17.5, 20.0]
    assert r4.iloc[2][["open", "high", "low", "close", "volume"]].tolist() == [18.0, 20.0, 17.0, 18.5, 9.0]
    r1d = resample_ohlcv(df, "1d")
    # 1d (label 00:00): O=10, H=20, L=9, C=18.5, V=45-6=39 (hour 5, volume 6, missing)
    assert len(r1d) == 1 and r1d.index[0] == pd.Timestamp(UTC0)
    assert r1d.iloc[0][["open", "high", "low", "close", "volume"]].tolist() == [10.0, 20.0, 9.0, 18.5, 39.0]


def test_funding_attachment_and_accrual_windows_are_causal() -> None:
    idx = pd.date_range(UTC0, periods=4, freq="4h", tz="UTC").as_unit("ns")  # 00,04,08,12
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    f = pd.Series([0.001, 0.002, 0.003],
                  index=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00"], tz="UTC").as_unit("ns"))
    feat = attach_causal_funding(df, f, "4h")["funding_rate"].tolist()
    # feature at bar t = last settlement strictly BEFORE the bar close:
    # 00:00 bar (close 04:00) -> 0.001 ; 04:00 bar (close 08:00) -> 0.001 (08:00 not yet < 08:00)
    # 08:00 bar (close 12:00) -> 0.002 ; 12:00 bar (close 16:00) -> 0.002
    assert feat == [0.001, 0.001, 0.002, 0.002]
    acc = _precompute_bar_funding(idx, f, "4h").tolist()
    # accrual window (open, next open]: 00->(00,04]=0 ; 04->(04,08]=0.002 ; 08->(08,12]=0 ; 12->(12,16]=0.003
    assert acc == pytest.approx([0.0, 0.002, 0.0, 0.003])


# ----------------------------------------------------------------------------------------------
# 3. Truncation lookahead test on REAL sample data (1h and 4h)
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=["1h", "4h"])
def real_market(request) -> MarketData:
    return load_market_data(["BTCUSDT", "ETHUSDT", "SOLUSDT"], interval=request.param)


def test_truncation_real_data_catches_leaky_and_clears_honest(real_market: MarketData) -> None:
    leaky = run_truncation_test(real_market, get_strategy("leaky_demo"), n_cuts=25, window=100, seed=11)
    assert leaky["passed"] is False and leaky["total_mismatches"] > 0
    assert leaky["cuts_checked"] == 75
    for name in list_strategies(include_leaky=False):
        res = run_truncation_test(real_market, get_strategy(name), n_cuts=25, window=100, seed=11)
        assert res["passed"] is True, (real_market.interval, name, res["examples"][:3])
        assert res["cuts_checked"] == 75 and res["total_mismatches"] == 0


# ----------------------------------------------------------------------------------------------
# 4. Locked TEST slice: never read before final_test, read exactly once
# ----------------------------------------------------------------------------------------------
class _SpyStrategy(Strategy):
    def __init__(self, inner: Strategy, log: list) -> None:
        self._inner, self._log = inner, log
        self.name, self.description = inner.name, inner.description
        self.default_params, self.param_grid = dict(inner.default_params), dict(inner.param_grid)
        self.session: AuditSession | None = None

    def is_valid_params(self, params):  # noqa: ANN001
        return self._inner.is_valid_params(params)

    def generate(self, df, params=None):  # noqa: ANN001
        if len(df):
            self._log.append(("generate", df.index.max(), bool(self.session and self.session.final_test_called)))
        return self._inner.generate(df, params)


class _SpyData:
    """Full-history MarketData proxy that counts content reads made before final_test."""

    def __init__(self, md: MarketData, session: AuditSession) -> None:
        self._md, self._session, self.early_reads = md, session, 0

    @property
    def interval(self) -> str:
        return self._md.interval

    @property
    def symbols(self) -> list[str]:
        return list(self._md.ohlcv.keys())

    def _touch(self) -> None:
        if not self._session.final_test_called:
            self.early_reads += 1

    @property
    def ohlcv(self):
        self._touch()
        return self._md.ohlcv

    @property
    def funding(self):
        self._touch()
        return self._md.funding


def test_locked_test_is_never_touched_before_final_test(monkeypatch) -> None:
    md = load_market_data(["BTCUSDT"], interval="1d")
    log: list = []
    spy = _SpyStrategy(get_strategy("donchian_breakout"), log)
    session = AuditSession(strategy=spy, data=md, config=BacktestConfig(interval="1d"),
                           audit_config=AuditConfig(lookahead_cuts=5, mc_simulations=50, random_baseline_runs=20))
    spy.session = session
    session.data = _SpyData(md, session)  # type: ignore[assignment]

    real_prepare = bt_mod.prepare_market

    def _spy_prepare(data):  # noqa: ANN001
        ts = [df.index.max() for df in data.ohlcv.values() if len(df)]
        if ts:
            log.append(("prepare_market", max(ts), session.final_test_called))
        return real_prepare(data)

    for mod in (bt_mod, wf_mod, overfit_mod, costs_mod):
        monkeypatch.setattr(mod, "prepare_market", _spy_prepare)

    # Run every pre-test tool (in the "wrong" AI order too) and check nothing saw a test bar
    for fn in (session.get_overview, session.run_lookahead, session.run_walkforward, session.run_overfit,
               session.run_monte_carlo, session.run_random_baseline, session.run_cost_stress,
               session.run_regime_breakdown, session.run_prop_firm):
        fn()
    assert "error" in session.run_backtest_period("{}", "test")
    assert session.run_backtest_period("{}", "validation")["period"] == "validation"
    test_start = session.splits.test_start
    pre = [(kind, ts) for kind, ts, after in log if not after]
    assert pre and all(ts < test_start for _, ts in pre), [p for p in pre if p[1] >= test_start][:3]
    assert session.test_access_count == 0 and session.data.early_reads == 0

    ft = session.run_final_test("{}")
    assert ft["status"] == "evaluated_once" and session.test_access_count == 1
    assert any(after and ts >= test_start for _, ts, after in log)  # spy really observes test reads
    assert "error" in session.run_final_test("{}") and session.test_access_count == 1
    test_trades = session.cached_results["final_test_result"].trades
    assert (pd.to_datetime(test_trades["entry_time"]) >= test_start).all()


# ----------------------------------------------------------------------------------------------
# 5. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) - hand example
# ----------------------------------------------------------------------------------------------
def test_dsr_matches_hand_computation() -> None:
    r = np.array([0.02, -0.01] * 5)  # T = 10
    # mean = 0.005 ; deviations +-0.015 -> sum sq = 10*0.000225 = 0.00225 ; var = 0.00225/9 = 0.00025
    # std = 0.0158113883 ; SR = 0.005/0.0158113883 = 0.316227766 (= 1/sqrt(10))
    # skew = 0 (symmetric two-point sample)
    # biased g2 = m4/m2^2 - 3 = 1 - 3 = -2 ; unbiased G2 = ((n+1)g2 + 6)(n-1)/((n-2)(n-3))
    #   = (11*-2 + 6)*9/(8*7) = -144/56 = -2.5714286 ; Pearson kurtosis = G2 + 3 = 0.4285714
    # sigma_SR = sqrt((1 - 0*SR + (0.4285714-1)/4 * SR^2)/(T-1)) = sqrt((1 - 0.0142857)/9) = 0.3309438
    sr, sig = 0.316227766, 0.3309438
    one = compute_deflated_sharpe_ratio(r, n_trials=1)
    # N=1 -> SR0 = 0 -> DSR = Phi(0.316227766/0.3309438) = Phi(0.9555331) = 0.83035
    assert one["sr_observed"] == pytest.approx(0.3162, abs=1e-4)
    assert one["kurtosis"] == pytest.approx(0.4286, abs=1e-4)
    assert one["sr_benchmark"] == 0.0
    assert one["dsr"] == pytest.approx(0.8303, abs=2e-4)
    # N=10 trials with trial SRs [0.1,0.2,0.3] -> V = 0.01 -> sqrt V = 0.1
    # SR0 = 0.1*((1-0.5772157)*Phi^-1(0.9) + 0.5772157*Phi^-1(1-1/(10e)))
    #     = 0.1*(0.4227843*1.2815516 + 0.5772157*1.7892418) = 0.1*(0.5418210 + 1.0327776) = 0.1574599
    # z = (0.3162278 - 0.1574599)/0.3309438 = 0.4797429 -> DSR = Phi(0.4797) = 0.68429
    ten = compute_deflated_sharpe_ratio(r, n_trials=10, trial_sharpes=np.array([0.1, 0.2, 0.3]))
    assert ten["sr_benchmark"] == pytest.approx(0.1575, abs=1e-4)
    assert ten["dsr"] == pytest.approx(0.6843, abs=2e-4)
    assert math.isclose(sr / sig, 0.9555331, rel_tol=1e-5)


# ----------------------------------------------------------------------------------------------
# 6. PBO via CSCV
# ----------------------------------------------------------------------------------------------
def test_pbo_noise_about_half_and_hand_built_extremes() -> None:
    pbos = [compute_cscv_pbo(np.random.default_rng(s).normal(0, 0.01, size=(500, 10)), n_blocks=10)["pbo"]
            for s in range(8)]
    assert 0.35 <= float(np.mean(pbos)) <= 0.65  # pure noise -> PBO ~ 0.5

    wiggle = np.array([0.1, -0.1, 0.05, -0.05, 0.0])  # zero-mean within-block noise
    # A has block means [3,1,-2,-2] (sum 0) and B = -A. For any half split, IS-mean(A) = -OOS-mean(A),
    # so the IS winner is ALWAYS the OOS loser: rank 1 of 2 -> w = 1/3 -> logit < 0 in all 6 splits -> PBO 1
    a = np.concatenate([m + wiggle for m in (3.0, 1.0, -2.0, -2.0)])
    worst = compute_cscv_pbo(np.column_stack([a, -a]), n_blocks=4)
    # C(4,2) = 6 splits ; logit = ln((1/3)/(2/3)) = ln(0.5) = -0.6931
    assert worst["pbo"] == 1.0 and worst["n_splits"] == 6
    assert worst["logits"] == [round(math.log(0.5), 4)] * 6
    # A positive in every block, B zero-mean: A wins IS and OOS every time -> w = 2/3 -> PBO 0
    good = np.concatenate([1.0 + wiggle] * 4)
    noise = np.concatenate([wiggle] * 4)
    assert compute_cscv_pbo(np.column_stack([good, noise]), n_blocks=4)["pbo"] == 0.0


# ----------------------------------------------------------------------------------------------
# 7. Scorecard: a strategy that loses on the locked test can never be PROMISING
# ----------------------------------------------------------------------------------------------
def _sc_inputs(test=(1.0, 0.20, 0.30, 50), val=(1.0, 0.30), wf=1.0, trunc=True):
    t_sh, t_ret, t_exp, t_n = test
    return dict(
        lookahead_res={"status": "pass" if trunc else "fail", "truncation": {"passed": trunc},
                       "delay": {"expectancy_collapse_pct": 10.0, "warn": False}},
        walkforward_res={"validation_metrics": {"sharpe": val[0], "expectancy_r": val[1]},
                         "rolling_oos_metrics": {"sharpe": wf}},
        overfit_res={"dsr": {"dsr": 0.9}, "pbo": {"pbo": 0.2}, "stability": {"stability_ratio": 0.8}},
        montecarlo_res={"random_baseline": {"p_value": 0.01},
                        "bootstrap": {"p95_max_drawdown": 0.15, "prob_loss": 0.05}},
        cost_res={"break_even_cost_bps": 40.0, "stress_2x_5bps_expectancy_r": 0.2, "base_expectancy_r": 0.25},
        regime_res={"yearly": [{"return": 0.1}] * 4,
                    "regimes": {k: {"n_trades": 5, "net_pnl": 10.0} for k in ("bull", "bear", "chop")},
                    "outlier_check": {"still_profitable": True}},
        full_metrics={"max_drawdown": 0.12},
        test_metrics={"sharpe": t_sh, "total_return": t_ret, "expectancy_r": t_exp, "n_trades": t_n},
        trades=pd.DataFrame({"R": [2.0, -1.0, 1.5, -1.0, 2.5, -1.0]}),
    )


def test_scorecard_gates_and_monotonicity() -> None:
    good = compute_scorecard(**_sc_inputs())
    # locked_test sub-score: clip((1.0+0.2)/1.6)*60 = 0.75*60 = 45 ; clip((0.30+0.05)/0.30)*40 = 40 -> 85
    assert good["sub_scores"]["locked_test"] == 85.0
    # integrity: 100 - 10*0.35 = 96.5
    assert good["sub_scores"]["integrity"] == 96.5
    assert good["verdict"] == "PROMISING" and good["gates"] == []

    few = compute_scorecard(**_sc_inputs(test=(1.0, 0.20, 0.30, 5)))
    assert few["sub_scores"]["locked_test"] == 42.5  # 85 * 5/10 trades
    assert few["verdict"] != "PROMISING"

    lost = compute_scorecard(**_sc_inputs(test=(-0.59, -0.11, -0.14, 150)))
    assert lost["verdict"] == "NEEDS WORK" and lost["overall_score"] <= 55.0
    assert "test_fail" in [g["gate"] for g in lost["gates"]]

    both = compute_scorecard(**_sc_inputs(test=(-0.5, -0.1, -0.1, 80), val=(-0.3, -0.05), wf=-0.2))
    assert both["verdict"] == "REJECT" and both["overall_score"] <= 40.0

    cheat = compute_scorecard(**_sc_inputs(trunc=False))
    assert cheat["verdict"] == "REJECT" and cheat["overall_score"] <= 20.0

    prev_score, prev_rank = -1.0, -1
    rank = {"REJECT": 0, "NEEDS WORK": 1, "PROMISING": 2}
    for sh in np.linspace(-1.5, 2.0, 15):
        sc = compute_scorecard(**_sc_inputs(test=(sh, 0.1 * sh, 0.2 * sh, 60)))
        assert sc["overall_score"] >= prev_score - 1e-9 and rank[sc["verdict"]] >= prev_rank
        if sh <= 0:
            assert sc["verdict"] != "PROMISING"
        prev_score, prev_rank = sc["overall_score"], rank[sc["verdict"]]


# ----------------------------------------------------------------------------------------------
# 8. Random-entry baseline charges funding like the engine
# ----------------------------------------------------------------------------------------------
def test_random_baseline_includes_funding() -> None:
    df = generate_random_walk_ohlcv(n_bars=600, interval="4h", seed=5)
    f = pd.Series(0.001, index=pd.date_range(df.index[0], df.index[-1], freq="8h", tz="UTC").as_unit("ns"))
    md = MarketData(ohlcv={"SYNUSDT": df}, funding={"SYNUSDT": f}, interval="4h")
    sig = np.zeros(len(df), dtype=np.int8)
    sig[30::20] = 1  # long-only
    s = _sigs(df, sig, 3.0, max_bars=[6] * len(df))
    out = {}
    for on in (True, False):
        cfg = BacktestConfig(interval="4h", funding_enabled=on)
        real = run_backtest(md, {"SYNUSDT": s}, config=cfg)
        out[on] = run_random_entry_baseline(md, real, {"atr_period": 14, "atr_stop_mult": 2.0}, n_runs=50, seed=1)
    # identical random draws; with +0.1%/8h funding every random LONG pays -> lower mean expectancy
    assert out[True]["random_mean_expectancy_r"] < out[False]["random_mean_expectancy_r"]


# ----------------------------------------------------------------------------------------------
# 9. Tools: compact valid JSON, correct numbers, lock + single final_test + verdict capture
# ----------------------------------------------------------------------------------------------
def test_clip_json_always_valid_and_bounded() -> None:
    payload = {"a": float("nan"), "b": [{"x": i * 1.123456789, "y": "z" * 40} for i in range(2000)], "c": np.int64(3)}
    out = _clip_json(payload)
    parsed = json.loads(out)
    assert len(out) <= MAX_TOOL_RESPONSE_CHARS and parsed["a"] is None and parsed["c"] == 3


def test_every_tool_returns_compact_correct_json_and_enforces_lock() -> None:
    md = load_market_data(["BTCUSDT"], interval="1d")
    session = AuditSession(strategy=get_strategy("ema_trend"), data=md, config=BacktestConfig(interval="1d"),
                           audit_config=AuditConfig(lookahead_cuts=5, mc_simulations=50, random_baseline_runs=20))
    tools = {t.name: t for t in build_beta_tools(session)}
    assert len(tools) == 12

    def call(name: str, **kw) -> dict:
        out = tools[name].call(kw)
        assert isinstance(out, str) and len(out) < 3000, (name, len(out))
        return json.loads(out)

    wf = call("walk_forward")
    assert wf["validation"]["sharpe"] == session.cached_results["walkforward"]["validation_metrics"]["sharpe"]
    ov = call("get_overview")
    assert ov["splits"]["test_locked"] is True and ov["strategy"] == "ema_trend"
    la = call("lookahead_test")
    assert la["truncation"]["passed"] is True and la["truncation"]["cuts_checked"] == 5
    assert call("overfit_tests")["n_trials"] == len(get_strategy("ema_trend").expand_param_grid())
    mc = call("monte_carlo")
    assert mc["bootstrap"]["n_simulations"] == 50
    rb = call("random_baseline")
    assert rb["random_baseline"]["n_runs"] == 20
    assert 0 < rb["random_baseline"]["p_value"] <= 1
    cs = call("cost_stress")
    assert len(cs["expectancy_matrix"]) == 4 and len(cs["expectancy_matrix"][0]) == 4
    assert "yearly" in call("regime_breakdown")
    assert "pass_rate" in call("prop_firm_check")
    assert "error" in call("run_backtest", params_json="{}", period="test")
    exp = call("run_backtest", params_json='{"fast_span": 10}', period="validation")
    assert exp["params"]["fast_span"] == 10 and exp["experiment_number"] == 1
    assert session.test_access_count == 0

    ft = call("final_test", params_json="{}")
    assert ft["status"] == "evaluated_once" and session.test_access_count == 1
    assert ft["test_metrics"]["sharpe"] == session.cached_results["final_test_result"].metrics["sharpe"]
    assert "error" in call("final_test", params_json="{}") and session.test_access_count == 1
    assert "frozen" in call("run_backtest", params_json="{}", period="train")["error"].lower()

    sv = call("submit_verdict", verdict="PROMISING", score=95.0, summary="x", strengths=["s"], weaknesses=["w"],
              red_flags=[], improvements=["i"], recommended_risk_per_trade=0.005)
    assert sv["recorded"] is True and session.verdict is not None and session.verdict.source == "ai"
    # Hard gates once the scorecard exists: ema_trend 1d loses on the test slice -> cannot stay PROMISING
    session.ensure_complete_audit()
    session.enforce_hard_gates()
    assert session.cached_results["scorecard"]["verdict_ceiling"] != "PROMISING"
    assert session.verdict.verdict != "PROMISING" and session.verdict.score <= 69.9


# ----------------------------------------------------------------------------------------------
# 10. Claude tool runner: exact SDK call, real tool execution through the SDK (mock transport),
#     refusal and API errors fall back to the offline critic. No network, no API key.
# ----------------------------------------------------------------------------------------------
def _small_session(strategy: str = "ema_trend") -> AuditSession:
    md = load_market_data(["BTCUSDT"], interval="1d")
    return AuditSession(strategy=get_strategy(strategy), data=md, config=BacktestConfig(interval="1d"),
                        audit_config=AuditConfig(lookahead_cuts=5, mc_simulations=50, random_baseline_runs=20))


def _mock_client(script: list, captured: list) -> anthropic.Anthropic:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        step = script[min(len(captured) - 1, len(script) - 1)]
        if isinstance(step, int):
            return httpx2.Response(step, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})
        if step in ("END", "REFUSE"):
            content, stop = [{"type": "text", "text": "Done."}], ("end_turn" if step == "END" else "refusal")
        else:
            content = [{"type": "tool_use", "id": f"toolu_{len(captured)}_{i}", "name": n, "input": a}
                       for i, (n, a) in enumerate(step)]
            stop = "tool_use"
        return httpx2.Response(200, json={
            "id": f"msg_{len(captured)}", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    return anthropic.Anthropic(api_key="sk-test", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def _patch_client(monkeypatch, client: anthropic.Anthropic, runner_kwargs: list) -> None:
    real_runner = client.beta.messages.tool_runner

    def _capturing_runner(**kwargs):
        runner_kwargs.append(kwargs)
        return real_runner(**kwargs)

    monkeypatch.setattr(client.beta.messages, "tool_runner", _capturing_runner)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_kw: client)
    monkeypatch.delenv("QC_OFFLINE", raising=False)


def test_ai_tool_runner_exact_call_and_gated_verdict(monkeypatch) -> None:
    captured: list = []
    runner_kwargs: list = []
    verdict_args = {"verdict": "PROMISING", "score": 91, "summary": "ok", "strengths": ["a"], "weaknesses": [],
                    "red_flags": [], "improvements": ["i"], "recommended_risk_per_trade": 0.004}
    script = [
        [("get_overview", {}), ("lookahead_test", {}), ("walk_forward", {})],
        [("run_backtest", {"params_json": "{}", "period": "test"}), ("final_test", {"params_json": "{}"})],
        [("final_test", {"params_json": "{}"}), ("submit_verdict", verdict_args)],
        "END",
    ]
    _patch_client(monkeypatch, _mock_client(script, captured), runner_kwargs)
    session = _small_session()
    v = run_ai_critic(session, use_ai=True, api_key="sk-test")

    kw = runner_kwargs[0]
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 16000 and kw["max_iterations"] == 24
    assert kw["output_config"] == {"effort": "high"} and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert isinstance(kw["system"], str) and "final_test" in kw["system"]
    assert len(kw["tools"]) == 12 and kw["messages"][0]["role"] == "user"
    assert "server-side-fallback-2026-07-01" in captured[0]["headers"].get("anthropic-beta", "")
    assert len(captured) == 4

    results = [b for c in captured[1:] for m in c["body"]["messages"][-1:] if isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"]
    texts = [b["content"] if isinstance(b["content"], str) else b["content"][0]["text"] for b in results]
    assert all(len(t) < 3000 for t in texts) and all(isinstance(json.loads(t), dict) for t in texts)
    assert sum('"evaluated_once"' in t for t in texts) == 1
    assert sum("already been executed" in t for t in texts) == 1
    assert any("test slice is locked" in t for t in texts)
    assert session.test_access_count == 1
    # AI said PROMISING 91, but the 1d locked test is lost -> hard gate caps it
    assert v.source == "ai" and v.verdict != "PROMISING" and v.score <= 69.9
    assert any("Hard gate" in rf for rf in v.red_flags)


@pytest.mark.parametrize("failure", ["refusal", 500, 401])
def test_ai_refusal_and_api_errors_fall_back_offline(monkeypatch, failure) -> None:
    captured: list = []
    script = [[("get_overview", {})], "REFUSE"] if failure == "refusal" else [failure]
    _patch_client(monkeypatch, _mock_client(script, captured), [])
    events: list = []
    session = _small_session()
    v = run_ai_critic(session, use_ai=True, api_key="sk-test", on_event=lambda k, m: events.append((k, m)))
    assert v.source == "offline" and v.verdict in ("PROMISING", "NEEDS WORK", "REJECT")
    assert session.test_access_count == 1 and "scorecard" in session.cached_results
    assert any(k == "system" for k, _ in events)


def test_offline_verdict_for_negative_test_is_not_promising() -> None:
    session = _small_session("donchian_breakout")
    v = run_offline_critic(session)
    test_m = session.cached_results["final_test"]["test_metrics"]
    # BTC 1d donchian loses on the bundled locked test (2024-11 .. 2025-10)
    assert test_m["sharpe"] < 0 and test_m["total_return"] < 0
    assert v.verdict != "PROMISING" and v.score <= 55.0
    assert v.score == session.cached_results["scorecard"]["overall_score"]
    assert any("test_fail" in rf for rf in v.red_flags)
