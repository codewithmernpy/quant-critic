"""Tests for No-Code Strategy Builder, custom plugins, NL parser, and Verdict Strictness modes."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import quantcritic.strategies.builder as builder_mod
import quantcritic.strategies.registry as registry_mod
from quantcritic.__main__ import main as cli_main
from quantcritic.agent.offline import run_offline_critic, synthesize_verdict_from_session
from quantcritic.agent.tools import AuditSession
from quantcritic.audit.lookahead import run_truncation_test
from quantcritic.audit.scorecard import compute_scorecard
from quantcritic.config import AuditConfig, BacktestConfig
from quantcritic.data.loader import MarketData, load_market_data
from quantcritic.data.synthetic import make_ohlcv_from_bars
from quantcritic.engine.backtest import run_backtest
from quantcritic.report.html import generate_audit_reports
from quantcritic.strategies.builder import (
    SPEC_PROMPT,
    SUPPORTED_INDICATORS,
    SUPPORTED_OPERATORS,
    BuilderStrategy,
    Condition,
    ExitSpec,
    IndicatorOperand,
    NumberOperand,
    StrategySpec,
    compute_indicator_series,
    parse_strategy_description,
    save_strategy_spec,
)
from quantcritic.strategies.registry import (
    get_strategy,
    list_strategies,
    reload_custom_strategies,
)

UTC0 = "2024-01-01 00:00:00+00:00"


# ----------------------------------------------------------------------------------------------
# 1. Spec JSON round-trip and validation errors (including anti-peeking checks)
# ----------------------------------------------------------------------------------------------
def test_spec_json_roundtrip_and_validation_errors() -> None:
    spec = StrategySpec(
        name="Trend_Alpha-1",
        description="Complex round-trip spec",
        long_rules=[
            Condition(
                left=IndicatorOperand(indicator="close"),
                operator="greater_than",
                right=IndicatorOperand(indicator="ema", period=50),
            ),
            Condition(
                left=IndicatorOperand(indicator="rsi", period=14),
                operator="crosses_above",
                right=NumberOperand(value=45.0),
            ),
        ],
        short_rules=[
            Condition(
                left=IndicatorOperand(indicator="macd_hist", fast_period=8, slow_period=21, signal_period=5),
                operator="crosses_below",
                right=NumberOperand(value=0.0),
            )
        ],
        mirror_long_for_short=False,
        combine="all",
        exit=ExitSpec(
            atr_period=14,
            atr_stop_mult=2.25,
            tp_r=3.0,
            max_bars=48,
            exit_rules=[
                Condition(
                    left=IndicatorOperand(indicator="bb_bw_pct", period=20, std_dev=2.0, lookback=60),
                    operator="greater_than",
                    right=NumberOperand(value=90.0),
                )
            ],
            exit_combine="any",
        ),
        notes="Test note",
        unsupported=["trailing stop"],
    )
    assert spec.name == "trend_alpha-1"
    raw_json = spec.model_dump_json()
    restored = StrategySpec.model_validate_json(raw_json)
    assert restored == spec

    # Invalid / peeking specs must raise ValidationError
    bad_payloads = [
        # Missing required atr_stop_mult
        {
            "name": "missing_stop",
            "long_rules": [{"left": {"indicator": "close"}, "operator": "greater_than", "right": {"kind": "number", "value": 10}}],
            "exit": {"atr_period": 14},
        },
        # atr_stop_mult <= 0
        {
            "name": "zero_stop",
            "long_rules": [{"left": {"indicator": "close"}, "operator": "greater_than", "right": {"kind": "number", "value": 10}}],
            "exit": {"atr_period": 14, "atr_stop_mult": 0.0},
        },
        # Empty rules
        {
            "name": "empty_rules",
            "long_rules": [],
            "short_rules": [],
            "exit": {"atr_period": 14, "atr_stop_mult": 2.0},
        },
        # mirror=True with no long rules
        {
            "name": "mirror_no_long",
            "long_rules": [],
            "mirror_long_for_short": True,
            "exit": {"atr_period": 14, "atr_stop_mult": 2.0},
        },
        # Deliberate peeking attempt: negative shift / center=True
        {
            "name": "peeking_shift",
            "long_rules": [
                {
                    "left": {"indicator": "close", "shift": -1},
                    "operator": "greater_than",
                    "right": {"kind": "number", "value": 100},
                }
            ],
            "exit": {"atr_period": 14, "atr_stop_mult": 2.0},
        },
        {
            "name": "peeking_center",
            "long_rules": [
                {
                    "left": {"indicator": "sma", "period": 5, "center": True},
                    "operator": "greater_than",
                    "right": {"kind": "number", "value": 100},
                }
            ],
            "exit": {"atr_period": 14, "atr_stop_mult": 2.0},
        },
        # Negative lookback / period
        {
            "name": "neg_period",
            "long_rules": [{"left": {"indicator": "ema", "period": -5}, "operator": "greater_than", "right": {"kind": "number", "value": 100}}],
            "exit": {"atr_period": 14, "atr_stop_mult": 2.0},
        },
        # MACD fast >= slow
        {
            "name": "bad_macd",
            "long_rules": [
                {
                    "left": {"indicator": "macd_line", "fast_period": 26, "slow_period": 12, "signal_period": 9},
                    "operator": "greater_than",
                    "right": {"kind": "number", "value": 0},
                }
            ],
            "exit": {"atr_period": 14, "atr_stop_mult": 2.0},
        },
        # Non-finite number operand
        {
            "name": "nan_val",
            "long_rules": [{"left": {"indicator": "close"}, "operator": "greater_than", "right": {"kind": "number", "value": float("nan")}}],
            "exit": {"atr_period": 14, "atr_stop_mult": 2.0},
        },
        # Path traversal / illegal name
        {
            "name": "../evil",
            "long_rules": [{"left": {"indicator": "close"}, "operator": "greater_than", "right": {"kind": "number", "value": 10}}],
            "exit": {"atr_period": 14, "atr_stop_mult": 2.0},
        },
    ]
    for bad in bad_payloads:
        with pytest.raises(ValidationError):
            StrategySpec.model_validate(bad)


# ----------------------------------------------------------------------------------------------
# 2. Hand-computed verification for EVERY indicator + exit_rules next-open execution
# ----------------------------------------------------------------------------------------------
def test_every_indicator_matches_hand_computation() -> None:
    # 7 carefully chosen bars
    bars = [
        (10.0, 12.0, 9.0, 10.0),   # b0
        (10.0, 13.0, 9.5, 12.0),   # b1
        (12.0, 15.0, 11.0, 14.0),  # b2
        (14.0, 15.5, 10.0, 11.0),  # b3
        (11.0, 16.0, 10.5, 15.0),  # b4
        (15.0, 18.0, 14.0, 17.0),  # b5
        (17.0, 19.0, 15.0, 16.0),  # b6
    ]
    df = make_ohlcv_from_bars(bars, start_time=UTC0, interval="4h")
    df["volume"] = [100.0, 200.0, 300.0, 200.0, 500.0, 100.0, 400.0]
    df["funding_rate"] = [0.0001, 0.0002, 0.0003, 0.0001, 0.0004, 0.0002, 0.0005]

    # OHLC
    assert compute_indicator_series(df, IndicatorOperand(indicator="open")).iloc[2] == pytest.approx(12.0)
    assert compute_indicator_series(df, IndicatorOperand(indicator="high")).iloc[2] == pytest.approx(15.0)
    assert compute_indicator_series(df, IndicatorOperand(indicator="low")).iloc[2] == pytest.approx(11.0)
    assert compute_indicator_series(df, IndicatorOperand(indicator="close")).iloc[2] == pytest.approx(14.0)

    # SMA(3) at b2 = (10+12+14)/3 = 12.0; at b3 = (12+14+11)/3 = 12.333333...
    sma3 = compute_indicator_series(df, IndicatorOperand(indicator="sma", period=3))
    assert np.isnan(sma3.iloc[1])
    assert sma3.iloc[2] == pytest.approx(12.0)
    assert sma3.iloc[3] == pytest.approx(37.0 / 3.0)

    # EMA(3): alpha = 2/(3+1) = 0.5, adjust=False, min_periods=3
    # e0=10, e1=0.5*12+0.5*10=11, e2=0.5*14+0.5*11=12.5, e3=0.5*11+0.5*12.5=11.75
    ema3 = compute_indicator_series(df, IndicatorOperand(indicator="ema", period=3))
    assert np.isnan(ema3.iloc[1])
    assert ema3.iloc[2] == pytest.approx(12.5)
    assert ema3.iloc[3] == pytest.approx(11.75)

    # RSI(2): alpha = 1/2 = 0.5, min_periods=2 on diff:
    # d = [NaN, +2, +2, -3, +4, ...]
    # gain = [NaN, 2, 2, 0, 4], loss = [NaN, 0, 0, 3, 0]
    # ag1=2, ag2=0.5*2+0.5*2=2, ag3=0.5*0+0.5*2=1 ; al1=0, al2=0, al3=0.5*3+0.5*0=1.5
    # b2: ag=2, al=0 -> RSI=100; b3: ag=1, al=1.5 -> RS=2/3 -> RSI = 100 - 100/(1+2/3) = 40.0
    rsi2 = compute_indicator_series(df, IndicatorOperand(indicator="rsi", period=2))
    assert rsi2.iloc[1] == pytest.approx(50.0)  # pre-warmup default in base.compute_rsi
    assert rsi2.iloc[2] == pytest.approx(100.0)
    assert rsi2.iloc[3] == pytest.approx(40.0)

    # ATR(2): TR = [3.0, 3.5, 4.0, 5.5, 5.5, ...] ; alpha=0.5, min_periods=2
    # a0=3.0, a1=0.5*3.5+0.5*3.0=3.25, a2=0.5*4.0+0.5*3.25=3.625
    atr2 = compute_indicator_series(df, IndicatorOperand(indicator="atr", period=2))
    assert np.isnan(atr2.iloc[0])
    assert atr2.iloc[1] == pytest.approx(3.25)
    assert atr2.iloc[2] == pytest.approx(3.625)

    # Donchian(3) shifted by 1: at b3, uses PREVIOUS 3 bars (b0, b1, b2):
    # highs=[12,13,15] -> upper=15.0; lows=[9,9.5,11] -> lower=9.0
    # At b4, uses (b1, b2, b3): highs=[13,15,15.5] -> upper=15.5; lows=[9.5,11,10] -> lower=9.5
    dc_u = compute_indicator_series(df, IndicatorOperand(indicator="donchian_upper", period=3))
    dc_l = compute_indicator_series(df, IndicatorOperand(indicator="donchian_lower", period=3))
    assert np.isnan(dc_u.iloc[2])
    assert dc_u.iloc[3] == pytest.approx(15.0)
    assert dc_l.iloc[3] == pytest.approx(9.0)
    assert dc_u.iloc[4] == pytest.approx(15.5)
    assert dc_l.iloc[4] == pytest.approx(9.5)

    # Bollinger(3, k=2.0) at b2: closes=[10,12,14], mid=12.0, pop std=sqrt((4+0+4)/3)=sqrt(8/3)
    sd_b2 = math.sqrt(8.0 / 3.0)
    assert compute_indicator_series(df, IndicatorOperand(indicator="bb_mid", period=3, std_dev=2.0)).iloc[2] == pytest.approx(12.0)
    assert compute_indicator_series(df, IndicatorOperand(indicator="bb_upper", period=3, std_dev=2.0)).iloc[2] == pytest.approx(12.0 + 2.0 * sd_b2)
    assert compute_indicator_series(df, IndicatorOperand(indicator="bb_lower", period=3, std_dev=2.0)).iloc[2] == pytest.approx(12.0 - 2.0 * sd_b2)

    # Bollinger bandwidth percentile(n=2, k=2.0, lookback=5):
    # bw = 2*k*sd / mid = 4*sd/mid for window 2 (valid from b1..b6 -> 6 values; percentile valid at b5 and b6)
    # b1: [10,12] mid=11, sd=1 -> bw=4/11=0.3636
    # b2: [12,14] mid=13, sd=1 -> bw=4/13=0.3077
    # b3: [14,11] mid=12.5, sd=1.5 -> bw=6/12.5=0.48
    # b4: [11,15] mid=13, sd=2 -> bw=8/13=0.6154
    # b5: [15,17] mid=16, sd=1 -> bw=4/16=0.25 -> smallest among [b1..b5] -> rank 1/5 = 20.0%
    bw_pct = compute_indicator_series(df, IndicatorOperand(indicator="bb_bw_pct", period=2, std_dev=2.0, lookback=5))
    assert np.isnan(bw_pct.iloc[4])
    assert bw_pct.iloc[5] == pytest.approx(20.0)

    # MACD(fast=2, slow=3, signal=2):
    # ema2: b0=10, b1=(2/3)*12+(1/3)*10=34/3, b2=(2/3)*14+(1/3)*(34/3)=118/9, b3=(2/3)*11+(1/3)*(118/9)=316/27
    # ema3: b2=12.5, b3=11.75
    # line at b2 = 118/9 - 12.5 = 118/9 - 112.5/9 = 5.5/9 = 11/18
    # line at b3 = 316/27 - 11.75 = 316/27 - 317.25/27 = -1.25/27 = -5/108
    # signal at b3 = (2/3)*line[b3] + (1/3)*line[b2]
    l2 = 118.0 / 9.0 - 12.5
    l3 = 316.0 / 27.0 - 11.75
    s3 = (2.0 / 3.0) * l3 + (1.0 / 3.0) * l2
    h3 = l3 - s3
    assert compute_indicator_series(df, IndicatorOperand(indicator="macd_line", fast_period=2, slow_period=3, signal_period=2)).iloc[3] == pytest.approx(l3)
    assert compute_indicator_series(df, IndicatorOperand(indicator="macd_signal", fast_period=2, slow_period=3, signal_period=2)).iloc[3] == pytest.approx(s3)
    assert compute_indicator_series(df, IndicatorOperand(indicator="macd_hist", fast_period=2, slow_period=3, signal_period=2)).iloc[3] == pytest.approx(h3)

    # volume_z(3) at b2: [100, 200, 300] -> mean=200, ddof=1 std=100 -> z=(300-200)/100 = 1.0
    vz = compute_indicator_series(df, IndicatorOperand(indicator="volume_z", period=3))
    assert np.isnan(vz.iloc[1])
    assert vz.iloc[2] == pytest.approx(1.0)

    # funding_z(3) at b2: [0.0001, 0.0002, 0.0003] -> mean=0.0002, std=0.0001 -> z=1.0
    fz = compute_indicator_series(df, IndicatorOperand(indicator="funding_z", period=3))
    assert np.isnan(fz.iloc[1])
    assert fz.iloc[2] == pytest.approx(1.0)

    # roc(2) at b2: (14/10 - 1)*100 = +40.0; at b3: (11/12 - 1)*100 = -8.333333...
    roc2 = compute_indicator_series(df, IndicatorOperand(indicator="roc", period=2))
    assert np.isnan(roc2.iloc[1])
    assert roc2.iloc[2] == pytest.approx(40.0)
    assert roc2.iloc[3] == pytest.approx((11.0 / 12.0 - 1.0) * 100.0)


def test_builder_exit_rules_close_position_at_next_open() -> None:
    bars = [
        (100.0, 101.0, 99.0, 100.0),   # b0 warmup
        (100.0, 101.0, 99.0, 100.0),   # b1 warmup
        (100.0, 101.0, 99.0, 100.0),   # b2 warmup
        (100.0, 101.0, 99.0, 100.0),   # b3 close=100 > 95? wait: cross above 102 -> b3=100, b4=105 triggers!
        (104.0, 106.0, 103.0, 105.0),  # b4 entry signal (+1)
        (106.0, 107.0, 104.0, 106.0),  # b5 enters at open=106; exit rule close<102 is False
        (106.0, 107.0, 100.5, 101.0),  # b6 close=101 < 102 -> exit_signal=True (ATR stop huge so not stopped)
        (102.0, 103.0, 101.0, 102.5),  # b7 exits at open=102!
    ]
    df = make_ohlcv_from_bars(bars, start_time=UTC0, interval="4h")
    spec = StrategySpec(
        name="exit_rule_test",
        long_rules=[
            Condition(
                left=IndicatorOperand(indicator="close"),
                operator="crosses_above",
                right=NumberOperand(value=102.0),
            )
        ],
        mirror_long_for_short=False,
        exit=ExitSpec(
            atr_period=1,
            atr_stop_mult=10.0,
            exit_rules=[
                Condition(
                    left=IndicatorOperand(indicator="close"),
                    operator="less_than",
                    right=NumberOperand(value=102.0),
                )
            ],
        ),
    )
    strat = BuilderStrategy(spec)
    sig_df = strat.generate(df)
    assert sig_df["signal"].iloc[4] == 1
    assert bool(sig_df["exit_signal"].iloc[6]) is True

    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0, funding_enabled=False, interval="4h")
    res = run_backtest(MarketData(ohlcv={"TESTUSDT": df}, funding={}, interval="4h"), strat, config=cfg)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["entry_time"] == df.index[5]
    assert tr["exit_time"] == df.index[7]
    assert tr["exit_reason"] == "exit_signal"
    assert tr["exit_price"] == pytest.approx(102.0)


# ----------------------------------------------------------------------------------------------
# 3. Causality: 20 seeded random valid specs pass truncation on real 4h sample data
# ----------------------------------------------------------------------------------------------
def _random_operand(rng: random.Random, ind: str | None = None) -> IndicatorOperand:
    chosen = ind or rng.choice(SUPPORTED_INDICATORS)
    if chosen in ("macd_line", "macd_signal", "macd_hist"):
        fast = rng.choice([5, 8, 12])
        slow = fast + rng.choice([8, 14, 20])
        sig = rng.choice([5, 9])
        return IndicatorOperand(indicator=chosen, fast_period=fast, slow_period=slow, signal_period=sig)  # type: ignore[arg-type]
    if chosen == "bb_bw_pct":
        return IndicatorOperand(
            indicator=chosen,
            period=rng.choice([14, 20, 30]),
            std_dev=rng.choice([1.5, 2.0, 2.5]),
            lookback=rng.choice([30, 60, 120]),
        )
    if chosen in ("bb_upper", "bb_lower", "bb_mid"):
        return IndicatorOperand(
            indicator=chosen,
            period=rng.choice([10, 20, 30]),
            std_dev=rng.choice([1.5, 2.0, 2.5]),
        )
    if chosen in ("rsi", "volume_z", "funding_z"):
        return IndicatorOperand(indicator=chosen, period=rng.choice([7, 14, 30, 60]))  # type: ignore[arg-type]
    return IndicatorOperand(indicator=chosen, period=rng.choice([5, 10, 20, 50]))  # type: ignore[arg-type]


def _random_condition(rng: random.Random, forced_ind: str | None = None) -> Condition:
    left = _random_operand(rng, ind=forced_ind)
    op = rng.choice(SUPPORTED_OPERATORS)
    if left.indicator in ("close", "open", "high", "low", "ema", "sma", "donchian_upper", "donchian_lower", "bb_upper", "bb_lower", "bb_mid"):
        right_ind = rng.choice(["ema", "sma", "bb_upper", "bb_lower", "bb_mid", "donchian_upper", "donchian_lower", "close"])
        right: IndicatorOperand | NumberOperand = _random_operand(rng, ind=right_ind)
    elif left.indicator == "rsi":
        right = NumberOperand(value=float(rng.choice([30, 40, 50, 60, 70])))
    elif left.indicator == "bb_bw_pct":
        right = NumberOperand(value=float(rng.choice([15, 25, 50, 80])))
    elif left.indicator in ("volume_z", "funding_z"):
        right = NumberOperand(value=float(rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0])))
    elif left.indicator == "roc":
        right = NumberOperand(value=float(rng.choice([-3.0, 0.0, 3.0])))
    elif left.indicator == "atr":
        right = _random_operand(rng, ind="atr")
    else:
        right = _random_operand(rng, ind="macd_signal") if left.indicator == "macd_line" else NumberOperand(value=0.0)
    return Condition(left=left, operator=op, right=right)  # type: ignore[arg-type]


def test_20_seeded_random_specs_pass_truncation_on_real_4h_data() -> None:
    md = load_market_data(["BTCUSDT"], interval="4h")
    rng = random.Random(20261125)

    # Guarantee every single supported indicator appears across the 20 seeded specs
    for i in range(20):
        forced = SUPPORTED_INDICATORS[i % len(SUPPORTED_INDICATORS)]
        n_long = rng.choice([1, 2])
        long_rules = [_random_condition(rng, forced_ind=forced)]
        if n_long == 2:
            long_rules.append(_random_condition(rng))
        mirror = rng.choice([True, False])
        short_rules = [] if mirror else [_random_condition(rng)]
        exit_rules = [_random_condition(rng)] if (i % 3 == 0) else []
        spec = StrategySpec(
            name=f"rand_spec_{i}",
            description=f"Seeded random causal spec {i}",
            long_rules=long_rules,
            short_rules=short_rules,
            mirror_long_for_short=mirror,
            combine=rng.choice(["all", "any"]),  # type: ignore[arg-type]
            exit=ExitSpec(
                atr_period=rng.choice([7, 14, 21]),
                atr_stop_mult=rng.choice([1.5, 2.0, 3.0]),
                tp_r=rng.choice([None, 2.0, 3.0]),
                max_bars=rng.choice([None, 12, 36]),
                exit_rules=exit_rules,
                exit_combine=rng.choice(["any", "all"]),  # type: ignore[arg-type]
            ),
        )
        strat = BuilderStrategy(spec)
        res = run_truncation_test(md, strat, n_cuts=8, window=100, seed=100 + i)
        assert res["passed"] is True, (i, forced, res["examples"][:3])
        assert res["total_mismatches"] == 0


# ----------------------------------------------------------------------------------------------
# 4. Builder registration + backtest / audit / CLI + 5. Python plugin discovery
# ----------------------------------------------------------------------------------------------
def test_builder_strategy_registered_and_runs_in_backtest_audit_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    user_dir = tmp_path / "user_strategies"
    examples_dir = user_dir / "examples"
    user_dir.mkdir(parents=True)
    examples_dir.mkdir(parents=True)
    monkeypatch.setattr(registry_mod, "USER_STRATEGIES_DIR", user_dir)
    monkeypatch.setattr(builder_mod, "USER_STRATEGIES_DIR", user_dir)

    spec = StrategySpec(
        name="cli_smoke_ema",
        description="CLI and audit smoke strategy",
        long_rules=[
            Condition(
                left=IndicatorOperand(indicator="ema", period=10),
                operator="crosses_above",
                right=IndicatorOperand(indicator="ema", period=30),
            )
        ],
        mirror_long_for_short=True,
        exit=ExitSpec(atr_period=14, atr_stop_mult=2.0, tp_r=2.0, max_bars=24),
    )
    save_strategy_spec(spec, directory=user_dir)
    reload_custom_strategies()
    try:
        assert "custom:cli_smoke_ema" in list_strategies()
        strat = get_strategy("custom:cli_smoke_ema")
        md = load_market_data(["BTCUSDT"], interval="4h")
        bt_res = run_backtest(md, strat, config=BacktestConfig(interval="4h"))
        assert bt_res.metrics["n_trades"] > 5

        session = AuditSession(
            strategy=strat,
            data=md,
            config=BacktestConfig(interval="4h"),
            audit_config=AuditConfig(lookahead_cuts=5, mc_simulations=50, random_baseline_runs=20, strictness="balanced"),
        )
        verdict = run_offline_critic(session)
        assert verdict.verdict in ("PROMISING", "NEEDS WORK", "REJECT")
        assert session.cached_results["lookahead"]["truncation"]["passed"] is True

        reports_dir = tmp_path / "reports"
        rc_bt = cli_main([
            "backtest", "--strategy", "custom:cli_smoke_ema",
            "--symbols", "BTCUSDT", "--interval", "4h", "--no-download",
        ])
        assert rc_bt == 0
        rc_au = cli_main([
            "audit", "--strategy", "custom:cli_smoke_ema",
            "--symbols", "BTCUSDT", "--interval", "4h",
            "--strictness", "relaxed", "--no-download", "--output-dir", str(reports_dir),
        ])
        assert rc_au == 0
        out = capsys.readouterr().out
        assert "Verdict mode : relaxed" in out
        assert "custom:cli_smoke_ema" in out
        html_files = list(reports_dir.glob("custom_cli_smoke_ema_*.html"))
        json_files = list(reports_dir.glob("custom_cli_smoke_ema_*.json"))
        assert len(html_files) == 1 and html_files[0].stat().st_size > 1000
        assert len(json_files) == 1 and json_files[0].stat().st_size > 500
    finally:
        monkeypatch.undo()
        reload_custom_strategies()


def test_python_plugin_discovery_registers_valid_and_skips_broken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    plugin_dir = tmp_path / "custom"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
    # template_strategy.py must be ignored unless renamed
    (plugin_dir / "template_strategy.py").write_text(
        "from quantcritic.strategies.base import Strategy, build_signal_frame, compute_atr\n"
        "class IgnoredTemplate(Strategy):\n"
        "    name = 'ignored_template_plugin'\n"
        "    def generate(self, df, params=None):\n"
        "        return build_signal_frame(df.index, [0]*len(df), compute_atr(df, 14))\n",
        encoding="utf-8",
    )
    # Valid causal plugin
    (plugin_dir / "good_plugin.py").write_text(
        "import numpy as np\n"
        "from quantcritic.strategies.base import Strategy, build_signal_frame, compute_atr, compute_ema\n"
        "class GoodPlugin(Strategy):\n"
        "    name = 'good_plugin_sma'\n"
        "    description = 'Good causal plugin'\n"
        "    default_params = {'period': 10, 'atr_period': 14, 'atr_stop_mult': 2.0}\n"
        "    def generate(self, df, params=None):\n"
        "        p = self.merge_params(params)\n"
        "        c = df['close'].astype(float)\n"
        "        ema = compute_ema(c, int(p['period']))\n"
        "        bull = (c > ema) & (c.shift(1) <= ema.shift(1))\n"
        "        sig = np.zeros(len(df), dtype=np.int8)\n"
        "        sig[bull.fillna(False).to_numpy()] = 1\n"
        "        sig[:min(20, len(sig))] = 0\n"
        "        return build_signal_frame(df.index, sig, compute_atr(df, int(p['atr_period'])) * float(p['atr_stop_mult']))\n",
        encoding="utf-8",
    )
    # Broken plugins: syntax error, runtime import error, duplicate built-in name
    (plugin_dir / "broken_syntax.py").write_text("def oops(:\n    pass\n", encoding="utf-8")
    (plugin_dir / "broken_runtime.py").write_text("raise RuntimeError('plugin import boom')\n", encoding="utf-8")
    (plugin_dir / "dup_builtin.py").write_text(
        "from quantcritic.strategies.base import Strategy, build_signal_frame, compute_atr\n"
        "class Dup(Strategy):\n"
        "    name = 'ema_trend'\n"
        "    def generate(self, df, params=None):\n"
        "        return build_signal_frame(df.index, [0]*len(df), compute_atr(df, 14))\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(registry_mod, "CUSTOM_STRATEGIES_DIR", plugin_dir)
    try:
        with caplog.at_level("WARNING"):
            reload_custom_strategies()
        names = list_strategies()
        assert "good_plugin_sma" in names
        assert "ignored_template_plugin" not in names
        assert "ema_trend" in names

        md = load_market_data(["BTCUSDT"], interval="1d")
        assert run_truncation_test(md, get_strategy("good_plugin_sma"), n_cuts=5, window=50)["passed"] is True
        assert any("broken_syntax" in rec.message or "broken_runtime" in rec.message for rec in caplog.records)
    finally:
        monkeypatch.undo()
        reload_custom_strategies()


# ----------------------------------------------------------------------------------------------
# 6. Mocked Claude beta.messages.parse for Plain-English -> StrategySpec
# ----------------------------------------------------------------------------------------------
def test_plain_english_parse_success_refusal_validation_and_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "closed-bar" in SPEC_PROMPT.lower()
    assert "atr" in SPEC_PROMPT.lower()
    assert "unsupported" in SPEC_PROMPT.lower()
    for ind in ("bb_bw_pct", "funding_z", "donchian_upper", "macd_line", "roc"):
        assert ind in SPEC_PROMPT

    # Missing key -> clear message
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    no_key = parse_strategy_description("Buy when RSI < 30")
    assert no_key.spec is None and no_key.error and "API key" in no_key.error

    sample_spec = StrategySpec(
        name="nl_pullback",
        description="Parsed from English",
        long_rules=[
            Condition(
                left=IndicatorOperand(indicator="close"),
                operator="greater_than",
                right=IndicatorOperand(indicator="ema", period=200),
            )
        ],
        mirror_long_for_short=True,
        exit=ExitSpec(atr_period=14, atr_stop_mult=2.0),
        notes="Defaulted ATR stop to 2.0x ATR(14).",
        unsupported=["trailing stop not supported directly"],
    )

    # 1) Success via real anthropic SDK + MockTransport
    captured_req: list[dict] = []

    def _ok_handler(request: httpx2.Request) -> httpx2.Response:
        captured_req.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": sample_spec.model_dump_json()}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )

    ok_client = anthropic.Anthropic(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(_ok_handler)),
    )
    ok_res = parse_strategy_description("Trend pullback with trailing stop", api_key="sk-test", client=ok_client)
    assert ok_res.spec == sample_spec
    assert ok_res.error is None and ok_res.refusal is False
    assert ok_res.spec.unsupported == ["trailing stop not supported directly"]
    assert captured_req[0]["body"]["model"] == "claude-opus-5-5"
    assert captured_req[0]["body"]["max_tokens"] == 16000
    assert captured_req[0]["body"]["output_config"]["effort"] == "high"
    assert captured_req[0]["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in captured_req[0]["headers"].get("anthropic-beta", "")

    # 2) Refusal
    class _RefusalBetaMessages:
        def parse(self, **kwargs):
            return SimpleNamespace(stop_reason="refusal", parsed_output=None)

    refusal_client = SimpleNamespace(beta=SimpleNamespace(messages=_RefusalBetaMessages()))
    ref_res = parse_strategy_description("anything", api_key="sk-test", client=refusal_client)
    assert ref_res.spec is None and ref_res.refusal is True and "refusal" in (ref_res.error or "").lower()

    # 3) pydantic.ValidationError (e.g. bad JSON from model via MockTransport)
    def _bad_json_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": '{"name": "broken", "long_rules": [], "exit": {"atr_stop_mult": -1}}'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )

    bad_client = anthropic.Anthropic(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(_bad_json_handler)),
    )
    val_res = parse_strategy_description("bad schema", api_key="sk-test", client=bad_client)
    assert val_res.spec is None and val_res.error and "validation" in val_res.error.lower()

    # 4) anthropic.APIError (e.g. HTTP 500)
    def _err_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "server down"}})

    err_client = anthropic.Anthropic(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(_err_handler)),
    )
    api_res = parse_strategy_description("server error", api_key="sk-test", client=err_client)
    assert api_res.spec is None and api_res.error and "api error" in api_res.error.lower()


# ----------------------------------------------------------------------------------------------
# 7. Strictness mode boundaries (strict / balanced / relaxed) across scorecard, AI gates, reports
# ----------------------------------------------------------------------------------------------
def _sc_args(
    *,
    trunc: bool = True,
    val_sh: float = 1.0,
    val_exp: float = 0.25,
    wf_sh: float = 1.0,
    test_sh: float = 1.0,
    test_ret: float = 0.20,
    test_exp: float = 0.25,
    test_trades: int = 25,
    p_val: float = 0.05,
    dsr: float = 0.70,
    base_exp: float = 0.20,
    stress_exp: float = 0.15,
) -> dict:
    return dict(
        lookahead_res={
            "status": "pass" if trunc else "fail",
            "truncation": {"passed": trunc},
            "delay": {"expectancy_collapse_pct": 5.0, "warn": False},
        },
        walkforward_res={
            "validation_metrics": {"sharpe": val_sh, "expectancy_r": val_exp},
            "rolling_oos_metrics": {"sharpe": wf_sh},
        },
        overfit_res={
            "dsr": {"dsr": dsr},
            "pbo": {"pbo": 0.20},
            "stability": {"stability_ratio": 0.85},
        },
        montecarlo_res={
            "random_baseline": {"p_value": p_val},
            "bootstrap": {"p95_max_drawdown": 0.12, "prob_loss": 0.05},
        },
        cost_res={
            "break_even_cost_bps": 35.0,
            "stress_2x_5bps_expectancy_r": stress_exp,
            "base_expectancy_r": base_exp,
        },
        regime_res={
            "yearly": [{"return": 0.12}] * 4,
            "regimes": {k: {"n_trades": 5, "net_pnl": 100.0} for k in ("bull", "bear", "chop")},
            "outlier_check": {"still_profitable": True},
        },
        full_metrics={"max_drawdown": 0.10},
        test_metrics={
            "sharpe": test_sh,
            "total_return": test_ret,
            "expectancy_r": test_exp,
            "n_trades": test_trades,
        },
        trades=pd.DataFrame({"R": [2.0, -1.0, 1.8, -1.0, 2.2, -1.0]}),
    )


def test_strictness_gate_boundaries_all_three_modes() -> None:
    # In ALL modes: lookahead fail -> REJECT (score <= 20); test lost -> NEVER PROMISING (<= 55)
    for mode in ("strict", "balanced", "relaxed"):
        cheat = compute_scorecard(**_sc_args(trunc=False), strictness=mode)
        assert cheat["verdict"] == "REJECT"
        assert cheat["overall_score"] <= 20.0
        assert cheat["strictness"] == mode

        lost_test = compute_scorecard(**_sc_args(test_sh=-0.3, test_ret=-0.05, test_exp=-0.1, test_trades=40), strictness=mode)
        assert lost_test["verdict"] != "PROMISING"
        assert lost_test["overall_score"] <= 55.0
        assert "test_fail" in [g["gate"] for g in lost_test["gates"]]

    # Strict: needs p <= 0.10, DSR >= 0.50, trades >= 10, pre-OOS positive, score >= 70
    assert compute_scorecard(**_sc_args(p_val=0.10, dsr=0.50, test_trades=10), strictness="strict")["verdict"] == "PROMISING"
    assert compute_scorecard(**_sc_args(p_val=0.11, dsr=0.50, test_trades=10), strictness="strict")["verdict"] == "NEEDS WORK"
    assert compute_scorecard(**_sc_args(p_val=0.10, dsr=0.49, test_trades=10), strictness="strict")["verdict"] == "NEEDS WORK"
    assert compute_scorecard(**_sc_args(val_sh=-0.1, val_exp=-0.02, test_trades=30), strictness="strict")["verdict"] == "NEEDS WORK"

    # Balanced (default): p <= 0.20, DSR >= 0.30, trades >= 10, PROMISING >= 65
    bal_border = compute_scorecard(**_sc_args(p_val=0.20, dsr=0.30, test_trades=10), strictness="balanced")
    assert bal_border["verdict"] == "PROMISING" and bal_border["overall_score"] >= 65.0
    assert compute_scorecard(**_sc_args(p_val=0.21, dsr=0.30, test_trades=10), strictness="balanced")["verdict"] == "NEEDS WORK"
    assert compute_scorecard(**_sc_args(p_val=0.20, dsr=0.29, test_trades=10), strictness="balanced")["verdict"] == "NEEDS WORK"
    assert compute_scorecard(**_sc_args(p_val=0.20, dsr=0.30, test_trades=9), strictness="balanced")["verdict"] == "NEEDS WORK"

    # Balanced pre-OOS loss + test profitable: < 20 test trades -> NEEDS WORK; >= 20 -> allowed PROMISING
    bal_few = compute_scorecard(**_sc_args(val_sh=-0.2, val_exp=-0.05, test_trades=19, p_val=0.15, dsr=0.40), strictness="balanced")
    assert bal_few["verdict"] == "NEEDS WORK" and bal_few["overall_score"] <= 60.0
    bal_many = compute_scorecard(**_sc_args(val_sh=-0.2, val_exp=-0.05, test_trades=20, p_val=0.15, dsr=0.40), strictness="balanced")
    assert bal_many["verdict"] == "PROMISING" and bal_many["overall_score"] >= 65.0

    # Balanced both test lost AND pre-OOS lost -> REJECT (<= 40)
    bal_both = compute_scorecard(**_sc_args(val_sh=-0.2, test_sh=-0.3, test_ret=-0.05, test_exp=-0.05), strictness="balanced")
    assert bal_both["verdict"] == "REJECT" and bal_both["overall_score"] <= 40.0

    # Relaxed: only gates = lookahead, test trades >= 5, base expectancy > 0, PROMISING >= 60
    rel_below_thr = compute_scorecard(
        **_sc_args(val_sh=-0.5, val_exp=-0.1, wf_sh=-0.4, test_trades=5, p_val=0.90, dsr=0.05, base_exp=0.05, test_sh=2.0, test_exp=0.4),
        strictness="relaxed",
    )
    assert rel_below_thr["overall_score"] == pytest.approx(59.9) and rel_below_thr["verdict"] == "NEEDS WORK"
    rel_pass = compute_scorecard(
        **_sc_args(val_sh=-0.5, val_exp=-0.1, wf_sh=-0.4, test_trades=5, p_val=0.90, dsr=0.05, base_exp=0.05, test_sh=2.0, test_exp=0.4, stress_exp=0.20),
        strictness="relaxed",
    )
    assert rel_pass["verdict"] == "PROMISING" and rel_pass["overall_score"] >= 60.0
    rel_few = compute_scorecard(**_sc_args(test_trades=4, base_exp=0.2), strictness="relaxed")
    assert rel_few["verdict"] == "NEEDS WORK" and rel_few["overall_score"] <= 59.9
    rel_neg_exp = compute_scorecard(**_sc_args(test_trades=20, base_exp=0.0), strictness="relaxed")
    assert rel_neg_exp["verdict"] == "NEEDS WORK" and rel_neg_exp["overall_score"] <= 59.9


def test_strictness_enforced_for_ai_verdict_offline_and_reports(tmp_path: Path) -> None:
    md = load_market_data(["BTCUSDT"], interval="1d")
    session = AuditSession(
        strategy=get_strategy("ema_trend"),
        data=md,
        config=BacktestConfig(interval="1d"),
        audit_config=AuditConfig(lookahead_cuts=5, mc_simulations=50, random_baseline_runs=20, strictness="balanced"),
    )
    assert session.strictness == "balanced"
    session.ensure_complete_audit()
    # Pretend AI submitted PROMISING 90 despite losing test
    session.record_verdict(
        verdict="PROMISING",
        score=90.0,
        summary="Test",
        strengths=["s"],
        weaknesses=[],
        red_flags=[],
        improvements=[],
        recommended_risk_per_trade=0.005,
        source="ai",
    )
    session.enforce_hard_gates()
    assert session.verdict is not None
    assert session.verdict.verdict != "PROMISING"
    assert session.verdict.score <= 55.0

    # Offline synthesis & HTML/JSON report include Verdict mode
    session.verdict = synthesize_verdict_from_session(session)
    assert "Verdict mode: balanced" in session.verdict.summary
    html_p, json_p = generate_audit_reports(session, output_dir=tmp_path, basename="strictness_check")
    html_text = html_p.read_text(encoding="utf-8")
    payload = json.loads(json_p.read_text(encoding="utf-8"))
    assert "Verdict mode: balanced" in html_text
    assert payload["meta"]["strictness"] == "balanced"
    assert payload["scorecard"]["strictness"] == "balanced"
