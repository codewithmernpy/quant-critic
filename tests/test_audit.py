"""Tests for walk-forward test-set lock, overfitting PBO/DSR, sizing, and offline audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantcritic.agent.offline import run_offline_critic
from quantcritic.agent.tools import AuditSession
from quantcritic.audit.overfit import compute_cscv_pbo, compute_deflated_sharpe_ratio
from quantcritic.config import AuditConfig, BacktestConfig
from quantcritic.data.loader import load_market_data
from quantcritic.report.html import generate_audit_reports
from quantcritic.risk.sizing import (
    capped_fractional_kelly,
    estimate_risk_of_ruin,
    fixed_fractional_size,
    volatility_target_size,
)
from quantcritic.strategies.registry import get_strategy


def test_test_set_lock_and_final_test_once_only() -> None:
    md = load_market_data(["BTCUSDT"], interval="1d")
    cfg = BacktestConfig(interval="1d")
    audit_cfg = AuditConfig(lookahead_cuts=10, mc_simulations=100, random_baseline_runs=30)
    session = AuditSession(
        strategy=get_strategy("ema_trend"),
        data=md,
        config=cfg,
        audit_config=audit_cfg,
    )

    session.get_overview()
    session.run_lookahead()
    session.run_walkforward()
    session.run_overfit()
    session.run_monte_carlo()
    session.run_random_baseline()
    session.run_cost_stress()
    session.run_regime_breakdown()
    session.run_prop_firm()

    # Locked test slice MUST NOT have been accessed yet
    assert session.test_access_count == 0
    assert session.final_test_called is False

    # Invalid period to run_backtest is blocked
    bad_period = session.run_backtest_period("{}", period="test")
    assert "error" in bad_period
    assert session.test_access_count == 0

    # First final_test call succeeds and locks decisions
    ft1 = session.run_final_test("{}")
    assert ft1["status"] == "evaluated_once"
    assert session.test_access_count == 1
    assert session.final_test_called is True
    assert session.decisions_frozen is True

    # Second final_test call returns an error
    ft2 = session.run_final_test("{}")
    assert "error" in ft2
    assert session.test_access_count == 1

    # run_backtest after final_test is refused
    frozen_res = session.run_backtest_period("{}", period="train")
    assert "error" in frozen_res
    assert "frozen" in frozen_res["error"].lower()


def test_overfit_pbo_on_pure_noise_and_dsr_sanity() -> None:
    pbos: list[float] = []
    for seed in (10, 20, 30, 40, 50):
        rng = np.random.default_rng(seed)
        noise_mat = rng.normal(loc=0.0, scale=0.01, size=(600, 12))
        pbo_res = compute_cscv_pbo(noise_mat, n_blocks=10)
        assert pbo_res["n_splits"] == 252
        pbos.append(float(pbo_res["pbo"]))
    mean_pbo = float(np.mean(pbos))
    assert 0.30 <= mean_pbo <= 0.70

    # DSR penalizes larger number of trials
    rng2 = np.random.default_rng(42)
    sample_rets = rng2.normal(loc=0.0005, scale=0.01, size=500)
    dsr_1 = compute_deflated_sharpe_ratio(sample_rets, n_trials=1)
    dsr_24 = compute_deflated_sharpe_ratio(sample_rets, n_trials=24)
    assert 0.0 <= dsr_24["dsr"] <= 1.0
    assert dsr_24["dsr"] <= dsr_1["dsr"]


def test_sizing_and_risk_of_ruin_math() -> None:
    # Uncapped fixed-fractional sizing
    qty, risk_amt = fixed_fractional_size(
        equity=10_000.0,
        entry_price=50_000.0,
        stop_dist=2_500.0,
        risk_per_trade=0.005,
        max_leverage=3.0,
    )
    assert risk_amt == pytest.approx(50.0)
    assert qty == pytest.approx(50.0 / 2_500.0)

    # Leverage-capped sizing
    qty_cap, risk_cap = fixed_fractional_size(
        equity=10_000.0,
        entry_price=100.0,
        stop_dist=0.10,
        risk_per_trade=0.01,
        max_leverage=3.0,
    )
    assert qty_cap * 100.0 == pytest.approx(30_000.0)
    assert risk_cap == pytest.approx(30.0)

    # Volatility targeting
    v_qty = volatility_target_size(10_000.0, price=100.0, realized_vol_ann=0.30, target_vol_ann=0.15)
    assert v_qty == pytest.approx(50.0)

    # Capped Kelly: p=0.5, win=2R, loss=1R -> full Kelly 0.25, quarter Kelly 0.0625 -> capped at 0.02
    assert capped_fractional_kelly(0.5, 2.0, 1.0, fraction=0.25, max_risk=0.02) == pytest.approx(0.02)
    assert capped_fractional_kelly(0.3, 1.0, 1.0) == 0.0

    # Risk of ruin bounds & monotonicity
    assert estimate_risk_of_ruin(0.4, 1.0, 1.0) == 1.0
    ror_low = estimate_risk_of_ruin(0.55, 1.5, 1.0, risk_per_trade=0.005)
    ror_high = estimate_risk_of_ruin(0.55, 1.5, 1.0, risk_per_trade=0.02)
    assert 0.0 <= ror_low < ror_high <= 1.0


def test_offline_audit_end_to_end_one_symbol_writes_html_and_json(tmp_path: Path) -> None:
    md = load_market_data(["BTCUSDT"], interval="4h")
    cfg = BacktestConfig(interval="4h")
    audit_cfg = AuditConfig(lookahead_cuts=10, mc_simulations=100, random_baseline_runs=40)
    session = AuditSession(
        strategy=get_strategy("donchian_breakout"),
        data=md,
        config=cfg,
        audit_config=audit_cfg,
    )
    verdict = run_offline_critic(session)
    assert verdict.verdict in ("PROMISING", "NEEDS WORK", "REJECT")
    assert session.test_access_count == 1

    html_p, json_p = generate_audit_reports(session, output_dir=tmp_path, basename="btc_4h_test")
    assert html_p.is_file() and html_p.stat().st_size > 10_000
    assert json_p.is_file() and json_p.stat().st_size > 2_000
    html_text = html_p.read_text(encoding="utf-8")
    assert "QuantCritic" in html_text
    assert "donchian_breakout" in html_text
