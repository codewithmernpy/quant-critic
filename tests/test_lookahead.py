"""Tests verifying lookahead truncation catches leaky_demo and clears honest strategies."""

from __future__ import annotations

import pytest

from quantcritic.audit.lookahead import run_truncation_test
from quantcritic.data.loader import MarketData
from quantcritic.data.synthetic import generate_funding_series, generate_random_walk_ohlcv
from quantcritic.strategies.registry import get_strategy, list_strategies


@pytest.fixture(scope="module")
def synthetic_market() -> MarketData:
    df = generate_random_walk_ohlcv(n_bars=500, interval="4h", seed=123)
    f = generate_funding_series(df.index[0], df.index[-1], seed=123)
    df["funding_rate"] = f.reindex(df.index, method="ffill").fillna(0.0)
    return MarketData(ohlcv={"SYNTHUSDT": df}, funding={"SYNTHUSDT": f}, interval="4h")


def test_leaky_demo_fails_truncation_test(synthetic_market: MarketData) -> None:
    strat = get_strategy("leaky_demo")
    res = run_truncation_test(synthetic_market, strat, n_cuts=15, window=80, seed=42)
    assert res["passed"] is False
    assert res["total_mismatches"] > 0
    assert len(res["examples"]) > 0


@pytest.mark.parametrize("strategy_name", list_strategies(include_leaky=False))
def test_honest_strategies_pass_truncation_test(
    synthetic_market: MarketData, strategy_name: str
) -> None:
    strat = get_strategy(strategy_name)
    res = run_truncation_test(synthetic_market, strat, n_cuts=15, window=80, seed=42)
    assert res["passed"] is True, f"{strategy_name} failed truncation: {res['examples'][:3]}"
    assert res["total_mismatches"] == 0
