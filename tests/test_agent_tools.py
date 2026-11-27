"""Tests for Anthropic @beta_tool schema generation and API error fallback handling."""

from __future__ import annotations

import httpx2 as httpx
import anthropic

from quantcritic.agent.critic import run_ai_critic
from quantcritic.agent.offline import run_offline_critic
from quantcritic.agent.tools import AuditSession, build_beta_tools
from quantcritic.config import AuditConfig, BacktestConfig
from quantcritic.data.loader import load_market_data
from quantcritic.strategies.registry import get_strategy


def test_beta_tools_build_and_valid_json_schemas() -> None:
    md = load_market_data(["BTCUSDT"], interval="1d")
    session = AuditSession(
        strategy=get_strategy("donchian_breakout"),
        data=md,
        config=BacktestConfig(interval="1d"),
        audit_config=AuditConfig(lookahead_cuts=5, mc_simulations=50, random_baseline_runs=20),
    )
    tools = build_beta_tools(session)
    expected_names = {
        "get_overview",
        "run_backtest",
        "lookahead_test",
        "walk_forward",
        "overfit_tests",
        "monte_carlo",
        "random_baseline",
        "cost_stress",
        "regime_breakdown",
        "prop_firm_check",
        "final_test",
        "submit_verdict",
    }
    tool_dicts = [t.to_dict() for t in tools]
    assert {td["name"] for td in tool_dicts} == expected_names

    for td in tool_dicts:
        assert isinstance(td["description"], str) and len(td["description"]) > 10
        schema = td["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema

    sv_schema = next(td["input_schema"] for td in tool_dicts if td["name"] == "submit_verdict")
    for list_field in ("strengths", "weaknesses", "red_flags", "improvements"):
        assert sv_schema["properties"][list_field]["type"] == "array"
    assert sv_schema["properties"]["recommended_risk_per_trade"]["type"] == "number"


def test_ai_critic_falls_back_cleanly_on_api_connection_error(monkeypatch) -> None:
    md = load_market_data(["BTCUSDT"], interval="1d")
    session = AuditSession(
        strategy=get_strategy("ema_trend"),
        data=md,
        config=BacktestConfig(interval="1d"),
        audit_config=AuditConfig(lookahead_cuts=5, mc_simulations=50, random_baseline_runs=20),
    )

    class _BrokenMessages:
        def tool_runner(self, **kwargs):
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=req)

    class _BrokenBeta:
        messages = _BrokenMessages()

    class _FakeAnthropic:
        def __init__(self, **kwargs) -> None:
            self.beta = _BrokenBeta()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    monkeypatch.delenv("QC_OFFLINE", raising=False)

    events: list[tuple[str, str]] = []
    verdict = run_ai_critic(
        session=session,
        use_ai=True,
        api_key="sk-ant-fake-test-key",
        on_event=lambda k, m: events.append((k, m)),
    )
    assert verdict.source == "offline"
    assert verdict.verdict in ("PROMISING", "NEEDS WORK", "REJECT")
    assert any("fallback" in msg.lower() for _, msg in events)


def test_offline_critic_no_duplicate_tool_steps_after_precheck() -> None:
    md = load_market_data(["BTCUSDT"], interval="1d")
    cfg = BacktestConfig(interval="1d")
    audit_cfg = AuditConfig(lookahead_cuts=5, mc_simulations=50, random_baseline_runs=20)

    # Simulate page/CLI flow: check_min_train_val_trades runs before run_offline_critic
    prechecked = AuditSession(
        strategy=get_strategy("donchian_breakout"),
        data=md,
        config=cfg,
        audit_config=audit_cfg,
    )
    prechecked.check_min_train_val_trades(min_trades=5)
    v_pre = run_offline_critic(prechecked)

    # Compare against a fresh offline run without precheck
    fresh = AuditSession(
        strategy=get_strategy("donchian_breakout"),
        data=md,
        config=cfg,
        audit_config=audit_cfg,
    )
    v_fresh = run_offline_critic(fresh)

    tools_pre = [entry["tool"] for entry in prechecked.tool_log]
    tools_fresh = [entry["tool"] for entry in fresh.tool_log]
    assert tools_pre.count("walk_forward") == 1
    assert len(tools_pre) == len(set(tools_pre))
    assert all(a != b for a, b in zip(tools_pre, tools_pre[1:]))
    assert [entry["step"] for entry in prechecked.tool_log] == list(range(1, len(prechecked.tool_log) + 1))
    assert tools_pre == tools_fresh
    assert v_pre.verdict == v_fresh.verdict
    assert v_pre.score == v_fresh.score

    # Also verify run_ai_critic(..., use_ai=False) path after precheck
    via_critic = AuditSession(
        strategy=get_strategy("donchian_breakout"),
        data=md,
        config=cfg,
        audit_config=audit_cfg,
    )
    via_critic.check_min_train_val_trades(min_trades=5)
    v_ai_off = run_ai_critic(via_critic, use_ai=False)
    tools_ai_off = [entry["tool"] for entry in via_critic.tool_log]
    assert tools_ai_off == tools_fresh
    assert v_ai_off.score == v_fresh.score
