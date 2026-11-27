"""Claude AI Strategy Critic runner using Anthropic's official beta tool_runner."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import anthropic

from quantcritic.agent.offline import run_offline_critic, synthesize_verdict_from_session
from quantcritic.agent.prompts import SYSTEM_PROMPT, build_kickoff_prompt
from quantcritic.agent.tools import AuditSession, CriticVerdict, build_beta_tools
from quantcritic.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


def run_ai_critic(
    session: AuditSession,
    use_ai: bool = True,
    api_key: str | None = None,
    model: str | None = None,
    on_event: Callable[[str, str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> CriticVerdict:
    """Run the Claude AI critic agent or fall back cleanly to the deterministic offline critic."""
    offline_env = os.getenv("QC_OFFLINE", "0").strip() == "1"
    effective_key = api_key or os.getenv("ANTHROPIC_API_KEY", "").strip()

    if not use_ai or offline_env or not effective_key:
        reason = "offline mode requested" if (not use_ai or offline_env) else "ANTHROPIC_API_KEY not configured"
        logger.info("Running deterministic offline critic (%s).", reason)
        if on_event is not None:
            on_event("system", f"Using deterministic offline critic ({reason}).")
        return run_offline_critic(session, progress_callback=progress_callback)

    chosen_model = model or os.getenv("QC_MODEL", DEFAULT_MODEL)
    kickoff = build_kickoff_prompt(
        strategy_name=session.strategy.name,
        symbols=session.data.symbols,
        interval=session.data.interval,
        strictness=session.strictness or "balanced",
    )

    def _on_tool(tool_name: str, args: dict[str, Any], out_json: str) -> None:
        if on_event is not None:
            on_event("tool_result", f"[{tool_name}] {out_json[:300]}")

    tools = build_beta_tools(session, on_tool_event=_on_tool)

    try:
        client = anthropic.Anthropic(api_key=effective_key)
        runner = client.beta.messages.tool_runner(
            model=chosen_model,
            max_tokens=16000,
            max_iterations=24,
            system=SYSTEM_PROMPT,
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            tools=tools,
            messages=[{"role": "user", "content": kickoff}],
        )
        for idx, message in enumerate(runner, start=1):
            if progress_callback is not None:
                progress_callback(min(idx, 11), 12, f"AI Critic turn {idx}")
            for block in getattr(message, "content", []):
                b_type = getattr(block, "type", "")
                if b_type == "text":
                    text = getattr(block, "text", "").strip()
                    if text:
                        session.agent_thoughts.append(text)
                        if on_event is not None:
                            on_event("assistant", text)
                elif b_type == "tool_use":
                    t_name = getattr(block, "name", "")
                    t_in = getattr(block, "input", {})
                    if on_event is not None:
                        on_event("tool_use", f"Calling {t_name}({t_in})")

            if getattr(message, "stop_reason", None) == "refusal":
                logger.warning("Claude returned stop_reason='refusal'; falling back to offline critic.")
                if on_event is not None:
                    on_event("system", "Model returned stop_reason='refusal'; completing via offline critic.")
                break

    except anthropic.APIError as exc:
        # APIConnectionError / APITimeoutError, RateLimitError and every other APIStatusError
        # (auth, 4xx, 5xx) plus response-validation errors all derive from APIError.
        logger.warning("Anthropic API error (%s); falling back to offline critic.", exc)
        if on_event is not None:
            on_event("system", f"Anthropic API fallback triggered: {type(exc).__name__}: {exc}")
        if session.verdict is None:
            return run_offline_critic(session, progress_callback=progress_callback)

    # Ensure every audit section is populated (for the final params) and a verdict exists
    session.ensure_complete_audit()
    if session.verdict is not None:
        # The AI may have submitted before the locked test ran: re-apply the hard gates now
        session.enforce_hard_gates()
    else:
        cv = synthesize_verdict_from_session(session)
        session.record_verdict(
            verdict=cv.verdict,
            score=cv.score,
            summary=cv.summary,
            strengths=cv.strengths,
            weaknesses=cv.weaknesses,
            red_flags=cv.red_flags,
            improvements=cv.improvements,
            recommended_risk_per_trade=cv.recommended_risk_per_trade,
            source="offline",
        )
    return session.verdict  # type: ignore[return-value]
