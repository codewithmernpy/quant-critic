"""AI and offline strategy critic agents."""

from quantcritic.agent.critic import run_ai_critic
from quantcritic.agent.offline import run_offline_critic, synthesize_verdict_from_session
from quantcritic.agent.prompts import SYSTEM_PROMPT, build_kickoff_prompt
from quantcritic.agent.tools import AuditSession, CriticVerdict, build_beta_tools

__all__ = [
    "AuditSession",
    "CriticVerdict",
    "SYSTEM_PROMPT",
    "build_beta_tools",
    "build_kickoff_prompt",
    "run_ai_critic",
    "run_offline_critic",
    "synthesize_verdict_from_session",
]
