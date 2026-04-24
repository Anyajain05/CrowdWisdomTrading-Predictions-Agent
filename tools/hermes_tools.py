from __future__ import annotations

from typing import Optional

from utils.config import (
    HERMES_ENABLED,
    HERMES_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from utils.logger import get_logger

logger = get_logger("hermes")


def hermes_available() -> bool:
    if not HERMES_ENABLED:
        return False
    try:
        from run_agent import AIAgent  # noqa: F401
        return True
    except Exception:
        return False


def get_hermes_agent(system_prompt: Optional[str] = None):
    if not HERMES_ENABLED:
        raise RuntimeError("Hermes integration is disabled by config")

    from run_agent import AIAgent

    return AIAgent(
        model=HERMES_MODEL,
        api_key=OPENROUTER_API_KEY or None,
        base_url=OPENROUTER_BASE_URL,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        max_iterations=8,
        ephemeral_system_prompt=system_prompt,
    )


def synthesize_with_hermes(prompt: str, system_prompt: Optional[str] = None) -> str:
    agent = get_hermes_agent(system_prompt=system_prompt)
    response = agent.chat(prompt)
    if not isinstance(response, str):
        raise RuntimeError("Hermes returned a non-string response")
    return response.strip()
