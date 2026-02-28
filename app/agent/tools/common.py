from __future__ import annotations

import json
from typing import Any

from app.agent.state import AgentState


def as_positive_int(value: Any, default: int, *, min_value: int = 1, max_value: int = 50) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(parsed, max_value))


def require_human_approval(state: AgentState | None) -> None:
    if not state or state.get("human_approved") is not True:
        raise ValueError("This action requires explicit human approval before execution.")


def to_tool_content(value: Any) -> str:
    """Convert tool output into LLM-safe content."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
