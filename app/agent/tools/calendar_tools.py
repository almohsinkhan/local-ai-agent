from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.agent.state import AgentState
from app.agent.tools.common import as_positive_int, require_human_approval, to_tool_content
from app.tools.calendar import add_event, effective_timezone_name, list_events


def _normalize_add_event_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Normalize UTC-marked local wall times into calendar-local times."""
    args = dict(raw_args or {})
    tz_name = effective_timezone_name()
    if tz_name.upper() == "UTC":
        return args

    for key in ("start_iso", "end_iso"):
        value = args.get(key)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue

        if parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0):
            args[key] = parsed.replace(tzinfo=None).isoformat(timespec="seconds")
    return args


def _default_calendar_range(days_ahead: int = 7) -> tuple[str, str]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    end = now + timedelta(days=days_ahead)
    return now.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


@tool
def list_events_tool(time_min: str = "", time_max: str = "", max_results: int = 10) -> str:
    """List calendar events inside a date range.

    Use when:
    - The user asks to show or check calendar events.

    What arguments it expects:
    - time_min: ISO start (optional).
    - time_max: ISO end (optional).
    - max_results: number of events (1..25).

    What not to do:
    - Do not create/update events with this tool.

    Special behavior:
    - Defaults to now through next 7 days when range is missing.
    """
    default_min, default_max = _default_calendar_range()
    result = list_events(
        time_min=str(time_min or "").strip() or default_min,
        time_max=str(time_max or "").strip() or default_max,
        max_results=as_positive_int(max_results, 10, max_value=25),
    )
    return to_tool_content(result)


@tool
def add_event_tool(
    summary: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    location: str = "",
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Create a calendar event only after approval.

    Use when:
    - The user asks to schedule/create an event.

    What arguments it expects:
    - summary, start_iso, end_iso (required).
    - description, location (optional).

    What not to do:
    - Do not execute without approval.

    Special behavior:
    - Normalizes local-time intents deterministically before create.
    """
    require_human_approval(state)
    args = _normalize_add_event_args(
        {
            "summary": summary,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "description": description,
            "location": location,
        }
    )
    result = add_event(**args)
    return to_tool_content(result)
