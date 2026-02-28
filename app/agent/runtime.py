from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from app.agent.state import AgentState
from app.agent.tooling import GUARDED_ACTIONS, TOOLS
from app.observability import langsmith_traceable, timed
from app.tools.calendar import effective_timezone_name

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "moonshotai/kimi-k2-instruct-0905")
GROQ_API_KEY = (os.getenv("GROQ_API_KEY", "") or "").strip().strip('"').strip("'")

LLM = ChatGroq(model=MODEL_NAME, temperature=0, api_key=GROQ_API_KEY)
LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)


def _current_datetime_context() -> str:
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    tz_name = effective_timezone_name()
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        tz_name = "UTC"
        local_tz = timezone.utc
    now_local = now_utc.astimezone(local_tz)
    return (
        "Current date/time context for planning dates and times:\n"
        f"- UTC now: {now_utc.isoformat().replace('+00:00', 'Z')}\n"
        f"- Local now ({tz_name}): {now_local.isoformat()}\n"
        "Resolve relative dates like today/tomorrow/next week using this context. "
        "Use local timezone for scheduling unless user explicitly asks for another timezone."
    )


def _assistant_system_prompt() -> str:
    return (
        "You are a personal desktop assistant. "
        "Speak clearly, simply, and concisely. "
        "Keep answers short unless the user asks for details. "
        "Never mention internal state, tools, or APIs in your final user-facing answer.\n\n"
        f"{_current_datetime_context()}"
    )


def _safe_json_load(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _latest_tool_outputs(messages: list) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            outputs.append(
                {
                    "name": getattr(msg, "name", ""),
                    "tool_call_id": getattr(msg, "tool_call_id", ""),
                    "content": _safe_json_load(msg.content),
                }
            )
            continue
        if outputs:
            break
    outputs.reverse()
    return outputs


def _latest_ai_tool_calls(messages: list) -> list[dict[str, Any]]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return [dict(call) for call in msg.tool_calls]
    return []


@langsmith_traceable(name="assistant", run_type="chain")
@timed("assistant")
def assistant(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    response = LLM_WITH_TOOLS.invoke([SystemMessage(content=_assistant_system_prompt()), *messages])

    updates: AgentState = {
        "messages": [response],
        "approval_rejected": False,
    }

    first_call = response.tool_calls[0] if response.tool_calls else None
    if first_call:
        updates["planned_action"] = {
            "name": str(first_call.get("name", "")),
            "args": dict(first_call.get("args", {}) or {}),
        }
    else:
        updates["planned_action"] = {"name": "respond", "args": {}}
        updates["human_approved"] = None

    latest_outputs = _latest_tool_outputs(messages)
    if latest_outputs:
        updates["last_tool_result"] = latest_outputs
        for output in latest_outputs:
            if output.get("name") == "get_emails_tool" and isinstance(output.get("content"), list):
                updates["last_email_results"] = output["content"]

    return updates


@langsmith_traceable(name="execute_action", run_type="chain")
@timed("execute_action")
def execute_action(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    tool_calls = _latest_ai_tool_calls(messages)
    if not tool_calls:
        return {"approval_rejected": False}

    guarded_present = any(str(call.get("name", "")) in GUARDED_ACTIONS for call in tool_calls)
    if not guarded_present:
        return {"approval_rejected": False}

    if state.get("human_approved") is True:
        return {"approval_rejected": False}

    approval_error = "Action not approved." if state.get("human_approved") is False else "Action requires approval."
    blocked_messages: list[ToolMessage] = []
    for call in tool_calls:
        blocked_messages.append(
            ToolMessage(
                content=approval_error,
                tool_call_id=str(call.get("id", "")),
                name=str(call.get("name", "")),
            )
        )

    return {
        "messages": blocked_messages,
        "approval_rejected": True,
        "human_approved": None,
    }


def route_after_assistant(state: AgentState) -> Literal["execute_action", "end"]:
    messages = state.get("messages", [])
    if not messages:
        return "end"
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execute_action"
    return "end"


def route_after_execute(state: AgentState) -> Literal["tools", "assistant"]:
    if state.get("approval_rejected"):
        return "assistant"
    return "tools"
