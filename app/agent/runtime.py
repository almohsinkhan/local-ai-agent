from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - optional dependency
    ChatOllama = None

from app.agent.state import AgentState
from app.agent.tooling import GUARDED_ACTIONS, TOOLS
from app.observability import langsmith_traceable, timed
from app.tools.calendar import effective_timezone_name

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "moonshotai/kimi-k2-instruct-0905")
GROQ_API_KEY = (os.getenv("GROQ_API_KEY", "") or "").strip().strip('"').strip("'")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LLM = ChatGroq(model=MODEL_NAME, temperature=0, api_key=GROQ_API_KEY)
LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)
OLLAMA_LLM = (
    ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    if ChatOllama is not None
    else None
)
OLLAMA_WITH_TOOLS = OLLAMA_LLM.bind_tools(TOOLS) if OLLAMA_LLM is not None else None


def _is_capacity_or_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "503" in text
        or "429" in text
        or "over capacity" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    )


def invoke_with_resilience(messages: list, retries: int = 2):
    attempts = max(1, retries + 1)
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            return LLM_WITH_TOOLS.invoke(messages)
        except Exception as exc:
            if not _is_capacity_or_rate_limit_error(exc):
                raise
            last_error = exc
            if attempt < attempts - 1:
                wait = (2**attempt) + random.random()
                time.sleep(wait)

    if OLLAMA_WITH_TOOLS is not None:
        return OLLAMA_WITH_TOOLS.invoke(messages)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Primary model failed and fallback model is unavailable.")

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


def _compact_runtime_messages(messages: list) -> list:
    latest_human_idx: int | None = None
    latest_ai_idx: int | None = None

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if latest_human_idx is None and isinstance(msg, HumanMessage):
            latest_human_idx = idx
        if latest_ai_idx is None and isinstance(msg, AIMessage):
            latest_ai_idx = idx
        if latest_human_idx is not None and latest_ai_idx is not None:
            break

    selected_indices: list[int] = []

    if latest_ai_idx is not None:
        selected_indices.append(latest_ai_idx)
        for idx in range(latest_ai_idx + 1, len(messages)):
            if isinstance(messages[idx], ToolMessage):
                selected_indices.append(idx)
                continue
            break

    if latest_human_idx is not None:
        selected_indices.append(latest_human_idx)

    selected_indices = sorted(set(selected_indices))
    return [messages[idx] for idx in selected_indices]


def _extract_intent_with_ollama(runtime_messages: list) -> dict[str, Any] | None:
    if OLLAMA_LLM is None or not runtime_messages:
        return None

    latest_human = next((msg for msg in reversed(runtime_messages) if isinstance(msg, HumanMessage)), None)
    if latest_human is None:
        return None

    try:
        intent_response = OLLAMA_LLM.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract intent from the user's latest message and return compact JSON with keys: "
                        "intent, entities, urgency, needs_tool. No markdown."
                    )
                ),
                HumanMessage(content=str(latest_human.content)),
            ]
        )
        parsed = _safe_json_load(intent_response.content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


@langsmith_traceable(name="assistant", run_type="chain")
@timed("assistant")
def assistant(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    runtime_messages = _compact_runtime_messages(messages)
    intent_context = _extract_intent_with_ollama(runtime_messages)

    MAX_TOOL_CALLS = 3
    current_count = state.get("tool_call_count", 0)

    system_prompt = _assistant_system_prompt()

    if intent_context:
        system_prompt = (
            f"{system_prompt}\n\nParsed user intent: "
            f"{json.dumps(intent_context, ensure_ascii=True)}"
        )

    model_messages = [SystemMessage(content=system_prompt), *runtime_messages]

    if current_count >= MAX_TOOL_CALLS:
        final_messages = [
            SystemMessage(
                content="You already have enough information. "
                        "Provide the final answer without calling any tools."
            ),
            *runtime_messages,
        ]
        response = LLM.invoke(final_messages)
        tool_calls = []
    else:
        response = invoke_with_resilience(model_messages)
        tool_calls = response.tool_calls or []

    new_count = current_count + len(tool_calls)

    updates: AgentState = {
        "messages": [response],
        "approval_rejected": False,
        "tool_call_count": new_count,
    }

    first_call = tool_calls[0] if tool_calls else None

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
            if (
                output.get("name") == "get_emails_tool"
                and isinstance(output.get("content"), list)
            ):
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
