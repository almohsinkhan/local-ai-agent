import json
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal, TypedDict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.observability import langsmith_traceable, timed
from app.tools.calendar import add_event, effective_timezone_name, get_current_time_iso, list_events
from app.tools.gmail import get_emails, send_email
from app.tools.search import web_search, get_latest_news
from app.tools.tasks import add_task, complete_task, list_tasks

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "moonshotai/kimi-k2-instruct-0905")
GROQ_API_KEY = (os.getenv("GROQ_API_KEY", "") or "").strip().strip('"').strip("'")
GUARDED_ACTIONS = {"send_email", "add_event","add_task"}
ALLOWED_ACTIONS = {
    "respond",
    "get_emails",
    "send_email",
    "list_events",
    "add_event",
    "web_search",
    "get_latest_news",
    "get_current_time_iso",
    "add_task",
    "list_tasks",
    "complete_task"
}

# create llm instance
LLM = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
    api_key=GROQ_API_KEY
)

#STATE 

class PlannedAction(TypedDict, total=False):
    name: Literal[
        "respond",
        "get_emails",
        "send_email",
        "list_events",
        "add_event",
        "web_search",
        "get_latest_news",
        "get_current_time_iso",
        "add_task",
        "list_tasks",
        "complete_task"
    ]
    args: dict[str, Any]
    reason: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    planned_action: PlannedAction
    last_result: Any
    analyzed_emails: list[dict]
    human_approved: bool | None

    last_email_results: list[dict]
    last_actionable_emails: dict | None
    last_task: list[dict]


# UTIL 
def _extract_json(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    try:
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}


def _latest_human_message(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _as_positive_int(value: Any, default: int, *, min_value: int = 1, max_value: int = 50) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(parsed, max_value))


def _normalize_add_event_args(raw_args: dict[str, Any]) -> dict[str, Any]:
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

        # Planner sometimes returns +00:00 even for local-time intents.
        # Keep wall-clock time and drop the UTC offset so calendar timezone is used.
        if parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0):
            args[key] = parsed.replace(tzinfo=None).isoformat(timespec="seconds")
    return args
    
def calculate_priority(email_data: dict) -> int:
    score = 0
    if email_data.get("importance") == "high":
        score += 2
    if email_data.get("requires_reply"):
        score += 1
    if email_data.get("deadline"):
        score += 1
    return score


def _default_calendar_range(days_ahead: int = 7) -> tuple[str, str]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    end = now + timedelta(days=days_ahead)
    return now.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


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
        "Current date/time context for planning calendar tasks:\n"
        f"- UTC now: {now_utc.isoformat().replace('+00:00', 'Z')}\n"
        f"- Local now ({tz_name}): {now_local.isoformat()}\n"
        "Use local timezone for scheduling requests unless user explicitly asks for another timezone. "
        "For add_event start/end, prefer local datetime format without UTC suffix (YYYY-MM-DDTHH:MM:SS). "
        "Use this context for relative dates like today/tomorrow/next week. Do not invent impossible dates."
    )



# PLANNER 
def _planner_prompt() -> str:
    return """
You are an action planner.
Return ONLY valid JSON:
{
  "name": string,
  "args": object,
  "reason": string
}

Allowed actions:
- respond
- get_emails
- send_email
- list_events
- add_event
- web_search
- get_latest_news
- get_current_time_iso
- add_task
- list_tasks
- complete_task
Planning rules:
1) If user asks to check/find/search emails, choose "get_emails".
2) If user mentions "unread", include: is:unread
3) If user mentions sender/person ("from John", "email from Amazon"), include: from:<sender>
4) If user mentions topic/subject ("about internship", "related to project", "subject meeting"),
   include those keywords or subject:<keyword>.
5) Only include filters explicitly requested by the user.
   Never add is:unread unless user asked for unread.
6) Combine filters correctly:
   - "unread emails from Amazon" -> "from:amazon is:unread"
   - "emails about project from John" -> "from:john project"
   - "unread emails about job" -> "job is:unread"
7) For general email search, use get_emails with keyword query only.
8) If user wants to send an email, choose "send_email".
9) If user asks to check/list/show calendar events, choose "list_events".
10) For "list_events", include args when available:
    - time_min: ISO8601 datetime
    - time_max: ISO8601 datetime
    - max_results: integer
11) If user asks to create/schedule/add a calendar event, choose "add_event".
12) For "add_event", include:
    - summary, start_iso, end_iso
    Optional: description, location
    - Resolve relative dates/times (today, tomorrow, next Monday) using provided current date/time context.
    - Use local timezone time values; do not output Z or +00:00 unless user explicitly asks for UTC.
13) If user asks to search web/news, choose "web_search" or "get_latest_news".
14) If no tool fits, choose "respond".
15) For get_emails always include args:
    - query: string

Examples:
User: "unread emails"
{
  "name": "get_emails",
  "args": {"query": "is:unread", "max_results": 5},
  "reason": "Search unread emails"
}

User: "check if there are any email from unstop"
{
  "name": "get_emails",
  "args": {"query": "from:unstop", "max_results": 5},
  "reason": "Search by sender"
}

User: "any unread email from unstop"
{
  "name": "get_emails",
  "args": {"query": "from:unstop is:unread", "max_results": 5},
  "reason": "Search unread from sender"
}

User: "is there any email about internship from unstop"
{
  "name": "get_emails",
  "args": {"query": "from:unstop internship", "max_results": 5},
  "reason": "Search by sender + keyword"
}

User: "find emails about project update"
{
  "name": "get_emails",
  "args": {"query": "project update", "max_results": 5},
  "reason": "Keyword search"
}

16) if user asks for current time, choose "get_current_time_iso" with no args.
17) if user asks to create a task, choose "add_task" with args:
- title: string
- notes: string (optional)
- due_iso: ISO8601 datetime (optional, resolve relative times using current date/time context)
18) if user asks to list tasks, choose "list_tasks" with args:
- due_min_iso: ISO8601 datetime (optional, resolve relative times using current date/time context)
- due_max_iso: ISO8601 datetime (optional, resolve relative times using current date/time context)
- max_results: integer (optional, default 10)
19) if user asks to complete a task(s), choose "complete_task" with args:
- task_ids: list of task titles or IDs 
- If user says "mark all tasks", "complete all tasks",
set args:
{
  "all": true
}
Do not answer the user. Only output valid JSON.
"""


@langsmith_traceable(name="plan_action", run_type="chain")
@timed("plan_action")
def plan_action(state: AgentState) -> AgentState:
    latest_user = _latest_human_message(state.get("messages", []))

    response = LLM.invoke([
        SystemMessage(content=_planner_prompt()),
        SystemMessage(content=_current_datetime_context()),
        HumanMessage(content=latest_user)
    ])

    planned = _extract_json(response.content)

    if planned.get("name") not in ALLOWED_ACTIONS:
        planned = {"name": "respond", "args": {}, "reason": "fallback"}

    if "args" not in planned:
        planned["args"] = {}
    if "reason" not in planned:
        planned["reason"] = "planned action"

    return {
        "planned_action": planned,
        "human_approved": None
    }


def route_after_plan(state: AgentState) -> str:
    if state.get("planned_action", {}).get("name") == "respond":
        return "respond"
    return "execute_action"


# TOOL EXECUTION 

@langsmith_traceable(name="execute_action", run_type="tool")
@timed("execute_action")
def execute_action(state: AgentState) -> AgentState:
    action = state.get("planned_action", {})
    name = action.get("name")
    args = action.get("args", {}) or {}

    if name in GUARDED_ACTIONS and state.get("human_approved") is not True:
        return {"last_result": {"ok": False, "error": "Action not approved"}}

    try:
        if name == "get_emails":
            base_query = "in:inbox category:primary"
            user_query = str(args.get("query", "")).strip()
            query = f"{base_query} {user_query}".strip()
            result = get_emails(
                query=query,
                max_results=_as_positive_int(args.get("max_results"), 5, max_value=25),
            )
            return {
                "last_result": {"ok": True, "action": name, "data": result},
                "last_email_results": result
            }

        elif name == "send_email":
            result = send_email(
                to=args.get("to", ""),
                subject=args.get("subject", ""),
                body=args.get("body", "")
            )

        elif name == "list_events":
            default_min, default_max = _default_calendar_range()
            result = list_events(
                time_min=args.get("time_min", default_min),
                time_max=args.get("time_max", default_max),
                max_results=_as_positive_int(args.get("max_results"), 10, max_value=25),
            )

        elif name == "add_event":
            result = add_event(**_normalize_add_event_args(args))

        elif name == "web_search":
            result = web_search(**args)

        elif name == "get_latest_news":
            result = get_latest_news(**args)

        elif name == "get_current_time_iso":
            result = get_current_time_iso()

        elif name == "add_task":
            result = add_task(
                title=str(args.get("title", "")).strip(),
                notes=str(args.get("notes", "")),
                due_iso=str(args.get("due_iso", "")).strip(),
            )
        elif name == "list_tasks":
            result = list_tasks(
                due_min_iso=str(args.get("due_min_iso", "")).strip(),
                due_max_iso=str(args.get("due_max_iso", "")).strip(),
                max_results=_as_positive_int(args.get("max_results"), 10, max_value=25),
            )

        elif name == "complete_task":
            if args.get("all") is True:
                tasks = list_tasks(max_results=100)

                if not tasks:
                    result = {"message": "No open tasks."}
                else:
                    completed = []

                    for task in tasks:
                        complete_task(task_ids=[task["id"]])
                        completed.append(task["title"])

                    result = {
                        "completed": completed,
                        "count": len(completed)
                    }
            else:
                user_value = args.get("task_ids", [])
                if isinstance(user_value, str):
                    user_value = [user_value]

                task = list_tasks(max_results=50)
                completed = []
                not_found = []

                for identifier in user_value:
                    identifier = identifier.strip().lower()

                    match = next((t for t in task if t.get("id", "").lower() == identifier or t.get("title", "").lower() == identifier), None)

                    if not match:
                        not_found.append(identifier)
                        continue

                    complete_task(task_ids=[match.get("id")])
                    completed.append(match["title"])

                result = {
                    "completed": completed,
                    "not_found": not_found
                }

        else:
            result = {}

        return {"last_result": {"ok": True, "action": name, "data": result}}

    except Exception as e:
        return {"last_result": {"ok": False, "error": str(e)}}


# EMAIL ANALYZER 

@langsmith_traceable(name="analyze_emails", run_type="chain")
@timed("analyze_emails")
def analyze_emails(state: AgentState) -> AgentState:
    last = state.get("last_result", {})
    if not last.get("ok") or last.get("action") != "get_emails":
        return {}

    user_query = _latest_human_message(state.get("messages", []))

    emails = last.get("data", [])
    if not emails:
        return {"analyzed_emails": [{"message": "No emails found."}]}


    analyzed = []

    for email in emails:
        combined_prompt = f"""
User query: {user_query}

Email:
Subject: {email.get("subject")}
From: {email.get("from")}
Body: {email.get("body")}

If relevant to user query:
Return JSON:
{{
 "relevant": true,
 "importance": "",
 "category": "",
 "requires_reply": false,
 "deadline": null,
 "action_items": [],
 "summary": ""
}}

If NOT relevant:
Return:
{{"relevant": false}}
"""

        response = LLM.invoke(combined_prompt)
        data = _extract_json(response.content)

        if not data.get("relevant"):
            continue

        data["priority_score"] = calculate_priority(data)
        data["email_id"] = email.get("id")

        analyzed.append(data)


    if not analyzed:
        return {"analyzed_emails": [{"message": "No relevant emails found."}]}
    

    analyzed.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
      
    return {"analyzed_emails": analyzed,
            "last_actionable_emails": {email["id"]: email for email in emails if email.get("id") in {item.get("email_id") for item in analyzed}}}


# RESPONDER 

@langsmith_traceable(name="respond", run_type="chain")
@timed("respond")
def respond(state: AgentState) -> AgentState:
    action = state.get("planned_action", {}).get("name")
    last_result = state.get("last_result", {})

    # Direct generation
    if action == "respond":
        latest_user = _latest_human_message(state.get("messages", []))
        response = LLM.invoke(latest_user)
        return {"messages": [AIMessage(content=response.content)]}

    # Return explicit tool errors directly instead of rephrasing with the LLM.
    if isinstance(last_result, dict) and not last_result.get("ok", True):
        error = str(last_result.get("error", "Tool execution failed."))
        return {"messages": [AIMessage(content=f"Could not complete `{action}`: {error}")]}

    # Email summary
    if state.get("analyzed_emails"):

        prompt = f"""
Summarize these results clearly for the user:

{json.dumps(state['analyzed_emails'], indent=2)}
"""
        response = LLM.invoke(prompt)
        return {"messages": [AIMessage(content=response.content)]}

    # Fallback explanation

    prompt = f"""
Tool result:
{json.dumps(last_result, indent=2)}

Explain clearly.
"""
    response = LLM.invoke([
    SystemMessage(content="""
You are a personal desktop assistant.
Speak clearly, simply, and concisely.
Do not explain internal JSON.
Do not mention tools or APIs.
Respond like a helpful human assistant.
Keep answers short unless user asks for details.
"""),
    HumanMessage(content=prompt)
])
    return {"messages": [AIMessage(content=response.content)]}


#  ROUTING 

def route_after_execute(state: AgentState) -> str:
    if state.get("last_result", {}).get("action") == "get_emails":
        return "analyze_emails"
    return "respond"


# GRAPH 

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("plan_action", plan_action)
    builder.add_node("execute_action", execute_action)
    builder.add_node("analyze_emails", analyze_emails)
    builder.add_node("respond", respond)

    builder.add_edge(START, "plan_action")
    builder.add_conditional_edges("plan_action", route_after_plan)
    builder.add_conditional_edges("execute_action", route_after_execute)
    builder.add_edge("analyze_emails", "respond")
    builder.add_edge("respond", END)

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["execute_action"]
    )
