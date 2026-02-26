import json
import os
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.tools.calendar import add_event, list_events, now_utc_iso
from app.tools.gmail import get_emails, send_email
from app.tools.search import web_search, get_latest_news

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    load_dotenv(".env.example")

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = (os.getenv("GROQ_API_KEY", "") or "").strip().strip('"').strip("'")
GUARDED_ACTIONS = {"send_email", "add_event"}


# ================= STATE ================= #

class PlannedAction(TypedDict, total=False):
    name: Literal[
        "respond",
        "get_emails",
        "send_email",
        "list_events",
        "add_event",
        "web_search",
        "get_latest_news"
    ]
    args: dict[str, Any]
    reason: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    planned_action: PlannedAction
    last_result: Any
    analyzed_emails: list[dict]
    human_approved: bool | None


# ================= UTIL ================= #

def _extract_json(raw: str) -> dict[str, Any]:
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
        return json.loads(raw)
    except Exception:
        return {}


def _llm(temperature: float = 0) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to .env or export it in your shell.")
    return ChatGroq(model=MODEL_NAME, temperature=temperature, api_key=GROQ_API_KEY)


def calculate_priority(data: dict) -> int:
    score = 0
    if data.get("importance") == "important":
        score += 3
    if data.get("requires_reply"):
        score += 2
    if data.get("deadline"):
        score += 3
    if data.get("category") == "work":
        score += 2
    return score


# ================= PLANNER ================= #

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

Rules for email search planning:
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
9) If request is not an email task, choose "respond".
10) For get_emails always include args:
    - query: string
    - max_results: 20

Examples:
User: "unread emails"
{
  "name": "get_emails",
  "args": {"query": "is:unread", "max_results": 20},
  "reason": "Search unread emails"
}

User: "check if there are any email from unstop"
{
  "name": "get_emails",
  "args": {"query": "from:unstop", "max_results": 20},
  "reason": "Search by sender"
}

User: "any unread email from unstop"
{
  "name": "get_emails",
  "args": {"query": "from:unstop is:unread", "max_results": 20},
  "reason": "Search unread from sender"
}

User: "is there any email about internship from unstop"
{
  "name": "get_emails",
  "args": {"query": "from:unstop internship", "max_results": 20},
  "reason": "Search by sender + keyword"
}

User: "find emails about project update"
{
  "name": "get_emails",
  "args": {"query": "project update", "max_results": 20},
  "reason": "Keyword search"
}

Do not answer the user. Only output valid JSON.
"""


def plan_action(state: AgentState) -> AgentState:
    llm = _llm(temperature=0)

    latest_user = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            latest_user = msg.content
            break

    response = llm.invoke([
        SystemMessage(content=_planner_prompt()),
        HumanMessage(content=latest_user)
    ])

    planned = _extract_json(response.content)

    # Validate action
    allowed = {
        "respond",
        "get_emails",
        "send_email",
        "list_events",
        "add_event",
        "web_search",
        "get_latest_news"
    }

    if planned.get("name") not in allowed:
        planned = {"name": "respond", "args": {}, "reason": "fallback"}

    if "args" not in planned:
        planned["args"] = {}
    if "reason" not in planned:
        planned["reason"] = "planned action"

    if planned.get("name") == "get_emails":
        query = str(planned["args"].get("query", "")).strip()
        planned["args"]["query"] = query
        planned["args"]["max_results"] = 20

    print("PLANNED:", planned)

    return {
        "planned_action": planned,
        "human_approved": None
    }


def route_after_plan(state: AgentState):
    if state.get("planned_action", {}).get("name") == "respond":
        return "respond"
    return "execute_action"


# ================= TOOL EXECUTION ================= #

def execute_action(state: AgentState) -> AgentState:
    action = state.get("planned_action", {})
    name = action.get("name")
    args = action.get("args", {}) or {}

    if name in GUARDED_ACTIONS and state.get("human_approved") is not True:
        return {"last_result": {"ok": False, "error": "Action not approved"}}

    try:
        if name == "get_emails":
            result = get_emails(
                query=args.get("query", ""),
                max_results=int(args.get("max_results", 10))
            )

        elif name == "send_email":
            result = send_email(
                to=args.get("to", ""),
                subject=args.get("subject", ""),
                body=args.get("body", "")
            )

        elif name == "list_events":
            result = list_events(
                time_min=now_utc_iso(),
                time_max=now_utc_iso(),
                max_results=10
            )

        elif name == "add_event":
            result = add_event(**args)

        elif name == "web_search":
            result = web_search(**args)

        elif name == "get_latest_news":
            result = get_latest_news(**args)

        else:
            result = {}

        return {"last_result": {"ok": True, "action": name, "data": result}}

    except Exception as e:
        return {"last_result": {"ok": False, "error": str(e)}}


# ================= EMAIL ANALYZER ================= #

def analyze_emails(state: AgentState) -> AgentState:
    last = state.get("last_result", {})
    if not last.get("ok") or last.get("action") != "get_emails":
        return {}

    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    emails = last.get("data", [])
    llm = _llm(temperature=0)

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

        response = llm.invoke(combined_prompt)
        data = _extract_json(response.content)

        if not data.get("relevant"):
            continue

        data["priority_score"] = calculate_priority(data)
        data["email_id"] = email.get("id")

        analyzed.append(data)

    if not analyzed:
        return {"analyzed_emails": [{"message": "No relevant emails found."}]}

    return {"analyzed_emails": analyzed}


# ================= RESPONDER ================= #

def respond(state: AgentState) -> AgentState:
    action = state.get("planned_action", {}).get("name")

    # Direct generation
    if action == "respond":
        llm = _llm()

        latest_user = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                latest_user = msg.content
                break

        response = llm.invoke(latest_user)
        return {"messages": [AIMessage(content=response.content)]}

    # Email summary
    if state.get("analyzed_emails"):
        llm = _llm()
        prompt = f"""
Summarize these results clearly for the user:

{json.dumps(state['analyzed_emails'], indent=2)}
"""
        response = llm.invoke(prompt)
        return {"messages": [AIMessage(content=response.content)]}

    # Fallback explanation
    llm = _llm()
    prompt = f"""
Tool result:
{json.dumps(state.get("last_result"), indent=2)}

Explain clearly.
"""
    response = llm.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}


# ================= ROUTING ================= #

def route_after_execute(state: AgentState):
    if state.get("last_result", {}).get("action") == "get_emails":
        return "analyze_emails"
    return "respond"


# ================= GRAPH ================= #

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
