from __future__ import annotations

from typing import Any, Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.agent.state import AgentState
from app.agent.tools.common import as_positive_int, require_human_approval, to_tool_content
from app.tools.gmail import get_emails, send_email


@tool
def get_emails_tool(query: str = "", max_results: int = 5) -> str:
    """Read inbox emails with deterministic query handling.

    Use when:
    - The user asks to read, search, or check inbox emails.

    What arguments it expects:
    - query: Gmail query string.
    - max_results: number of emails to return (1..25).

    What not to do:
    - Do not use this to send or change emails.

    Special behavior:
    - Search is always scoped to `in:inbox category:primary`.
    """
    base_query = "in:inbox category:primary"
    user_query = str(query or "").strip()
    final_query = f"{base_query} {user_query}".strip()
    result = get_emails(query=final_query, max_results=as_positive_int(max_results, 5, max_value=25))
    return to_tool_content(result)


@tool
def send_email_tool(
    to: str,
    subject: str,
    body: str,
    state: Annotated[AgentState, InjectedState],
) -> str:
    """Send an email only after approval.

    Use when:
    - The user clearly asks to send an email.

    What arguments it expects:
    - to: recipient email address.
    - subject: email subject.
    - body: email body.

    What not to do:
    - Do not execute if recipient is missing.
    - Do not use for draft-only requests.

    Special behavior:
    - Requires `human_approved=True` in state.
    """
    require_human_approval(state)
    recipient = str(to or "").strip()
    if not recipient:
        raise ValueError("Missing required field: to")
    result = send_email(to=recipient, subject=str(subject or ""), body=str(body or ""))
    return to_tool_content(result)
