from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PlannedAction(TypedDict, total=False):
    name: str
    args: dict[str, Any]


class AgentState(TypedDict, total=False):
    """Shared graph state.

    Keep this minimal so persistence/checkpoint evolution stays simple.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    human_approved: bool | None
    planned_action: PlannedAction
    approval_rejected: bool
    last_email_results: list[dict[str, Any]]
    last_tool_result: Any
