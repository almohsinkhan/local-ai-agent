from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.runtime import assistant, execute_action, route_after_assistant, route_after_execute
from app.agent.state import AgentState
from app.agent.tooling import GUARDED_ACTIONS, TOOLS


def _build_checkpointer(sqlite_path: str | None = None):
    if sqlite_path:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            return SqliteSaver.from_conn_string(sqlite_path)
        except Exception:
            pass
    return MemorySaver()


def build_graph(sqlite_path: str | None = None):
    """Build hybrid assistant graph.

    Flow:
    START -> assistant -> execute_action -> tools -> assistant -> END
    """
    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant)
    builder.add_node("execute_action", execute_action)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {"execute_action": "execute_action", "end": END},
    )
    builder.add_conditional_edges(
        "execute_action",
        route_after_execute,
        {"tools": "tools", "assistant": "assistant"},
    )
    builder.add_edge("tools", "assistant")

    return builder.compile(
        checkpointer=_build_checkpointer(sqlite_path),
        interrupt_before=["execute_action"],
    )


__all__ = ["build_graph", "GUARDED_ACTIONS"]
