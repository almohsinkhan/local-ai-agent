from app.agent.tools import (
    add_event_tool,
    add_task_tool,
    complete_task_tool,
    get_current_time_iso_tool,
    get_emails_tool,
    get_latest_news_tool,
    list_events_tool,
    list_tasks_tool,
    send_email_tool,
    web_search_tool,
)

# Any tool in this set requires explicit approval in the approval gate node.
GUARDED_ACTIONS = {
    "send_email_tool",
    "add_event_tool",
    "add_task_tool",
    "complete_task_tool",
}

TOOLS = [
    get_emails_tool,
    send_email_tool,
    list_events_tool,
    add_event_tool,
    web_search_tool,
    get_latest_news_tool,
    get_current_time_iso_tool,
    add_task_tool,
    list_tasks_tool,
    complete_task_tool,
]

__all__ = ["GUARDED_ACTIONS", "TOOLS"]
