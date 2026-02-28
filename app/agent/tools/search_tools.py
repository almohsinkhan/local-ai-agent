from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.agent.tools.common import as_positive_int, to_tool_content
from app.tools.calendar import get_current_time_iso
from app.tools.search import get_latest_news, web_search


@tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """Run web search for external public information.

    Use when:
    - The user asks to search the web.

    What arguments it expects:
    - query: search text.
    - max_results: number of results.

    What not to do:
    - Do not use for mailbox/calendar/tasks operations.

    Special behavior:
    - Reuses provider fallback logic from existing search implementation.
    """
    clean_query = str(query or "").strip()
    if not clean_query:
        raise ValueError("Missing required field: query")
    result = web_search(query=clean_query, max_results=as_positive_int(max_results, 5, max_value=15))
    return to_tool_content(result)


@tool
def get_latest_news_tool(max_results: int = 5) -> str:
    """Fetch latest headline summaries from configured feeds.

    Use when:
    - The user asks for top/latest news.

    What arguments it expects:
    - max_results: number of headlines.

    What not to do:
    - Do not use for deep domain research.

    Special behavior:
    - Merges multiple feeds, then caps the final list.
    """
    result = get_latest_news(max_results=as_positive_int(max_results, 5, max_value=20))
    return to_tool_content(result)


@tool
def get_current_time_iso_tool() -> str:
    """Return current configured local time in ISO format.

    Use when:
    - The user asks current time/date.

    What arguments it expects:
    - None.

    What not to do:
    - Do not use for event/task mutations.

    Special behavior:
    - Uses assistant effective timezone settings.
    """
    return get_current_time_iso()
