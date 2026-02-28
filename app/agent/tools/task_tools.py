from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.agent.state import AgentState
from app.agent.tools.common import as_positive_int, require_human_approval, to_tool_content
from app.tools.tasks import add_task, complete_task, list_tasks


def _normalize_identifiers(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return [str(item).strip() for item in value if str(item).strip()]


def _resolve_task_identifier(identifier: str, open_tasks: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Return (task_id, ambiguity_marker)."""
    key = identifier.strip().lower()
    if not key:
        return None, ""

    for task in open_tasks:
        if str(task.get("id", "")).strip().lower() == key:
            return str(task.get("id", "")), None

    exact_title_matches = [
        task for task in open_tasks if str(task.get("title", "")).strip().lower() == key
    ]
    if len(exact_title_matches) == 1:
        return str(exact_title_matches[0].get("id", "")), None
    if len(exact_title_matches) > 1:
        return None, identifier

    contains_matches = [
        task for task in open_tasks if key in str(task.get("title", "")).strip().lower()
    ]
    if len(contains_matches) == 1:
        return str(contains_matches[0].get("id", "")), None
    if len(contains_matches) > 1:
        return None, identifier

    return None, None

@tool
def add_task_tool(
    titles: list[str] | None = None,
    notes: str = "",
    due_iso: str = "",
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """
    Create one or more tasks in the user's to-do list.

    Use when:
    - The user asks to add or create tasks.
    - The user mentions multiple tasks separated by commas or 'and'.

    Arguments:
    - titles: list of task titles.
    - notes: optional extra details.
    - due_iso: optional ISO8601 datetime string.

    Safety:
    - Requires human approval before execution.
    - Will not create empty-title tasks.
    """

    require_human_approval(state)

    if not titles:
        raise ValueError("Missing required field: titles")

    created = []
    errors = []

    for raw_title in titles:
        clean_title = str(raw_title or "").strip()
        if not clean_title:
            errors.append("Empty title skipped")
            continue

        try:
            add_task(
                title=clean_title,
                notes=str(notes or ""),
                due_iso=str(due_iso or "").strip(),
            )
            created.append(clean_title)
        except Exception as e:
            errors.append({clean_title: str(e)})

    return to_tool_content({"created": created, "errors": errors})

@tool
def list_tasks_tool(due_min_iso: str = "", due_max_iso: str = "", max_results: int = 10) -> str:
    """List tasks with optional due-date filtering.

    Use when:
    - The user asks to view tasks.

    What arguments it expects:
    - due_min_iso, due_max_iso (optional).
    - max_results.

    What not to do:
    - Do not mark tasks complete with this tool.

    Special behavior:
    - Relies on deterministic date parsing in tasks layer.
    """
    result = list_tasks(
        due_min_iso=str(due_min_iso or "").strip(),
        due_max_iso=str(due_max_iso or "").strip(),
        max_results=as_positive_int(max_results, 10, max_value=50),
    )
    return to_tool_content(result)


@tool
def complete_task_tool(
    task_identifiers: str | list[str] | None = None,
    complete_all: bool = False,
    state: Annotated[AgentState, InjectedState] = None,
) -> str:
    """Complete one, many, or all tasks with deterministic title-to-ID resolution.

    Use when:
    - The user asks to complete tasks.

    What arguments it expects:
    - task_identifiers: id/title string, comma-separated string, or list.
    - complete_all: if true, complete every open task.

    What not to do:
    - Do not invent IDs.
    - Do not execute without approval.

    Special behavior:
    - Resolves title -> ID inside tool logic.
    - Supports single, multi, and complete-all flows.
    - Returns `not_found` and `ambiguous` instead of guessing.
    """
    require_human_approval(state)

    open_tasks = list_tasks(max_results=100)
    if not open_tasks:
        return to_tool_content(
            {"completed": [], "count": 0, "not_found": [], "ambiguous": [], "message": "No open tasks."}
        )

    if complete_all:
        completed_titles: list[str] = []
        for task in open_tasks:
            task_id = str(task.get("id", "")).strip()
            if not task_id:
                continue
            complete_task(task_ids=[task_id])
            completed_titles.append(str(task.get("title", "")))
        return to_tool_content({
            "completed": completed_titles,
            "count": len(completed_titles),
            "not_found": [],
            "ambiguous": [],
        })

    identifiers = _normalize_identifiers(task_identifiers)
    if not identifiers:
        raise ValueError("Provide task_identifiers or set complete_all=true.")

    resolved_ids: list[str] = []
    not_found: list[str] = []
    ambiguous: list[str] = []
    completed_titles: list[str] = []

    for raw_identifier in identifiers:
        task_id, maybe_ambiguous = _resolve_task_identifier(raw_identifier, open_tasks)
        if task_id:
            resolved_ids.append(task_id)
            continue
        if maybe_ambiguous is None:
            not_found.append(raw_identifier)
        else:
            ambiguous.append(raw_identifier)

    for task_id in resolved_ids:
        complete_task(task_ids=[task_id])
        match = next((task for task in open_tasks if str(task.get("id", "")) == task_id), None)
        if match:
            completed_titles.append(str(match.get("title", "")))

    return to_tool_content({
        "completed": completed_titles,
        "count": len(completed_titles),
        "not_found": not_found,
        "ambiguous": ambiguous,
    })
