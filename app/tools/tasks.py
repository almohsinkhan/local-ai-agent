from datetime import datetime, timezone
from typing import Any

from googleapiclient.discovery import build

from app.observability import langsmith_traceable, timed
from app.tools.calendar import _get_credentials, _parse_iso_datetime, normalize_to_calendar_tz


def _tasks_service():
    creds = _get_credentials()
    return build("tasks", "v1", credentials=creds)


def _to_utc_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@langsmith_traceable(name="add_task", run_type="tool")
@timed("add_task")
def add_task(title: str, notes: str = "", due_iso: str = "") -> dict[str, Any]:
    """Create a task. Keep this behind human approval in the graph."""
    service = _tasks_service()
    clean_title = str(title or "").strip()
    if not clean_title:
        raise ValueError("Missing required field: title")

    task_body = {
        "title": clean_title,
        "notes": str(notes or ""),
    }

    if due_iso:
        due_dt = _parse_iso_datetime(due_iso, "due_iso")
        due_dt = normalize_to_calendar_tz(due_dt)
        task_body["due"] = _to_utc_rfc3339(due_dt)

    created = service.tasks().insert(tasklist="@default", body=task_body).execute()
    return {
        "id": created.get("id"),
        "title": created.get("title"),
        "notes": created.get("notes"),
        "due": created.get("due"),
    }


@langsmith_traceable(name="list_tasks", run_type="tool")
@timed("list_tasks")
def list_tasks(due_min_iso: str = "", due_max_iso: str = "", max_results: int = 10) -> list[dict[str, Any]]:
    """List tasks, optionally filtering by due date range."""
    service = _tasks_service()

    query = {}
    if due_min_iso:
        due_min_dt = _parse_iso_datetime(due_min_iso, "due_min_iso")
        due_min_dt = normalize_to_calendar_tz(due_min_dt)
        query["dueMin"] = _to_utc_rfc3339(due_min_dt)
    if due_max_iso:
        due_max_dt = _parse_iso_datetime(due_max_iso, "due_max_iso")
        due_max_dt = normalize_to_calendar_tz(due_max_dt)
        query["dueMax"] = _to_utc_rfc3339(due_max_dt)

    tasks_result = service.tasks().list(tasklist="@default", **query, maxResults=max_results).execute()
    return tasks_result.get("items", [])


@langsmith_traceable(name="complete_task", run_type="tool")
@timed("complete_task")
def complete_task(task_id: str) -> dict[str, Any]:
    """Mark a task as completed by ID. Keep this behind human approval in the graph."""
    service = _tasks_service()
    now_iso = _to_utc_rfc3339(datetime.now(timezone.utc))
    updated = service.tasks().patch(
        tasklist="@default",
        task=task_id,
        body={"status": "completed", "completed": now_iso},
    ).execute()
    return {
        "id": updated.get("id"),
        "title": updated.get("title"),
        "status": updated.get("status"),
        "completed": updated.get("completed"),
    }
