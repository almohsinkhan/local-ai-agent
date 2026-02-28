import os
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.observability import langsmith_traceable, timed
load_dotenv()

SCOPES = [
    scope.strip()
    for scope in os.getenv(
        "GOOGLE_SCOPES",
        "https://www.googleapis.com/auth/gmail.modify,https://www.googleapis.com/auth/gmail.send,https://www.googleapis.com/auth/calendar,https://www.googleapis.com/auth/tasks",
    ).split(",")
    if scope.strip()
]
CLIENT_SECRET_FILE = os.getenv("GOOGLE_CLIENT_SECRET_FILE", "credentials.json")
TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")
CALENDAR_ID = os.getenv("CALENDAR_ID", "primary")
TIMEZONE = os.getenv("TIMEZONE", "UTC")

def _get_credentials() -> Credentials:
    creds = None
    token_path = Path(TOKEN_FILE)
    token_granted_scopes: set[str] = set()

    if token_path.exists():
        try:
            token_payload = json.loads(token_path.read_text(encoding="utf-8"))
            raw_scopes = token_payload.get("scopes", [])
            if isinstance(raw_scopes, str):
                token_granted_scopes = {s for s in raw_scopes.split() if s}
            elif isinstance(raw_scopes, list):
                token_granted_scopes = {str(s).strip() for s in raw_scopes if str(s).strip()}
        except Exception:
            token_granted_scopes = set()
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    needs_oauth = not creds
    if creds:
        # Existing token.json can be missing newly added scopes (e.g. Google Tasks).
        # Check granted scopes from token payload because creds.has_scopes can be misleading
        # when credentials are loaded with requested scopes.
        if token_granted_scopes and not set(SCOPES).issubset(token_granted_scopes):
            needs_oauth = True
        elif not token_granted_scopes:
            # Unknown scope state: prefer re-auth to avoid runtime permission failures.
            needs_oauth = True

    if not needs_oauth and creds and not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            needs_oauth = True

    if needs_oauth:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    if creds and not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
    elif creds and not creds.valid:
        # Fallback for any other invalid credential state.
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _calendar_service():
    creds = _get_credentials()
    return build("calendar", "v3", credentials=creds)


def _configured_timezone_name() -> str:
    return (TIMEZONE or "").strip() or "UTC"


@lru_cache(maxsize=1)
def _calendar_timezone_name() -> str:
    try:
        service = _calendar_service()
        calendar = service.calendars().get(calendarId=CALENDAR_ID).execute()
        tz_name = str(calendar.get("timeZone", "")).strip()
        if tz_name:
            return tz_name
    except Exception:
        pass
    return "UTC"


def effective_timezone_name() -> str:
    configured = _configured_timezone_name()
    # If configured timezone is UTC, prefer the calendar's own timezone.
    # This prevents local-time intents (e.g., "4 PM") from shifting unexpectedly.
    if configured.upper() == "UTC":
        return _calendar_timezone_name()
    return configured


def _calendar_tz():
    tz_name = effective_timezone_name()
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc


def normalize_to_calendar_tz(dt: datetime) -> datetime:
    tz = _calendar_tz()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _parse_iso_datetime(value: str, field_name: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Missing required field: {field_name}")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name}: {raw!r}. Use ISO format like '2026-02-28T09:00:00'."
        ) from exc


@langsmith_traceable(name="list_events", run_type="tool")
@timed("list_events")
def list_events(time_min: str, time_max: str, max_results: int = 10):
    service = _calendar_service()

    # Convert to proper datetime objects
    start_dt = _parse_iso_datetime(time_min, "time_min")
    end_dt = _parse_iso_datetime(time_max, "time_max")

    start_dt = normalize_to_calendar_tz(start_dt)
    end_dt = normalize_to_calendar_tz(end_dt)

    events_result = (
        service.events()
        .list(
            calendarId=CALENDAR_ID,
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=max_results,
        )
        .execute()
    )

    return events_result.get("items", [])

@langsmith_traceable(name="add_event", run_type="tool")
@timed("add_event")
def add_event(summary: str, start_iso: str, end_iso: str, description: str = "", location: str = "") -> dict[str, Any]:
    """Create an event. Keep this behind human approval in the graph."""
    start_dt = _parse_iso_datetime(start_iso, "start_iso")
    end_dt = _parse_iso_datetime(end_iso, "end_iso")
    start_dt = normalize_to_calendar_tz(start_dt)
    end_dt = normalize_to_calendar_tz(end_dt)
    if end_dt <= start_dt:
        raise ValueError("Invalid event range: end_iso must be later than start_iso.")

    service = _calendar_service()
    tz_name = effective_timezone_name()

    event_body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_dt.isoformat(timespec="seconds"), "timeZone": tz_name},
        "end": {"dateTime": end_dt.isoformat(timespec="seconds"), "timeZone": tz_name},
    }

    created = service.events().insert(calendarId=CALENDAR_ID, body=event_body).execute()
    return {
        "id": created.get("id"),
        "summary": created.get("summary"),
        "start": created.get("start", {}).get("dateTime"),
        "end": created.get("end", {}).get("dateTime"),
        "htmlLink": created.get("htmlLink", ""),
    }

def get_current_time_iso() -> str:
    """Get the current time in ISO8601 format."""
    tz_name = effective_timezone_name()
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    return datetime.now(tz).replace(microsecond=0).isoformat(timespec="seconds")

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
