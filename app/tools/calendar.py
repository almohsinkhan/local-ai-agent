import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

SCOPES = [
    scope.strip()
    for scope in os.getenv(
        "GOOGLE_SCOPES",
        "https://www.googleapis.com/auth/gmail.modify,https://www.googleapis.com/auth/gmail.send,https://www.googleapis.com/auth/calendar",
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

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds


def _calendar_service():
    creds = _get_credentials()
    return build("calendar", "v3", credentials=creds)


def list_events(time_min: str, time_max: str, max_results: int = 10) -> list[dict[str, Any]]:
    """List events in a time window (ISO8601 timestamps)."""
    service = _calendar_service()
    events_result = (
        service.events()
        .list(
            calendarId=CALENDAR_ID,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=max_results,
        )
        .execute()
    )
    events = events_result.get("items", [])
    output: list[dict[str, Any]] = []

    for event in events:
        output.append(
            {
                "id": event.get("id"),
                "summary": event.get("summary", "(no title)"),
                "start": event.get("start", {}).get("dateTime", event.get("start", {}).get("date")),
                "end": event.get("end", {}).get("dateTime", event.get("end", {}).get("date")),
                "htmlLink": event.get("htmlLink", ""),
            }
        )

    return output


def add_event(summary: str, start_iso: str, end_iso: str, description: str = "", location: str = "") -> dict[str, Any]:
    """Create an event. Keep this behind human approval in the graph."""
    service = _calendar_service()

    event_body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start_iso, "timeZone": TIMEZONE},
        "end": {"dateTime": end_iso, "timeZone": TIMEZONE},
    }

    created = service.events().insert(calendarId=CALENDAR_ID, body=event_body).execute()
    return {
        "id": created.get("id"),
        "summary": created.get("summary"),
        "start": created.get("start", {}).get("dateTime"),
        "end": created.get("end", {}).get("dateTime"),
        "htmlLink": created.get("htmlLink", ""),
    }


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
