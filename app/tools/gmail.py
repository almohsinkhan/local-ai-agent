import base64
import os
from email.message import EmailMessage
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
GMAIL_USER_ID = os.getenv("GMAIL_USER_ID", "me")


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


def _gmail_service():
    creds = _get_credentials()
    return build("gmail", "v1", credentials=creds)

def _extract_body(payload) -> str:
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
            elif part.get("mimeType") == "text/html":
                return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
    elif "body" in payload and payload["body"].get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
    return ""



def get_emails(query: str = "is:unread", max_results: int = 5) -> list[dict[str, Any]]:
    service = _gmail_service()
    response = (
        service.users()
        .messages()
        .list(userId=GMAIL_USER_ID, q=query, maxResults=max_results)
        .execute()
    )

    messages = response.get("messages", [])
    output = []

    for msg in messages:
        full = (
            service.users()
            .messages()
            .get(userId=GMAIL_USER_ID, id=msg["id"], format="full")
            .execute()
        )

        headers = {
            h["name"].lower(): h["value"]
            for h in full["payload"]["headers"]
        }

        body = _extract_body(full["payload"])

        output.append(
            {
                "id": full["id"],
                "threadId": full["threadId"],
                "from": headers.get("from", ""),
                "subject": headers.get("subject", ""),
                "date": headers.get("date", ""),
                "body": body,
            }
        )

    return output


def send_email(to: str, subject: str, body: str):
    service = _gmail_service()

    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    encoded = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    return (
        service.users()
        .messages()
        .send(userId=GMAIL_USER_ID, body={"raw": encoded})
        .execute()
    )
