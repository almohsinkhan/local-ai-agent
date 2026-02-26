# Local AI Agent (LangGraph + Groq)

A local AI assistant that uses LangGraph orchestration with Groq LLMs and optional external tools for Gmail, Google Calendar, web search, and RSS news.

## Features

- Email actions
  - Read inbox messages with query filters
  - Analyze and summarize relevant emails
  - Send emails (requires explicit approval)
- Calendar actions
  - List events
  - Create events (requires explicit approval)
- Search actions
  - Web search (Tavily if configured, otherwise DuckDuckGo)
  - Latest news headlines from RSS feeds
- Observability
  - LangSmith tracing for graph nodes, tools, and turn execution
  - Local execution timing logs for key steps
- Two interfaces
  - CLI chat (`main.py`)
  - Streamlit UI (`ui.py`)

## Tech Stack

- Python
- LangGraph / LangChain Core
- Groq (`langchain-groq`)
- Google APIs (Gmail + Calendar OAuth)
- DDGS / Tavily
- Streamlit
- LangSmith SDK

## Project Structure

```text
.
├── app/
│   ├── graph.py
│   └── tools/
│       ├── gmail.py
│       ├── calendar.py
│       └── search.py
├── main.py
├── ui.py
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.10+
- A Groq API key
- Google Cloud OAuth client credentials (`credentials.json`) for Gmail/Calendar features

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your environment file:

```bash
cp .env.example .env
```

4. Edit `.env` and set at least:

```env
GROQ_API_KEY=your_groq_api_key
```

5. (Optional but recommended) Set search API key:

```env
TAVILY_API_KEY=your_tavily_key
```

If `TAVILY_API_KEY` is not set, web search falls back to DuckDuckGo.

6. Enable tracing and timing:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=local-agent
ENABLE_TIME_TRACKING=true
```

With `ENABLE_TIME_TRACKING=true`, the app prints timing lines like:
- `[timing] plan_action: 120.52 ms`
- `[timing] chat_turn: 954.10 ms`

## Google OAuth Setup (Gmail + Calendar)

1. In Google Cloud Console, enable:
- Gmail API
- Google Calendar API

2. Create OAuth client credentials (Desktop app).

3. Download the JSON file and place it at project root as:
- `credentials.json`

4. Keep these env vars aligned with your files:

```env
GOOGLE_CLIENT_SECRET_FILE=credentials.json
GOOGLE_TOKEN_FILE=token.json
GOOGLE_SCOPES=https://www.googleapis.com/auth/gmail.modify,https://www.googleapis.com/auth/gmail.send,https://www.googleapis.com/auth/calendar
```

On first Gmail/Calendar use, a browser auth flow opens and stores access tokens in `token.json`.

## Run

CLI:

```bash
python main.py
```

Streamlit UI:

```bash
streamlit run ui.py
```

## Approval Flow

Sensitive actions are guarded and require explicit approval before execution:

- `send_email`
- `add_event`

In CLI, you will see an approval prompt. In Streamlit, Approve/Reject buttons are shown.

## Environment Variables

Defined in `.env.example`:

- `GROQ_MODEL` (default: `openai/gpt-oss-120b`)
- `GROQ_API_KEY`
- `LANGSMITH_TRACING` (default in sample: `true`)
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT` (example: `local-agent`)
- `LANGSMITH_ENDPOINT` (optional; defaults to LangSmith cloud endpoint)
- `ENABLE_TIME_TRACKING` (default in sample: `true`)
- `TAVILY_API_KEY` (optional)
- `GOOGLE_CLIENT_SECRET_FILE`
- `GOOGLE_TOKEN_FILE`
- `GOOGLE_SCOPES`
- `GMAIL_USER_ID` (default: `me`)
- `CALENDAR_ID` (default: `primary`)
- `TIMEZONE` (default: `UTC`)

## Troubleshooting

- `Missing GROQ_API_KEY`
  - Add `GROQ_API_KEY` to `.env` and restart.
- Google auth errors
  - Ensure `credentials.json` exists and APIs are enabled in Google Cloud.
  - Delete `token.json` and re-authenticate if scopes changed.
- Search results empty
  - Try again with a different query or configure `TAVILY_API_KEY`.
