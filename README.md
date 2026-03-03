# Local AI Agent (LangGraph + Groq + Ollama Fallback)

A local assistant built with a hybrid LangGraph architecture:

- `LLM.bind_tools(...)` for native tool calling
- `ToolNode` for execution
- guarded approval gate before sensitive actions
- deterministic task title-to-ID resolution inside tool logic
- Groq-first inference with retry/backoff and Ollama fallback (tool-calling preserved)

## Features

- Email actions
  - Read inbox emails with query filters
  - Send emails (approval required)
- Calendar actions
  - List events
  - Add events (approval required)
- Tasks actions
  - Add task (approval required)
  - List tasks
  - Complete one, many, or all tasks (approval required)
- Search actions
  - Web search (Tavily if configured, fallback to DuckDuckGo)
  - Latest news headlines from RSS feeds
- Interfaces
  - CLI chat (`main.py`)
  - Streamlit UI (`ui.py`)
  - Telegram bot (`telegram_bot.py`)
- Observability
  - LangSmith tracing
  - local timing logs
- Runtime resilience
  - retries Groq on overload/rate-limit errors (`503`, `429`, capacity/rate-limit messages)
  - exponential backoff + jitter
  - falls back to Ollama with `bind_tools(...)` so graph routing remains consistent

## Architecture

Graph flow:

`START -> assistant -> execute_action -> tools -> assistant -> END`

Where:
- `assistant` = LLM node with bound tools
- `execute_action` = approval gate node
- `tools` = `langgraph.prebuilt.ToolNode`

State is minimal and structured (`AgentState`) with:
- `messages`
- `human_approved`
- `planned_action`
- `approval_rejected`
- `last_email_results`
- `last_tool_result`

## Project Structure

```text
.
├── app/
│   ├── graph.py
│   ├── agent/
│   │   ├── state.py
│   │   ├── runtime.py
│   │   ├── tooling.py
│   │   └── tools/
│   │       ├── common.py
│   │       ├── gmail_tools.py
│   │       ├── calendar_tools.py
│   │       ├── search_tools.py
│   │       └── task_tools.py
│   └── tools/
│       ├── gmail.py
│       ├── calendar.py
│       ├── tasks.py
│       └── search.py
├── main.py
├── ui.py
├── telegram_bot.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- Groq API key
- Ollama (optional, for fallback) with a local pulled model
- Google OAuth desktop credentials (`credentials.json`) for Gmail/Calendar/Tasks

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

Optional Ollama integration dependency:

```bash
pip install langchain-ollama
```

3. Create `.env`:

```bash
cp .env.example .env
```

4. Set required values:

```env
GROQ_API_KEY=your_groq_api_key
```

5. Optional Ollama fallback values:

```env
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

6. Optional search API key:

```env
TAVILY_API_KEY=your_tavily_key
```

7. Optional tracing/timing:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=local-agent
ENABLE_TIME_TRACKING=true
```

## Google OAuth Setup

1. Enable in Google Cloud:
- Gmail API
- Google Calendar API
- Google Tasks API

2. Create OAuth client credentials (Desktop app).

3. Save credentials file as `credentials.json` in project root.

4. Ensure scopes include Gmail, Calendar, and Tasks:

```env
GOOGLE_SCOPES=https://www.googleapis.com/auth/gmail.modify,https://www.googleapis.com/auth/gmail.send,https://www.googleapis.com/auth/calendar,https://www.googleapis.com/auth/tasks
```

On first authenticated action, browser-based OAuth runs and `token.json` is created/updated.

## Run

CLI:

```bash
python main.py
```

Streamlit:

```bash
streamlit run ui.py
```

Telegram:

```bash
python telegram_bot.py
```

## Approval Flow

Sensitive actions are interrupted before execution and require explicit approval:

- `send_email_tool`
- `add_event_tool`
- `add_task_tool`
- `complete_task_tool`

## Checkpointing

- Default: in-memory checkpointing (`MemorySaver`)
- Optional: pass `sqlite_path` to `build_graph(sqlite_path="...")` to use SQLite saver if available

## Environment Variables

Common vars:

- `GROQ_MODEL`
- `GROQ_API_KEY`
- `OLLAMA_MODEL` (optional)
- `OLLAMA_BASE_URL` (optional)
- `TAVILY_API_KEY` (optional)
- `GOOGLE_CLIENT_SECRET_FILE`
- `GOOGLE_TOKEN_FILE`
- `GOOGLE_SCOPES`
- `GMAIL_USER_ID`
- `CALENDAR_ID`
- `TIMEZONE`
- `TELEGRAM_BOT_TOKEN` (for Telegram bot)
- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_ENDPOINT`
- `ENABLE_TIME_TRACKING`

## Troubleshooting

- `Missing GROQ_API_KEY`
  - Add `GROQ_API_KEY` to `.env`.
- Groq overload/rate-limit errors
  - Runtime retries automatically before fallback.
  - Ensure Ollama is running locally if fallback is expected: `ollama serve`
  - Ensure configured `OLLAMA_MODEL` is pulled: `ollama pull <model>`
- Google permission errors
  - Ensure APIs are enabled and scopes include Tasks if task operations fail.
  - Delete `token.json` and re-authenticate if scopes changed.
- Empty search results
  - Retry with a different query or configure `TAVILY_API_KEY`.
