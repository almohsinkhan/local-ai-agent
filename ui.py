import os
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import GUARDED_ACTIONS, build_graph

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    load_dotenv(".env.example")

st.set_page_config(page_title="Local Agent UI", page_icon="🤖", layout="wide")


@st.cache_resource
def get_app():
    return build_graph()


def init_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "config" not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _sync_pending_approval(app) -> None:
    snapshot = app.get_state(st.session_state.config)
    if not snapshot.next or "execute_action" not in snapshot.next:
        st.session_state.pending_approval = None
        return

    planned = snapshot.values.get("planned_action", {})
    action_name = planned.get("name", "respond")

    if action_name in GUARDED_ACTIONS:
        st.session_state.pending_approval = {
            "action": action_name,
            "args": planned.get("args", {}),
        }
    else:
        app.update_state(st.session_state.config, {"human_approved": True})
        app.invoke(None, config=st.session_state.config)
        _sync_pending_approval(app)


def _run_user_turn(app, user_text: str) -> None:
    app.invoke({"messages": [HumanMessage(content=user_text)]}, config=st.session_state.config)
    _sync_pending_approval(app)


def _resume_with_approval(app, approved: bool) -> None:
    app.update_state(st.session_state.config, {"human_approved": approved})
    app.invoke(None, config=st.session_state.config)
    _sync_pending_approval(app)


def _render_chat(app) -> None:
    snapshot = app.get_state(st.session_state.config)
    messages = snapshot.values.get("messages", [])

    if not messages:
        st.info("Ask something like: 'Show unread emails' or 'Find a 30-min free slot tomorrow'.")
        return

    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(_message_text(msg))
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.code(_message_text(msg), language="json")


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY. Set it in .env or export it in your shell.")
        st.code("export GROQ_API_KEY='your_key_here'")
        st.stop()

    init_state()
    app = get_app()

    st.title("LangGraph Local Agent")
    st.caption("ChatGroq + Gmail + Calendar + Search with human approval before send/calendar actions")

    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("New Session", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
            st.session_state.pending_approval = None
            st.rerun()

    _sync_pending_approval(app)

    pending = st.session_state.pending_approval
    if pending:
        st.warning("Approval required before executing a sensitive action.")
        st.write(f"Action: `{pending['action']}`")
        st.json(pending["args"])

        approve_col, reject_col = st.columns(2)
        with approve_col:
            if st.button("Approve", type="primary", use_container_width=True):
                _resume_with_approval(app, True)
                st.rerun()
        with reject_col:
            if st.button("Reject", use_container_width=True):
                _resume_with_approval(app, False)
                st.rerun()

    _render_chat(app)

    prompt = st.chat_input("Message the agent")
    if prompt:
        _run_user_turn(app, prompt)
        st.rerun()


if __name__ == "__main__":
    main()
