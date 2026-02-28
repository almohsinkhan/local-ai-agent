import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from app.graph import GUARDED_ACTIONS, build_graph

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    load_dotenv(".env.example")

TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN", "") or "").strip().strip('"').strip("'")
GROQ_API_KEY = (os.getenv("GROQ_API_KEY", "") or "").strip().strip('"').strip("'")

APP = build_graph()
CHAT_CONFIGS: dict[int, dict] = defaultdict(dict)
PENDING_APPROVALS: dict[int, dict] = {}


def _chat_config(chat_id: int) -> dict:
    config = CHAT_CONFIGS.get(chat_id)
    if not config:
        config = {"configurable": {"thread_id": f"telegram-{chat_id}"}}
        CHAT_CONFIGS[chat_id] = config
    return config


def _last_ai_text(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


def _resume_until_waiting(config: dict) -> dict | None:
    while True:
        snapshot = APP.get_state(config)
        if not snapshot.next:
            return None

        if "execute_action" not in snapshot.next:
            APP.invoke(None, config=config)
            continue

        planned = snapshot.values.get("planned_action", {})
        action_name = planned.get("name", "respond")
        action_args = planned.get("args", {})

        if action_name in GUARDED_ACTIONS:
            return {"action": action_name, "args": action_args}

        APP.update_state(config, {"human_approved": True})
        APP.invoke(None, config=config)


async def _send_approval_prompt(message_target, action_name: str, action_args: dict) -> None:
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Approve", callback_data="approval:yes"),
                InlineKeyboardButton("Reject", callback_data="approval:no"),
            ]
        ]
    )
    await message_target.reply_text(
        f"Approval required:\nAction: {action_name}\nArgs: {action_args}",
        reply_markup=keyboard,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text("Agent is ready on Telegram. Send a message.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_chat:
        return

    chat_id = update.effective_chat.id
    if chat_id in PENDING_APPROVALS:
        await update.message.reply_text("Please approve or reject the pending action first.")
        return

    user_message = (update.message.text or "").strip()
    if not user_message:
        return

    config = _chat_config(chat_id)

    try:
        APP.invoke({"messages": [HumanMessage(content=user_message)]}, config=config)
        pending = _resume_until_waiting(config)
        if pending:
            PENDING_APPROVALS[chat_id] = pending
            await _send_approval_prompt(update.message, pending["action"], pending["args"])
            return

        snapshot = APP.get_state(config)
        ai_response = _last_ai_text(snapshot.values.get("messages", [])) or "No response generated."
        await update.message.reply_text(ai_response)
    except Exception as exc:
        await update.message.reply_text(f"Agent error: {exc}")


async def handle_approval(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.callback_query or not update.effective_chat:
        return

    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id

    pending = PENDING_APPROVALS.get(chat_id)
    if not pending:
        await query.edit_message_text("No pending approval found.")
        return

    approved = query.data == "approval:yes"
    config = _chat_config(chat_id)

    try:
        APP.update_state(config, {"human_approved": approved})
        APP.invoke(None, config=config)
        PENDING_APPROVALS.pop(chat_id, None)

        pending_next = _resume_until_waiting(config)
        if pending_next:
            PENDING_APPROVALS[chat_id] = pending_next
            await query.edit_message_text(
                f"Approval {'accepted' if approved else 'rejected'}. Next action needs approval:"
                f"\nAction: {pending_next['action']}\nArgs: {pending_next['args']}"
            )
            await _send_approval_prompt(query.message, pending_next["action"], pending_next["args"])
            return

        snapshot = APP.get_state(config)
        ai_response = _last_ai_text(snapshot.values.get("messages", [])) or "No response generated."
        await query.edit_message_text(f"Approval {'accepted' if approved else 'rejected'}.")
        await query.message.reply_text(ai_response)
    except Exception as exc:
        await query.message.reply_text(f"Agent error: {exc}")


def run_telegram_bot() -> None:
    if not TELEGRAM_BOT_TOKEN:
        print("Missing TELEGRAM_BOT_TOKEN in .env")
        return
    if not GROQ_API_KEY:
        print("Missing GROQ_API_KEY in .env")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_approval, pattern="^approval:(yes|no)$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    run_telegram_bot()
