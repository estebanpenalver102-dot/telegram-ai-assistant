#!/usr/bin/env python3
"""
Telegram AI Assistant — v4 (webhook mode)
AI: Groq (free tier, llama-3.3-70b) with function calling
Webhook mode: no polling conflicts on Railway
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
import pytz

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Optional Google integrations
try:
    import gspread
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build as google_build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Web search
try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN     = os.environ["TELEGRAM_TOKEN"]
GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
WEBHOOK_URL        = os.environ.get("WEBHOOK_URL", "").rstrip("/")
PORT               = int(os.environ.get("PORT", 8080))
ALLOWED_USER_IDS   = os.environ.get("ALLOWED_USER_IDS", "")
GOOGLE_SA_JSON     = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_SHEET_ID    = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_CALENDAR_ID = os.environ.get("GOOGLE_CALENDAR_ID", "primary")
USER_TIMEZONE      = os.environ.get("USER_TIMEZONE", "America/New_York")
MODEL              = "llama-3.3-70b-versatile"

ALLOWED_IDS = set(int(x.strip()) for x in ALLOWED_USER_IDS.split(",") if x.strip())

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a smart personal AI assistant on Telegram. "
    "Use your tools proactively when needed: search the web for current info, "
    "set reminders, save notes to Google Sheets, and check calendar events. "
    "Be concise and practical. Use plain text — no markdown. "
    "When setting a reminder, confirm the exact time. "
    "When searching, summarize findings briefly."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current news, facts, prices, weather, or any real-time info.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder that will ping the user after N minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Reminder message"},
                    "minutes": {"type": "integer", "description": "Minutes from now"},
                },
                "required": ["message", "minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note to Google Sheets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":   {"type": "string", "description": "Short title"},
                    "content": {"type": "string", "description": "Note body"},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar",
            "description": "Fetch upcoming calendar events.",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "Days ahead (default 7)"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ── Scheduler & state ─────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler(timezone=USER_TIMEZONE)
user_histories: dict[int, list] = {}
_bot_app: Optional[Application] = None


# ── Tool implementations ──────────────────────────────────────────────────────

def _web_search(query: str) -> str:
    if not DDG_AVAILABLE:
        return "Web search unavailable."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(f"• {r.get('title','')}\n  {r.get('body','')}" for r in results[:5])
    except Exception as e:
        return f"Search error: {e}"


def _save_note(title: str, content: str) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON or not GOOGLE_SHEET_ID:
        return "⚠️ Google Sheets not configured."
    try:
        sa = json.loads(GOOGLE_SA_JSON)
        creds = Credentials.from_service_account_info(
            sa, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GOOGLE_SHEET_ID)
        try:
            ws = sh.worksheet("Notes")
        except Exception:
            ws = sh.add_worksheet(title="Notes", rows=1000, cols=4)
            ws.append_row(["Date", "Title", "Content"])
        ts = datetime.now(pytz.timezone(USER_TIMEZONE)).strftime("%Y-%m-%d %H:%M")
        ws.append_row([ts, title, content])
        return f'✅ Note saved: "{title}"'
    except Exception as e:
        return f"Sheets error: {e}"


def _get_calendar(days: int = 7) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON:
        return "⚠️ Google Calendar not configured."
    try:
        sa = json.loads(GOOGLE_SA_JSON)
        creds = Credentials.from_service_account_info(
            sa, scopes=["https://www.googleapis.com/auth/calendar.readonly"]
        )
        svc = google_build("calendar", "v3", credentials=creds)
        tz  = pytz.timezone(USER_TIMEZONE)
        now = datetime.now(tz)
        end = now + timedelta(days=days)
        result = svc.events().list(
            calendarId=GOOGLE_CALENDAR_ID,
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            maxResults=20,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
        if not events:
            return f"No events in the next {days} days."
        lines = []
        for e in events:
            start = e["start"].get("dateTime", e["start"].get("date", ""))
            if "T" in start:
                dt = datetime.fromisoformat(start).astimezone(tz)
                ts = dt.strftime("%a %b %d, %I:%M %p")
            else:
                ts = start
            lines.append(f"• {e.get('summary','Untitled')} — {ts}")
        return "\n".join(lines)
    except Exception as e:
        return f"Calendar error: {e}"


def _get_datetime() -> str:
    return datetime.now(pytz.timezone(USER_TIMEZONE)).strftime("%A, %B %d, %Y at %I:%M %p %Z")


def _dispatch_tool(name: str, args: dict, chat_id: int, user_id: int) -> str:
    if name == "web_search":
        return _web_search(args.get("query", ""))
    elif name == "set_reminder":
        mins   = int(args.get("minutes", 5))
        msg    = args.get("message", "Reminder!")
        tz     = pytz.timezone(USER_TIMEZONE)
        run_at = datetime.now(tz) + timedelta(minutes=mins)
        scheduler.add_job(
            _fire_reminder, "date",
            run_date=run_at,
            args=[chat_id, msg],
            id=f"rem_{user_id}_{run_at.timestamp()}",
        )
        return f"Reminder set for {mins} min from now: '{msg}'"
    elif name == "save_note":
        return _save_note(args.get("title", "Note"), args.get("content", ""))
    elif name == "get_calendar":
        return _get_calendar(int(args.get("days", 7)))
    elif name == "get_datetime":
        return _get_datetime()
    return f"Unknown tool: {name}"


async def _fire_reminder(chat_id: int, message: str):
    global _bot_app
    if _bot_app:
        try:
            await _bot_app.bot.send_message(chat_id=chat_id, text=f"⏰ Reminder: {message}")
        except Exception as e:
            logger.error(f"Reminder failed: {e}")


def is_allowed(user_id: int) -> bool:
    return not ALLOWED_IDS or user_id in ALLOWED_IDS


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("Not authorized.")
        return
    await update.message.reply_text(
        "👋 Hey! I'm your personal AI assistant.\n\n"
        "I can:\n"
        "🔍 Search the web for current info\n"
        "⏰ Set reminders (\"remind me in 30 mins to call John\")\n"
        "📝 Save notes to Google Sheets\n"
        "📅 Check your calendar\n"
        "💬 Answer anything\n\n"
        "/clear — fresh conversation\n"
        "/help — show this"
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_histories.pop(update.effective_user.id, None)
    await update.message.reply_text("✅ Conversation cleared!")


# ── Main message handler ──────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if not is_allowed(user_id):
        await update.message.reply_text("Not authorized.")
        return

    user_text = (update.message.text or "").strip()
    if not user_text:
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    history = user_histories.setdefault(user_id, [])
    history.append({"role": "user", "content": user_text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        for _ in range(6):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]})
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    logger.info(f"Tool: {tc.function.name}({args})")
                    result = _dispatch_tool(tc.function.name, args, chat_id, user_id)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                reply = (msg.content or "").strip() or "Done!"
                history.append({"role": "assistant", "content": reply})
                if len(history) > 40:
                    user_histories[user_id] = history[-40:]
                if len(reply) > 4096:
                    for i in range(0, len(reply), 4096):
                        await update.message.reply_text(reply[i:i+4096])
                else:
                    await update.message.reply_text(reply)
                return

        await update.message.reply_text("Done!")

    except Exception as e:
        logger.error(f"handle_message error: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Something went wrong. Try again!")


# ── Startup ───────────────────────────────────────────────────────────────────

async def post_init(app: Application):
    global _bot_app
    _bot_app = app
    scheduler.start()
    logger.info(f"Bot v4 running — webhook on port {PORT}")


def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if WEBHOOK_URL:
        logger.info(f"Starting in webhook mode: {WEBHOOK_URL}")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            webhook_url=WEBHOOK_URL,
            drop_pending_updates=True,
        )
    else:
        logger.info("No WEBHOOK_URL set — falling back to polling")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
