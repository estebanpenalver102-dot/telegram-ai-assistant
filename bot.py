#!/usr/bin/env python3
"""
Telegram AI Assistant — v2
Features: Web search, reminders, Google Sheets notes, Google Calendar
AI: Gemini 2.0 Flash (free tier) with function calling via google-genai SDK
"""
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import pytz

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
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

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN     = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY     = os.environ["GEMINI_API_KEY"]
ALLOWED_USER_IDS   = os.environ.get("ALLOWED_USER_IDS", "")
GOOGLE_SA_JSON     = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_SHEET_ID    = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_CALENDAR_ID = os.environ.get("GOOGLE_CALENDAR_ID", "primary")
USER_TIMEZONE      = os.environ.get("USER_TIMEZONE", "America/New_York")

ALLOWED_IDS = set(int(x.strip()) for x in ALLOWED_USER_IDS.split(",") if x.strip())

# ─── Gemini client ───────────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL  = "gemini-2.0-flash"

SYSTEM_PROMPT = (
    "You are a smart personal AI assistant on Telegram. "
    "You proactively use your tools when needed: search the web for current info, "
    "set reminders, save notes to Google Sheets, and check calendar events. "
    "Be concise and practical. Use plain text, no markdown. "
    "When setting a reminder, always confirm the exact time. "
    "When searching, summarize findings briefly."
)

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="web_search",
        description="Search the web for current news, facts, prices, weather, or any real-time info.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"query": types.Schema(type="STRING", description="Search query")},
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="set_reminder",
        description="Set a reminder that will ping the user after N minutes.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "message": types.Schema(type="STRING", description="Reminder message"),
                "minutes": types.Schema(type="INTEGER", description="Minutes from now"),
            },
            required=["message", "minutes"],
        ),
    ),
    types.FunctionDeclaration(
        name="save_note",
        description="Save a note or piece of information to Google Sheets.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "title":   types.Schema(type="STRING", description="Short title"),
                "content": types.Schema(type="STRING", description="Note body"),
            },
            required=["title", "content"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_calendar",
        description="Fetch upcoming calendar events.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"days": types.Schema(type="INTEGER", description="Days ahead (default 7)")},
            required=[],
        ),
    ),
    types.FunctionDeclaration(
        name="get_datetime",
        description="Get current date and time.",
        parameters=types.Schema(type="OBJECT", properties={}, required=[]),
    ),
]

GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
    ),
)

# ─── Scheduler ───────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler(timezone=USER_TIMEZONE)

# ─── Conversation history (per user) ─────────────────────────────────────────
user_histories: dict[int, list] = {}
_bot_app: Optional[Application] = None


# ─── Tool implementations ─────────────────────────────────────────────────────

def _web_search(query: str) -> str:
    if not DDG_AVAILABLE:
        return "Web search is unavailable."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        lines = [f"\u2022 {r.get('title','')}\n  {r.get('body','')}" for r in results[:5]]
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def _save_note(title: str, content: str) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON or not GOOGLE_SHEET_ID:
        return "\u26a0\ufe0f Google Sheets not configured (set GOOGLE_SERVICE_ACCOUNT_JSON + GOOGLE_SHEET_ID)."
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
        return f'\u2705 Note saved: "{title}"'
    except Exception as e:
        return f"Sheets error: {e}"


def _get_calendar(days: int = 7) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON:
        return "\u26a0\ufe0f Google Calendar not configured (set GOOGLE_SERVICE_ACCOUNT_JSON)."
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
            lines.append(f"\u2022 {e.get('summary', 'Untitled')} \u2014 {ts}")
        return "\n".join(lines)
    except Exception as e:
        return f"Calendar error: {e}"


def _get_datetime() -> str:
    tz = pytz.timezone(USER_TIMEZONE)
    return datetime.now(tz).strftime("%A, %B %d, %Y at %I:%M %p %Z")


def _dispatch_tool(name: str, args: dict, chat_id: int, user_id: int) -> str:
    if name == "web_search":
        return _web_search(args.get("query", ""))
    elif name == "set_reminder":
        mins = int(args.get("minutes", 5))
        msg  = args.get("message", "Reminder!")
        tz   = pytz.timezone(USER_TIMEZONE)
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
            await _bot_app.bot.send_message(chat_id=chat_id, text=f"\u23f0 Reminder: {message}")
        except Exception as e:
            logger.error(f"Reminder send failed: {e}")


def is_allowed(user_id: int) -> bool:
    return not ALLOWED_IDS or user_id in ALLOWED_IDS


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("Not authorized.")
        return
    await update.message.reply_text(
        "\U0001f44b Hey! I'm your personal AI assistant.\n\n"
        "I can:\n"
        "\U0001f50d Search the web for current info\n"
        "\u23f0 Set reminders (\"remind me in 30 mins to call John\")\n"
        "\U0001f4dd Save notes to Google Sheets (\"save note: ...\")\n"
        "\U0001f4c5 Check your calendar (\"what's on my calendar this week?\")\n"
        "\U0001f4ac Answer anything else\n\n"
        "/clear \u2014 fresh conversation\n"
        "/help \u2014 show this"
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_histories.pop(update.effective_user.id, None)
    await update.message.reply_text("\u2705 Conversation cleared!")


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
    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    try:
        for _ in range(6):
            response = client.models.generate_content(
                model=MODEL,
                contents=history,
                config=GEN_CONFIG,
            )

            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                break

            history.append(candidate.content)

            fn_calls = [
                p for p in candidate.content.parts
                if p.function_call and p.function_call.name
            ]
            if not fn_calls:
                break

            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            result_parts = []
            for p in fn_calls:
                fc = p.function_call
                args = dict(fc.args) if fc.args else {}
                logger.info(f"Tool: {fc.name}({args})")
                result = _dispatch_tool(fc.name, args, chat_id, user_id)
                result_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": result},
                        )
                    )
                )
            history.append(types.Content(role="user", parts=result_parts))

        reply = ""
        last_model = next((c for c in reversed(history) if c.role == "model"), None)
        if last_model:
            for part in last_model.parts:
                if hasattr(part, "text") and part.text:
                    reply += part.text

        reply = reply.strip() or "Done!"

        if len(history) > 40:
            user_histories[user_id] = history[-40:]

        if len(reply) > 4096:
            for i in range(0, len(reply), 4096):
                await update.message.reply_text(reply[i:i+4096])
        else:
            await update.message.reply_text(reply)

    except Exception as e:
        logger.error(f"handle_message error: {e}", exc_info=True)
        await update.message.reply_text("\u26a0\ufe0f Something went wrong. Try again!")


async def post_init(app: Application):
    global _bot_app
    _bot_app = app
    scheduler.start()
    logger.info("Bot v2 running \u2014 Gemini 2.0 Flash + new SDK.")


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

    logger.info("Starting bot...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
