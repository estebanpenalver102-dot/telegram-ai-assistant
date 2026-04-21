#!/usr/bin/env python3
"""
Telegram AI Assistant — Enhanced
Features: Web search, reminders, Google Sheets notes, Google Calendar
AI: Gemini Flash (free tier) with function calling
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
import google.generativeai as genai
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

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Config ─────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN          = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY          = os.environ["GEMINI_API_KEY"]
ALLOWED_USER_IDS        = os.environ.get("ALLOWED_USER_IDS", "")
GOOGLE_SA_JSON          = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_SHEET_ID         = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_CALENDAR_ID      = os.environ.get("GOOGLE_CALENDAR_ID", "primary")
USER_TIMEZONE           = os.environ.get("USER_TIMEZONE", "America/New_York")

ALLOWED_IDS = set(int(x.strip()) for x in ALLOWED_USER_IDS.split(",") if x.strip())

# ─── Gemini ──────────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

TOOLS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="web_search",
                description="Search the web for current news, facts, prices, weather, or any real-time info.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={"query": genai.protos.Schema(type=genai.protos.Type.STRING, description="Search query")},
                    required=["query"]
                )
            ),
            genai.protos.FunctionDeclaration(
                name="set_reminder",
                description="Set a reminder that will ping the user after N minutes.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "message": genai.protos.Schema(type=genai.protos.Type.STRING, description="Reminder message"),
                        "minutes": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Minutes from now"),
                    },
                    required=["message", "minutes"]
                )
            ),
            genai.protos.FunctionDeclaration(
                name="save_note",
                description="Save a note or piece of information to Google Sheets for future reference.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "title":   genai.protos.Schema(type=genai.protos.Type.STRING, description="Short title"),
                        "content": genai.protos.Schema(type=genai.protos.Type.STRING, description="Note body"),
                    },
                    required=["title", "content"]
                )
            ),
            genai.protos.FunctionDeclaration(
                name="get_calendar",
                description="Fetch upcoming calendar events.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={"days": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Days ahead (default 7)")},
                    required=[]
                )
            ),
            genai.protos.FunctionDeclaration(
                name="get_datetime",
                description="Get current date and time.",
                parameters=genai.protos.Schema(type=genai.protos.Type.OBJECT, properties={}, required=[])
            ),
        ]
    )
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=(
        "You are a smart personal AI assistant on Telegram. "
        "You proactively use your tools when needed: search the web for current info, set reminders, "
        "save notes to Google Sheets, and check calendar events. "
        "Be concise and practical. Use plain text. No markdown unless really helpful. "
        "When setting a reminder, always confirm the exact time. "
        "When searching, summarize findings briefly."
    ),
    tools=TOOLS,
    tool_config={"function_calling_config": {"mode": "AUTO"}}
)

# ─── Scheduler ───────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler(timezone=USER_TIMEZONE)

# ─── State ───────────────────────────────────────────────────────────────────
user_chats: dict[int, genai.ChatSession] = {}
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
        lines = []
        for r in results[:5]:
            lines.append(f"• {r.get('title','')}\n  {r.get('body','')}\n  {r.get('href','')}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def _save_note(title: str, content: str) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON or not GOOGLE_SHEET_ID:
        return "⚠️ Google Sheets not configured (set GOOGLE_SERVICE_ACCOUNT_JSON + GOOGLE_SHEET_ID)."
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
        return f"✅ Note saved: \"{title}\""
    except Exception as e:
        return f"Sheets error: {e}"


def _get_calendar(days: int = 7) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON:
        return "⚠️ Google Calendar not configured (set GOOGLE_SERVICE_ACCOUNT_JSON)."
    try:
        sa = json.loads(GOOGLE_SA_JSON)
        creds = Credentials.from_service_account_info(
            sa, scopes=["https://www.googleapis.com/auth/calendar.readonly"]
        )
        svc = google_build("calendar", "v3", credentials=creds)
        tz = pytz.timezone(USER_TIMEZONE)
        now = datetime.now(tz)
        end = now + timedelta(days=days)
        result = svc.events().list(
            calendarId=GOOGLE_CALENDAR_ID,
            timeMin=now.isoformat(),
            timeMax=end.isoformat(),
            maxResults=20,
            singleEvents=True,
            orderBy="startTime"
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
            lines.append(f"• {e.get('summary', 'Untitled')} — {ts}")
        return "\n".join(lines)
    except Exception as e:
        return f"Calendar error: {e}"


def _get_datetime() -> str:
    tz = pytz.timezone(USER_TIMEZONE)
    return datetime.now(tz).strftime("%A, %B %d, %Y at %I:%M %p %Z")


# ─── Reminder callback ────────────────────────────────────────────────────────

async def _fire_reminder(chat_id: int, message: str):
    global _bot_app
    if _bot_app:
        try:
            await _bot_app.bot.send_message(chat_id=chat_id, text=f"⏰ Reminder: {message}")
        except Exception as e:
            logger.error(f"Reminder send failed: {e}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_allowed(user_id: int) -> bool:
    return not ALLOWED_IDS or user_id in ALLOWED_IDS


def get_chat(user_id: int) -> genai.ChatSession:
    if user_id not in user_chats:
        user_chats[user_id] = model.start_chat()
    return user_chats[user_id]


# ─── Command handlers ────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("Not authorized.")
        return
    await update.message.reply_text(
        "👋 Hey! I'm your personal AI assistant.\n\n"
        "I can:\n"
        "🔍 Search the web for current info\n"
        "⏰ Set reminders (\"remind me in 30 mins to call John\")\n"
        "📝 Save notes to Google Sheets (\"save note: ...\")  \n"
        "📅 Check your calendar (\"what's on my calendar this week?\")\n"
        "💬 Answer anything else\n\n"
        "/clear — fresh conversation\n"
        "/help — show this"
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_chats.pop(update.effective_user.id, None)
    await update.message.reply_text("✅ Conversation cleared!")


# ─── Main message handler ─────────────────────────────────────────────────────

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

    try:
        chat_session = get_chat(user_id)
        response = chat_session.send_message(user_text)

        # Tool call loop
        for _ in range(6):  # max 6 tool rounds
            parts = response.candidates[0].content.parts if response.candidates else []
            fn_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call.name]
            if not fn_calls:
                break

            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            tool_results = []

            for p in fn_calls:
                fc = p.function_call
                name = fc.name
                args = dict(fc.args) if fc.args else {}
                logger.info(f"Tool: {name}({args})")

                if name == "web_search":
                    result = _web_search(args.get("query", ""))

                elif name == "set_reminder":
                    mins = int(args.get("minutes", 5))
                    msg  = args.get("message", "Reminder!")
                    run_at = datetime.now(pytz.timezone(USER_TIMEZONE)) + timedelta(minutes=mins)
                    scheduler.add_job(
                        _fire_reminder, "date",
                        run_date=run_at,
                        args=[chat_id, msg],
                        id=f"rem_{user_id}_{run_at.timestamp()}"
                    )
                    result = f"Reminder set for {mins} min from now: '{msg}'"

                elif name == "save_note":
                    result = _save_note(args.get("title", "Note"), args.get("content", ""))

                elif name == "get_calendar":
                    result = _get_calendar(int(args.get("days", 7)))

                elif name == "get_datetime":
                    result = _get_datetime()

                else:
                    result = f"Unknown tool: {name}"

                tool_results.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=name,
                            response={"result": result}
                        )
                    )
                )

            response = chat_session.send_message(
                genai.protos.Content(parts=tool_results, role="user")
            )

        # Extract reply text
        reply = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    reply += part.text

        reply = reply.strip() or "Done!"

        if len(reply) > 4096:
            for i in range(0, len(reply), 4096):
                await update.message.reply_text(reply[i:i+4096])
        else:
            await update.message.reply_text(reply)

    except Exception as e:
        logger.error(f"handle_message error: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Something went wrong. Try again!")


# ─── Startup ──────────────────────────────────────────────────────────────────

async def post_init(app: Application):
    global _bot_app
    _bot_app = app
    scheduler.start()
    logger.info("Scheduler started.")


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

    logger.info("Bot running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
