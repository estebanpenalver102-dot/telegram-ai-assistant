#!/usr/bin/env python3
"""
Telegram AI Assistant — v6
- Never stops: rate-limit recovery, model fallback, smart context trimming
- Multi-project mode: parallel named projects per user
- Asks clarifying questions before starting complex tasks
- Full internet access (10 tools + vision)
"""
import os, json, re, logging, math, httpx, time
from datetime import datetime, timedelta
from typing import Optional
from io import BytesIO

import pytz
from bs4 import BeautifulSoup
from groq import Groq, RateLimitError, APIStatusError
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes,
)

try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

try:
    import gspread
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build as google_build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO)
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

# Model ladder — try in order until one works
MODEL_LADDER = [
    "llama-3.3-70b-versatile",    # best quality, 6k tok/min free
    "llama-3.1-70b-versatile",    # fallback 1
    "llama-3.1-8b-instant",       # fallback 2 — very high limits
    "gemma2-9b-it",               # fallback 3
]
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

ALLOWED_IDS = set(int(x.strip()) for x in ALLOWED_USER_IDS.split(",") if x.strip())
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a powerful personal AI project assistant on Telegram. You have full internet access and can work on complex, multi-step projects over many messages.

CAPABILITIES:
- Search the web, fetch URLs, look up Wikipedia
- Get weather, crypto prices, do math
- Set reminders, save notes, check calendar
- Analyze photos/images
- Build projects: YouTube automation, backend systems, scripts, tools — step by step

BEHAVIOR RULES:
1. Before starting a complex task (building something, automating a workflow, writing a system), ask the essential clarifying questions first. Be specific — ask only what you truly need. Example: "Before I start the YouTube channel automation, I need to know: (a) What niche/topic? (b) AI-generated or curated content? (c) Do you have a YouTube account ready?"
2. For ongoing projects, always recap the current state at the start of each response so context is clear.
3. Break big tasks into numbered steps. Complete one at a time, confirm before moving on if the step requires user input.
4. Use tools proactively — if you need real data, look it up. Never say "I can't access that."
5. Be direct and concise. Plain text, no asterisks or markdown.
6. If you're mid-task and realize you need something from the user, just ask.
"""

TOOLS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web for current info, news, tutorials, tools, libraries, documentation, pricing, or anything real-time.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fetch_url", "description": "Fetch and read any web page, article, GitHub repo README, documentation page, or URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "wikipedia", "description": "Look up any topic on Wikipedia.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "weather", "description": "Get current weather for any city.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "crypto_price", "description": "Get current price and 24h change for any cryptocurrency.", "parameters": {"type": "object", "properties": {"coin": {"type": "string"}}, "required": ["coin"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate math expressions, formulas, or calculations.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "set_reminder", "description": "Set a reminder after N minutes.", "parameters": {"type": "object", "properties": {"message": {"type": "string"}, "minutes": {"type": "integer"}}, "required": ["message", "minutes"]}}},
    {"type": "function", "function": {"name": "save_note", "description": "Save a note or project update to Google Sheets.", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]}}},
    {"type": "function", "function": {"name": "get_calendar", "description": "Fetch upcoming calendar events.", "parameters": {"type": "object", "properties": {"days": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_datetime", "description": "Get current date and time.", "parameters": {"type": "object", "properties": {}, "required": []}}},
]

# ── State ──────────────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler(timezone=USER_TIMEZONE)
_bot_app: Optional[Application] = None

# user_id -> { "active": project_name, "projects": { name: [messages] } }
user_state: dict[int, dict] = {}
DEFAULT_PROJECT = "general"


def get_history(user_id: int) -> list:
    state = user_state.setdefault(user_id, {"active": DEFAULT_PROJECT, "projects": {DEFAULT_PROJECT: []}})
    proj  = state["active"]
    return state["projects"].setdefault(proj, [])


def get_active_project(user_id: int) -> str:
    return user_state.get(user_id, {}).get("active", DEFAULT_PROJECT)


def set_active_project(user_id: int, name: str):
    state = user_state.setdefault(user_id, {"active": DEFAULT_PROJECT, "projects": {}})
    state["active"] = name
    state["projects"].setdefault(name, [])


def list_projects(user_id: int) -> list[str]:
    return list(user_state.get(user_id, {}).get("projects", {DEFAULT_PROJECT: []}).keys())


# ── Context management ───────────────────────────────────────────────────────────

MAX_HISTORY = 50
KEEP_RECENT = 20


async def maybe_summarize(user_id: int, model: str):
    """If history is too long, summarize the oldest chunk to stay within token limits."""
    history = get_history(user_id)
    if len(history) <= MAX_HISTORY:
        return
    to_summarize = history[:-KEEP_RECENT]
    recent       = history[-KEEP_RECENT:]
    try:
        summary_resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize the following conversation into a compact project context note. Include: decisions made, work done, open questions, and user preferences. Be dense and factual."},
                {"role": "user", "content": "\n".join(f"{m['role'].upper()}: {m['content']}" for m in to_summarize)},
            ],
            max_tokens=600,
        )
        summary_text = summary_resp.choices[0].message.content.strip()
        summarized_msg = {"role": "system", "content": f"[Project context summary]\n{summary_text}"}
        state = user_state[user_id]
        proj  = state["active"]
        state["projects"][proj] = [summarized_msg] + recent
        logger.info(f"Summarized history for user {user_id}, project '{proj}'")
    except Exception as e:
        logger.warning(f"Summarization failed: {e} — trimming instead")
        state = user_state[user_id]
        state["projects"][state["active"]] = history[-KEEP_RECENT:]


# ── Rate-limit-safe LLM call ───────────────────────────────────────────────────────

def call_with_fallback(messages: list, max_tokens: int = 1500) -> tuple:
    """Try each model in the ladder. Returns (choice_message, model_used)."""
    last_err = None
    for model in MODEL_LADDER:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=max_tokens,
            )
            return resp.choices[0].message, model
        except RateLimitError as e:
            logger.warning(f"Rate limit on {model}: {e}. Trying next.")
            last_err = e
            time.sleep(1)
        except APIStatusError as e:
            if e.status_code == 429:
                logger.warning(f"429 on {model}: {e}. Trying next.")
                last_err = e
                time.sleep(1)
            else:
                raise
    raise last_err or RuntimeError("All models failed")


# ── Tool implementations ──────────────────────────────────────────────────────

def _web_search(query: str) -> str:
    if not DDG_AVAILABLE:
        return "Web search unavailable."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
        if not results:
            return "No results found."
        return "\n\n".join(f"Title: {r.get('title','')}\nURL: {r.get('href','')}\nSnippet: {r.get('body','')}" for r in results[:6])
    except Exception as e:
        return f"Search error: {e}"


def _fetch_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = httpx.get(url, headers=headers, timeout=12, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text  = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text  = "\n".join(lines)
        return text[:6000] + ("..." if len(text) > 6000 else "")
    except Exception as e:
        return f"Could not fetch URL: {e}"


def _wikipedia(query: str) -> str:
    try:
        r = httpx.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={httpx.utils.quote(query)}&format=json&srlimit=1", timeout=8)
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return "No Wikipedia article found."
        title = results[0]["title"]
        r2 = httpx.get(f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=true&explaintext=true&titles={httpx.utils.quote(title)}&format=json", timeout=8)
        pages = r2.json().get("query", {}).get("pages", {})
        extract = next(iter(pages.values())).get("extract", "No content.")
        return f"Wikipedia — {title}:\n\n{extract[:3000]}"
    except Exception as e:
        return f"Wikipedia error: {e}"


def _weather(location: str) -> str:
    try:
        resp = httpx.get(f"https://wttr.in/{httpx.utils.quote(location)}?format=4", timeout=8)
        return resp.text.strip() or "Weather unavailable."
    except Exception as e:
        return f"Weather error: {e}"


def _crypto_price(coin: str) -> str:
    try:
        symbol_map = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana", "bnb": "binancecoin", "xrp": "ripple", "ada": "cardano", "doge": "dogecoin", "avax": "avalanche-2", "matic": "matic-network", "ltc": "litecoin"}
        coin_id = symbol_map.get(coin.lower(), coin.lower().replace(" ", "-"))
        r = httpx.get(f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true", timeout=8)
        data = r.json().get(coin_id)
        if not data:
            return f"Price not found for '{coin}'."
        price  = data.get("usd", "?")
        change = data.get("usd_24h_change")
        return f"{coin.upper()}: ${price:,.4f}" + (f" (24h: {change:+.2f}%)" if change else "")
    except Exception as e:
        return f"Crypto error: {e}"


def _calculate(expression: str) -> str:
    try:
        safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe.update({"abs": abs, "round": round, "min": min, "max": max})
        expr = re.sub(r"[^0-9+\-*/().,%^ a-zA-Z]", "", expression).replace("^", "**").replace(",", "")
        result = eval(expr, {"__builtins__": {}}, safe)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def _save_note(title: str, content: str) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON or not GOOGLE_SHEET_ID:
        return "Google Sheets not configured."
    try:
        creds = Credentials.from_service_account_info(json.loads(GOOGLE_SA_JSON), scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GOOGLE_SHEET_ID)
        try:
            ws = sh.worksheet("Notes")
        except Exception:
            ws = sh.add_worksheet(title="Notes", rows=1000, cols=4)
            ws.append_row(["Date", "Title", "Content"])
        ws.append_row([datetime.now(pytz.timezone(USER_TIMEZONE)).strftime("%Y-%m-%d %H:%M"), title, content])
        return f'Note saved: "{title}"'
    except Exception as e:
        return f"Sheets error: {e}"


def _get_calendar(days: int = 7) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON:
        return "Google Calendar not configured."
    try:
        creds = Credentials.from_service_account_info(json.loads(GOOGLE_SA_JSON), scopes=["https://www.googleapis.com/auth/calendar.readonly"])
        svc   = google_build("calendar", "v3", credentials=creds)
        tz    = pytz.timezone(USER_TIMEZONE)
        now   = datetime.now(tz)
        result = svc.events().list(calendarId=GOOGLE_CALENDAR_ID, timeMin=now.isoformat(), timeMax=(now + timedelta(days=days)).isoformat(), maxResults=20, singleEvents=True, orderBy="startTime").execute()
        events = result.get("items", [])
        if not events:
            return f"No events in the next {days} days."
        lines = []
        for e in events:
            start = e["start"].get("dateTime", e["start"].get("date", ""))
            ts = datetime.fromisoformat(start).astimezone(tz).strftime("%a %b %d, %I:%M %p") if "T" in start else start
            lines.append(f"- {e.get('summary','Untitled')} ({ts})")
        return "\n".join(lines)
    except Exception as e:
        return f"Calendar error: {e}"


def _dispatch_tool(name: str, args: dict, chat_id: int, user_id: int, model: str) -> str:
    if name == "web_search":     return _web_search(args.get("query", ""))
    if name == "fetch_url":      return _fetch_url(args.get("url", ""))
    if name == "wikipedia":      return _wikipedia(args.get("query", ""))
    if name == "weather":        return _weather(args.get("location", ""))
    if name == "crypto_price":   return _crypto_price(args.get("coin", ""))
    if name == "calculate":      return _calculate(args.get("expression", ""))
    if name == "get_calendar":   return _get_calendar(int(args.get("days", 7)))
    if name == "get_datetime":   return datetime.now(pytz.timezone(USER_TIMEZONE)).strftime("%A, %B %d, %Y at %I:%M %p %Z")
    if name == "save_note":      return _save_note(args.get("title", "Note"), args.get("content", ""))
    if name == "set_reminder":
        mins = int(args.get("minutes", 5))
        msg  = args.get("message", "Reminder!")
        run_at = datetime.now(pytz.timezone(USER_TIMEZONE)) + timedelta(minutes=mins)
        scheduler.add_job(_fire_reminder, "date", run_date=run_at, args=[chat_id, msg], id=f"rem_{user_id}_{run_at.timestamp()}")
        return f"Reminder set for {mins} min: '{msg}'"
    return f"Unknown tool: {name}"


async def _fire_reminder(chat_id: int, message: str):
    if _bot_app:
        try:
            await _bot_app.bot.send_message(chat_id=chat_id, text=f"Reminder: {message}")
        except Exception as e:
            logger.error(f"Reminder failed: {e}")


def is_allowed(uid: int) -> bool:
    return not ALLOWED_IDS or uid in ALLOWED_IDS


# ── Commands ─────────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("Not authorized.")
        return
    await update.message.reply_text(
        "Hey! I'm your AI project assistant.\n\n"
        "I can run multiple projects in parallel — YouTube channels, backend systems, automations, scripts, research, and more. "
        "I'll ask you questions when I need info to get the job done.\n\n"
        "Commands:\n"
        "/project <name> — start or switch to a named project\n"
        "/projects — list all your active projects\n"
        "/status — what we're currently working on\n"
        "/clear — clear current project history\n"
        "/help — show this\n\n"
        "Just tell me what you want to build."
    )


async def cmd_project(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    name = " ".join(context.args).strip() if context.args else ""
    if not name:
        projects = list_projects(uid)
        kb = [[InlineKeyboardButton(p, callback_data=f"switch_project:{p}")] for p in projects]
        kb.append([InlineKeyboardButton("+ New project", callback_data="new_project")])
        await update.message.reply_text("Your projects:", reply_markup=InlineKeyboardMarkup(kb))
        return
    set_active_project(uid, name)
    history = get_history(uid)
    msg = f"Switched to project: {name}"
    if not history:
        msg += "\nThis is a new project. What are we building?"
    await update.message.reply_text(msg)


async def cmd_projects(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    projects = list_projects(uid)
    active   = get_active_project(uid)
    lines = [f"{'> ' if p == active else '  '}{p} ({'active' if p == active else f'{len(user_state[uid][\"projects\"][p])} messages'})" for p in projects]
    await update.message.reply_text("Projects:\n" + "\n".join(lines) + "\n\nUse /project <name> to switch.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    active  = get_active_project(uid)
    history = get_history(uid)
    msg_count = len([m for m in history if m["role"] != "system"])
    await update.message.reply_text(f"Active project: {active}\nMessages in context: {msg_count}\nAll projects: {', '.join(list_projects(uid))}")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    proj = get_active_project(uid)
    user_state.setdefault(uid, {"active": DEFAULT_PROJECT, "projects": {}})["projects"][proj] = []
    await update.message.reply_text(f"Cleared project: {proj}")


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid  = query.from_user.id
    data = query.data
    if data.startswith("switch_project:"):
        name = data.split(":", 1)[1]
        set_active_project(uid, name)
        await query.edit_message_text(f"Switched to: {name}")
    elif data == "new_project":
        await query.edit_message_text("Send: /project <name> to start a new project.")


# ── Photo handler ───────────────────────────────────────────────────────────────

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid     = update.effective_user.id
    chat_id = update.effective_chat.id
    if not is_allowed(uid):
        return
    caption = (update.message.caption or "Describe this image in detail.").strip()
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        photo   = update.message.photo[-1]
        file    = await context.bot.get_file(photo.file_id)
        img_url = file.file_path
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": caption}, {"type": "image_url", "image_url": {"url": img_url}}]}],
            max_tokens=1024,
        )
        reply = resp.choices[0].message.content.strip()
        history = get_history(uid)
        history.append({"role": "user", "content": f"[Photo] {caption}"})
        history.append({"role": "assistant", "content": reply})
        await update.message.reply_text(reply or "Could not analyze image.")
    except Exception as e:
        logger.error(f"Vision error: {e}", exc_info=True)
        await update.message.reply_text("Could not analyze that image.")


# ── Main message handler ──────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid       = update.effective_user.id
    chat_id   = update.effective_chat.id
    user_text = (update.message.text or "").strip()

    if not is_allowed(uid) or not user_text:
        if not is_allowed(uid):
            await update.message.reply_text("Not authorized.")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    model = MODEL_LADDER[0]
    await maybe_summarize(uid, model)

    history  = get_history(uid)
    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        for _ in range(10):
            msg, model = call_with_fallback(messages)

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls],
                })
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                for tc in msg.tool_calls:
                    args   = json.loads(tc.function.arguments or "{}")
                    logger.info(f"[{get_active_project(uid)}] Tool: {tc.function.name}({args})")
                    result = _dispatch_tool(tc.function.name, args, chat_id, uid, model)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                reply = (msg.content or "").strip() or "Done."
                history.append({"role": "assistant", "content": reply})
                for chunk in [reply[i:i+4096] for i in range(0, len(reply), 4096)]:
                    await update.message.reply_text(chunk)
                return

        await update.message.reply_text("Done.")

    except Exception as e:
        logger.error(f"handle_message error: {e}", exc_info=True)
        await update.message.reply_text("Hit an unexpected error. Try rephrasing or /clear and retry.")


# ── Startup ───────────────────────────────────────────────────────────────────

async def post_init(app: Application):
    global _bot_app
    _bot_app = app
    scheduler.start()
    logger.info(f"Bot v6 started — port {PORT}")


def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_start))
    app.add_handler(CommandHandler("project",  cmd_project))
    app.add_handler(CommandHandler("projects", cmd_projects))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("clear",    cmd_clear))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if WEBHOOK_URL:
        logger.info(f"Webhook: {WEBHOOK_URL}")
        app.run_webhook(listen="0.0.0.0", port=PORT, webhook_url=WEBHOOK_URL, drop_pending_updates=True)
    else:
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
