#!/usr/bin/env python3
"""
Telegram AI Assistant — v5 (full internet access)
AI: Groq llama-3.3-70b with 10 tools
- Web search (DuckDuckGo)
- Fetch & read any URL / article
- Wikipedia lookup
- Weather (wttr.in, no key needed)
- Math & code evaluation
- Currency / crypto prices (CoinGecko, free)
- Set reminders
- Save notes to Google Sheets
- Get calendar events
- Get current datetime
- Photo/image analysis (Groq Vision: llama-4-scout)
"""
import os
import json
import re
import logging
import math
import httpx
from datetime import datetime, timedelta
from typing import Optional
from io import BytesIO

import pytz
from bs4 import BeautifulSoup

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from groq import Groq
from apscheduler.schedulers.asyncio import AsyncIOScheduler

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

TEXT_MODEL         = "llama-3.3-70b-versatile"
VISION_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq vision model

ALLOWED_IDS = set(int(x.strip()) for x in ALLOWED_USER_IDS.split(",") if x.strip())
client      = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a powerful personal AI assistant running on Telegram. You have full access to the internet and can handle a wide range of tasks — research, real-time data, reminders, calculations, notes, and more.

Your tools:
- web_search: search the web for any topic, news, or facts
- fetch_url: read and extract content from any URL or web page
- wikipedia: look up any topic on Wikipedia
- weather: get current weather for any city
- crypto_price: get current price of any cryptocurrency
- calculate: evaluate any math expression or formula
- set_reminder: set a timed reminder (in minutes)
- save_note: save a note to Google Sheets
- get_calendar: check upcoming calendar events
- get_datetime: get the current date and time

Behavior rules:
- Be concise and direct. Use plain text (no markdown, no asterisks).
- Proactively use tools when the user asks for real-world info — don't say you can't check things.
- For research tasks, search + fetch the top result to give richer answers.
- Reminders: confirm the exact time when set.
- If the user sends a photo, describe and analyze it thoroughly.
- You have memory of the current conversation. Ask for clarification only if truly needed.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for any topic — news, facts, products, people, events, prices, tutorials, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and read the content of any web page or article by URL. Use to get detailed info from a specific site.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to fetch (https://...)"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia",
            "description": "Look up any topic, person, place, or concept on Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term or topic"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get current weather conditions for any city or location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name or location (e.g. 'New York', 'Miami FL')"}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crypto_price",
            "description": "Get the current price and 24h change of any cryptocurrency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "Coin name or symbol (e.g. 'bitcoin', 'ethereum', 'solana', 'BTC')"}
                },
                "required": ["coin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate any math expression, formula, or calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate (e.g. '(1500 * 0.07) + 200', 'sqrt(144)', 'log(1000)')"}
                },
                "required": ["expression"],
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
            "description": "Save a note, task, or piece of information to Google Sheets.",
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
                "properties": {
                    "days": {"type": "integer", "description": "Number of days to look ahead (default 7)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time in the user's timezone.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ── Scheduler & state ─────────────────────────────────────────────────────────
scheduler    = AsyncIOScheduler(timezone=USER_TIMEZONE)
user_histories: dict[int, list] = {}
_bot_app: Optional[Application] = None


# ── Tool implementations ──────────────────────────────────────────────────────

def _web_search(query: str) -> str:
    if not DDG_AVAILABLE:
        return "Web search unavailable."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r.get('title','')}\nURL: {r.get('href','')}\nSnippet: {r.get('body','')}"
            for r in results[:6]
        )
    except Exception as e:
        return f"Search error: {e}"


def _fetch_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AIBot/1.0)"}
        resp = httpx.get(url, headers=headers, timeout=12, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text  = "\n".join(lines)
        return text[:6000] + ("..." if len(text) > 6000 else "")
    except Exception as e:
        return f"Could not fetch URL: {e}"


def _wikipedia(query: str) -> str:
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={httpx.utils.quote(query)}&format=json&srlimit=1"
        r = httpx.get(search_url, timeout=8)
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return "No Wikipedia article found."
        title = results[0]["title"]
        summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=true&explaintext=true&titles={httpx.utils.quote(title)}&format=json"
        r2 = httpx.get(summary_url, timeout=8)
        pages = r2.json().get("query", {}).get("pages", {})
        page  = next(iter(pages.values()))
        extract = page.get("extract", "No content found.")
        return f"Wikipedia — {title}:\n\n{extract[:3000]}"
    except Exception as e:
        return f"Wikipedia error: {e}"


def _weather(location: str) -> str:
    try:
        url  = f"https://wttr.in/{httpx.utils.quote(location)}?format=4"
        resp = httpx.get(url, timeout=8)
        return resp.text.strip() or "Weather data unavailable."
    except Exception as e:
        return f"Weather error: {e}"


def _crypto_price(coin: str) -> str:
    try:
        symbol_map = {
            "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
            "bnb": "binancecoin", "xrp": "ripple", "ada": "cardano",
            "doge": "dogecoin", "shib": "shiba-inu", "avax": "avalanche-2",
            "dot": "polkadot", "link": "chainlink", "matic": "matic-network",
            "ltc": "litecoin", "uni": "uniswap", "atom": "cosmos",
        }
        coin_id = symbol_map.get(coin.lower(), coin.lower().replace(" ", "-"))
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
        r   = httpx.get(url, timeout=8)
        data = r.json().get(coin_id)
        if not data:
            return f"Could not find price for '{coin}'. Try the full name (e.g. 'bitcoin', 'ethereum')."
        price  = data.get("usd", "?")
        change = data.get("usd_24h_change", None)
        ch_str = f" (24h: {change:+.2f}%)" if change is not None else ""
        return f"{coin.upper()}: ${price:,.4f}{ch_str}"
    except Exception as e:
        return f"Crypto price error: {e}"


def _calculate(expression: str) -> str:
    try:
        safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_dict.update({"abs": abs, "round": round, "min": min, "max": max})
        expr = re.sub(r"[^0-9+\-*/().,%^ a-zA-Z]", "", expression)
        expr = expr.replace("^", "**").replace(",", "")
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def _save_note(title: str, content: str) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON or not GOOGLE_SHEET_ID:
        return "Google Sheets not configured."
    try:
        sa    = json.loads(GOOGLE_SA_JSON)
        creds = Credentials.from_service_account_info(sa, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc    = gspread.authorize(creds)
        sh    = gc.open_by_key(GOOGLE_SHEET_ID)
        try:
            ws = sh.worksheet("Notes")
        except Exception:
            ws = sh.add_worksheet(title="Notes", rows=1000, cols=4)
            ws.append_row(["Date", "Title", "Content"])
        ts = datetime.now(pytz.timezone(USER_TIMEZONE)).strftime("%Y-%m-%d %H:%M")
        ws.append_row([ts, title, content])
        return f'Note saved: "{title}"'
    except Exception as e:
        return f"Sheets error: {e}"


def _get_calendar(days: int = 7) -> str:
    if not GOOGLE_AVAILABLE or not GOOGLE_SA_JSON:
        return "Google Calendar not configured."
    try:
        sa    = json.loads(GOOGLE_SA_JSON)
        creds = Credentials.from_service_account_info(sa, scopes=["https://www.googleapis.com/auth/calendar.readonly"])
        svc   = google_build("calendar", "v3", credentials=creds)
        tz    = pytz.timezone(USER_TIMEZONE)
        now   = datetime.now(tz)
        end   = now + timedelta(days=days)
        result = svc.events().list(
            calendarId=GOOGLE_CALENDAR_ID,
            timeMin=now.isoformat(), timeMax=end.isoformat(),
            maxResults=20, singleEvents=True, orderBy="startTime",
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
            lines.append(f"- {e.get('summary','Untitled')} ({ts})")
        return "\n".join(lines)
    except Exception as e:
        return f"Calendar error: {e}"


def _get_datetime() -> str:
    return datetime.now(pytz.timezone(USER_TIMEZONE)).strftime("%A, %B %d, %Y at %I:%M %p %Z")


def _dispatch_tool(name: str, args: dict, chat_id: int, user_id: int) -> str:
    if name == "web_search":
        return _web_search(args.get("query", ""))
    elif name == "fetch_url":
        return _fetch_url(args.get("url", ""))
    elif name == "wikipedia":
        return _wikipedia(args.get("query", ""))
    elif name == "weather":
        return _weather(args.get("location", ""))
    elif name == "crypto_price":
        return _crypto_price(args.get("coin", ""))
    elif name == "calculate":
        return _calculate(args.get("expression", ""))
    elif name == "set_reminder":
        mins   = int(args.get("minutes", 5))
        msg    = args.get("message", "Reminder!")
        tz     = pytz.timezone(USER_TIMEZONE)
        run_at = datetime.now(tz) + timedelta(minutes=mins)
        scheduler.add_job(
            _fire_reminder, "date",
            run_date=run_at, args=[chat_id, msg],
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
            await _bot_app.bot.send_message(chat_id=chat_id, text=f"Reminder: {message}")
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
        "Hey! I'm your AI assistant with full internet access.\n\n"
        "I can:\n"
        "Search the web for anything\n"
        "Read and summarize any article or URL\n"
        "Look up Wikipedia\n"
        "Check live weather for any city\n"
        "Get crypto prices\n"
        "Do math and calculations\n"
        "Set reminders\n"
        "Save notes to Google Sheets\n"
        "Check your calendar\n"
        "Analyze photos you send me\n\n"
        "Just talk to me naturally.\n"
        "/clear — start fresh conversation\n"
        "/help — show this"
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_histories.pop(update.effective_user.id, None)
    await update.message.reply_text("Conversation cleared.")


# ── Photo handler (vision) ────────────────────────────────────────────────────────

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    if not is_allowed(user_id):
        return

    caption = (update.message.caption or "What is in this image? Describe it in detail.").strip()
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        photo   = update.message.photo[-1]
        file    = await context.bot.get_file(photo.file_id)
        img_url = file.file_path

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
            max_tokens=1024,
        )
        reply = response.choices[0].message.content.strip()

        history = user_histories.setdefault(user_id, [])
        history.append({"role": "user", "content": f"[Photo sent] {caption}"})
        history.append({"role": "assistant", "content": reply})

        await update.message.reply_text(reply or "I see the photo but couldn't generate a description.")

    except Exception as e:
        logger.error(f"Vision error: {e}", exc_info=True)
        await update.message.reply_text("Could not analyze that image. Try again!")


# ── Main message handler ──────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id   = update.effective_user.id
    chat_id   = update.effective_chat.id
    user_text = (update.message.text or "").strip()

    if not is_allowed(user_id) or not user_text:
        if not is_allowed(user_id):
            await update.message.reply_text("Not authorized.")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    history  = user_histories.setdefault(user_id, [])
    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        for _ in range(8):
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                })
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                for tc in msg.tool_calls:
                    args   = json.loads(tc.function.arguments or "{}")
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
                for chunk in [reply[i:i+4096] for i in range(0, len(reply), 4096)]:
                    await update.message.reply_text(chunk)
                return

        await update.message.reply_text("Done!")

    except Exception as e:
        logger.error(f"handle_message error: {e}", exc_info=True)
        await update.message.reply_text("Something went wrong. Try again!")


# ── Startup ───────────────────────────────────────────────────────────────────

async def post_init(app: Application):
    global _bot_app
    _bot_app = app
    scheduler.start()
    logger.info(f"Bot v5 started — webhook mode, port {PORT}")


def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    if WEBHOOK_URL:
        logger.info(f"Webhook: {WEBHOOK_URL}")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            webhook_url=WEBHOOK_URL,
            drop_pending_updates=True,
        )
    else:
        logger.info("Polling fallback (no WEBHOOK_URL)")
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
