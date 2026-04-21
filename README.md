# Telegram AI Assistant

Personal Telegram AI assistant — Gemini 1.5 Flash (free tier) + DuckDuckGo web search + APScheduler reminders + optional Google Sheets notes & Calendar.

## Features
| Feature | How to trigger | Requires |
|---------|---------------|---------|
| 💬 AI chat | Just talk | Gemini API key |
| 🔍 Web search | "search for...", "what's the weather in..." | Nothing extra |
| ⏰ Reminders | "remind me in 20 mins to..." | Nothing extra |
| 📝 Save notes | "save note: ..." | Google service account |
| 📅 Calendar | "what's on my calendar?" | Google service account |

## Deploy on Railway (free)

1. Fork or clone this repo to your GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub → select this repo
3. Set environment variables (Settings → Variables):

```
TELEGRAM_TOKEN=your_token_here
GEMINI_API_KEY=your_key_here
USER_TIMEZONE=America/New_York
```

4. Railway auto-detects Python and starts `python bot.py`. Done.

## Optional — Google Sheets + Calendar

Create a Google Cloud service account, enable Sheets & Calendar APIs, download the JSON key, then set:
```
GOOGLE_SERVICE_ACCOUNT_JSON={...single-line JSON...}
GOOGLE_SHEET_ID=your_sheet_id
GOOGLE_CALENDAR_ID=primary
```

Share your Sheet and Calendar with the service account's `client_email`.

## Commands
- `/start` or `/help` — show feature list
- `/clear` — reset conversation memory
