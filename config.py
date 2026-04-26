"""Central configuration loaded from environment variables / .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Required ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN: str = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

# ── OpenAI / LLM ─────────────────────────────────────────────────────────────
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# Whisper model name for the /audio/transcriptions endpoint
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-1")

# ── Proxy ─────────────────────────────────────────────────────────────────────
# Applied to both Telegram HTTP calls and yt-dlp downloads.
# Example: "socks5://user:pass@127.0.0.1:1080"
PROXY_URL: str | None = os.getenv("PROXY_URL")

# ── Storage ───────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))

# How many days before a cached note is considered stale and deleted.
NOTE_EXPIRE_DAYS: int = int(os.getenv("NOTE_EXPIRE_DAYS", "30"))

# How many days before a per-video task directory is cleaned up.
TASK_EXPIRE_DAYS: int = int(os.getenv("TASK_EXPIRE_DAYS", "1"))

# ── Pipeline ──────────────────────────────────────────────────────────────────
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
