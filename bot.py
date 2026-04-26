"""Telegram bot entry point.

Features
--------
* Detects Bilibili and YouTube URLs in every text message.
* Processes multiple URLs in a single message **sequentially**.
* Sends an interim "processing…" message that is later edited with the
  final Markdown note (or an error message).
* Supports an optional HTTP/SOCKS proxy for Telegram API calls via
  ``PROXY_URL`` in the environment.
* Launches the background cleanup thread before polling starts.
"""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

import config
from cleanup import start_cleanup_thread
from pipeline import find_urls, process_url

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Handlers ──────────────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    urls = find_urls(update.message.text)
    if not urls:
        return

    for url in urls:
        # Send a placeholder that will be replaced once processing finishes.
        status = await update.message.reply_text(f"⏳ 正在处理：{url}")
        try:
            loop = asyncio.get_event_loop()
            # Run blocking pipeline in a thread-pool so the event loop stays free.
            note = await loop.run_in_executor(None, process_url, url)
            # Telegram Markdown has a 4096-char limit per message.
            if len(note) > 4096:
                await status.edit_text(note[:4093] + "…", parse_mode="Markdown")
            else:
                await status.edit_text(note, parse_mode="Markdown")
        except Exception as exc:
            logger.error("error processing %s: %s", url, exc, exc_info=True)
            await status.edit_text(f"❌ 处理失败：{url}\n\n{exc}")


# ── Bot bootstrap ─────────────────────────────────────────────────────────────

def build_app() -> Application:
    """Construct and configure the Application instance."""
    builder = Application.builder().token(config.TELEGRAM_TOKEN)
    if config.PROXY_URL:
        builder = builder.proxy_url(config.PROXY_URL)
    app = builder.build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app


def main() -> None:
    start_cleanup_thread()
    app = build_app()
    logger.info("bot polling started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
