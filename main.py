import asyncio
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram import Dispatcher

from config import BotConfig
from bot.handler import router
from utils.cleanup import cleanup_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_session():
    if BotConfig.PROXY:
        logger.info("Bot using proxy")
        return AiohttpSession(proxy=BotConfig.PROXY)
    return AiohttpSession()


async def main():
    bot = Bot(
        token=BotConfig.TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
        session=_build_session(),
    )
    dp = Dispatcher()
    dp.include_router(router)

    # Start cleanup scheduler in background
    asyncio.create_task(cleanup_scheduler())

    logger.info("VideoBot starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
