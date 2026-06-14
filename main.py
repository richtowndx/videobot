import asyncio
import logging
import ssl
import sys
import abc
from collections.abc import AsyncGenerator
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, cast

import certifi
import aiohttp
from aiohttp import ClientError, ClientSession, FormData, TCPConnector
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods.base import TelegramMethod, TelegramType
from aiogram import Dispatcher

from config import BotConfig
from bot.handler import router
from utils.cleanup import cleanup_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AiohttpSessionHTTPProxy(AiohttpSession):
    """
    Custom session that uses aiohttp's native HTTP proxy support instead of aiohttp-socks.
    aiohttp-socks only supports SOCKS protocols, not HTTP proxies.
    """

    def __init__(
        self,
        proxy: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._proxy = proxy
        self._session: ClientSession | None = None
        self._connector_init: dict[str, Any] = {
            "ssl": ssl.create_default_context(cafile=certifi.where()),
            "limit": 100,
            "ttl_dns_cache": 3600,
        }

    @property
    def proxy(self) -> str | None:
        return self._proxy

    async def create_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                connector=TCPConnector(**self._connector_init),
                proxy=self._proxy,
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            await asyncio.sleep(0.25)

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        timeout: int | None = None,
    ) -> TelegramType:
        session = await self.create_session()

        url = self.api.api_url(token=bot.token, method=method.__api_method__)
        form = self.build_form_data(bot=bot, method=method)

        try:
            async with session.post(
                url,
                data=form,
                timeout=self.timeout if timeout is None else timeout,
            ) as resp:
                raw_result = await resp.text()
        except asyncio.TimeoutError as e:
            raise TelegramNetworkError(method=method, message="Request timeout error") from e
        except ClientError as e:
            raise TelegramNetworkError(method=method, message=f"{type(e).__name__}: {e}") from e

        response = self.check_response(
            bot=bot,
            method=method,
            status_code=resp.status,
            content=raw_result,
        )
        return cast(TelegramType, response.result)

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        session = await self.create_session()
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if raise_for_status:
                resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(chunk_size):
                yield chunk


def _build_session():
    if not BotConfig.PROXY:
        return AiohttpSession()

    parsed = urlparse(BotConfig.PROXY)
    scheme = parsed.scheme.lower()

    if scheme in ("socks4", "socks5", "socks"):
        logger.info(f"Bot using SOCKS proxy: {BotConfig.PROXY}")
        return AiohttpSession(proxy=BotConfig.PROXY)

    if scheme in ("http", "https"):
        logger.info(f"Bot using HTTP proxy: {BotConfig.PROXY}")
        return AiohttpSessionHTTPProxy(proxy=BotConfig.PROXY)

    logger.warning(f"Unknown proxy scheme '{scheme}', proxy will be ignored")
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
    await dp.start_polling(bot, polling_timeout=60)


if __name__ == "__main__":
    asyncio.run(main())
