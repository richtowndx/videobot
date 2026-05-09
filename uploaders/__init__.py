import logging

from config import _cfg
from .base import BaseUploader, UploadResult
from .http_put import HttpPutUploader
from .telegram import TelegramUploader

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseUploader]] = {
    "http_put": HttpPutUploader,
    "telegram": TelegramUploader,
}


def register_uploader(name: str, cls: type[BaseUploader]):
    _REGISTRY[name] = cls


class UploaderManager:
    def __init__(self):
        self._uploaders: list[BaseUploader] = []
        self._build()

    def _build(self):
        uploader_cfg = _cfg.get("uploader", {})
        types_list = uploader_cfg.get("type", [])

        for name in types_list:
            cls = _REGISTRY.get(name)
            if cls is None:
                logger.warning(f"Unknown uploader type: {name}, skipping")
                continue

            section = uploader_cfg.get(name, {})
            instance = cls(**section)
            self._uploaders.append(instance)
            logger.info(f"Registered uploader: {name} (enabled={instance.enabled})")

    def get_telegram_uploader(self) -> TelegramUploader | None:
        for u in self._uploaders:
            if isinstance(u, TelegramUploader):
                return u
        return None

    async def upload(self, file_path: str, title: str) -> list[UploadResult]:
        results: list[UploadResult] = []
        for uploader in self._uploaders:
            try:
                result = await uploader.upload(file_path, title)
            except Exception as e:
                result = UploadResult(
                    uploader=uploader.name, success=False, message=str(e)
                )
            results.append(result)
        return results
