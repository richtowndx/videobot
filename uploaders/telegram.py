import logging

from .base import BaseUploader, UploadResult

logger = logging.getLogger(__name__)


class TelegramUploader(BaseUploader):
    name = "telegram"

    def __init__(self, enabled: bool = True, **kwargs):
        super().__init__(enabled=enabled, **kwargs)
        self._message_func = None

    def set_sender(self, send_func):
        """send_func signature: async (title, content) -> None
        Called to deliver the document via Telegram bot."""
        self._message_func = send_func

    async def upload(self, file_path: str, title: str) -> UploadResult:
        if not self.enabled:
            return UploadResult(uploader=self.name, success=False, message="not enabled")

        if not self._message_func:
            return UploadResult(uploader=self.name, success=False, message="no sender configured")

        try:
            await self._message_func(title, file_path)
            return UploadResult(uploader=self.name, success=True, message="sent via Telegram")
        except Exception as e:
            logger.error(f"Telegram upload error: {e}", exc_info=True)
            return UploadResult(uploader=self.name, success=False, message=str(e))
