import logging
import re
import ssl

import aiohttp

from .base import BaseUploader, UploadResult

logger = logging.getLogger(__name__)


def _clean_filename(title: str) -> str:
    """Keep only CJK, letters, digits; append -summary.md."""
    name = re.sub(r"[^\w一-鿿]+", "", title[:80])
    return f"{name}-summary.md" if name else "summary.md"


class HttpPutUploader(BaseUploader):
    name = "http_put"

    def __init__(self, url: str, token: str, enabled: bool = True, **kwargs):
        super().__init__(enabled=enabled, **kwargs)
        self.url = url.rstrip("/")
        self.token = token

    async def upload(self, file_path: str, title: str) -> UploadResult:
        if not self.enabled:
            return UploadResult(uploader=self.name, success=False, message="not enabled")

        filename = _clean_filename(title)
        upload_url = f"{self.url}/upload/{filename}"

        try:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

            with open(file_path, "rb") as f:
                content = f.read()

            async with aiohttp.ClientSession() as session:
                async with session.put(
                    upload_url,
                    data=content,
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "text/markdown",
                    },
                    ssl=ssl_ctx,
                ) as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        logger.info(f"http_put uploaded {filename}: {body.strip()}")
                        return UploadResult(uploader=self.name, success=True, message=body.strip())
                    else:
                        logger.error(f"http_put failed ({resp.status}): {body.strip()}")
                        return UploadResult(
                            uploader=self.name, success=False,
                            message=f"HTTP {resp.status}: {body.strip()}",
                        )
        except Exception as e:
            logger.error(f"http_put error: {e}", exc_info=True)
            return UploadResult(uploader=self.name, success=False, message=str(e))
