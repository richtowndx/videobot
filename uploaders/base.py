import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    uploader: str
    success: bool
    message: str


class BaseUploader:
    name: str = "base"

    def __init__(self, enabled: bool = True, **kwargs):
        self.enabled = enabled

    async def upload(self, file_path: str, title: str, suffix: str = "_summary") -> UploadResult:
        raise NotImplementedError
