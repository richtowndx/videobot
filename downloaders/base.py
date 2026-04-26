from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class DownloadResult:
    file_path: str
    title: str
    video_id: str
    duration: float = 0
    platform: str = ""
    raw_info: Optional[dict] = None


class BaseDownloader(ABC):
    @abstractmethod
    def download_audio(self, url: str, output_dir: str) -> DownloadResult:
        pass

    @abstractmethod
    def extract_subtitles(self, url: str) -> Optional[str]:
        pass

    @abstractmethod
    def extract_info(self, url: str) -> dict:
        pass
