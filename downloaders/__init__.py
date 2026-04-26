from downloaders.base import BaseDownloader, DownloadResult
from downloaders.bilibili import BilibiliDownloader
from downloaders.youtube import YoutubeDownloader


def get_downloader(platform: str) -> BaseDownloader:
    if platform == "bilibili":
        return BilibiliDownloader()
    elif platform == "youtube":
        return YoutubeDownloader()
    raise ValueError(f"Unsupported platform: {platform}")
