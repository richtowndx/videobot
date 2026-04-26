import os
import re
import logging
import tempfile
from typing import Optional

import yt_dlp

from config import DownloaderConfig
from downloaders.base import BaseDownloader, DownloadResult

logger = logging.getLogger(__name__)

LANG_PRIORITY = ["zh-CN", "zh-Hans", "zh", "en", "ja"]


class BilibiliDownloader(BaseDownloader):
    def __init__(self):
        self._cookie_path = self._resolve_cookie()

    @staticmethod
    def _resolve_cookie() -> Optional[str]:
        """Cookie 优先级: 环境变量 > cookie_bilibili.txt 文件"""
        env_cookie = DownloaderConfig.BILIBILI_COOKIE
        if env_cookie:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="bili_cookie_", delete=False
            )
            tmp.write(env_cookie)
            tmp.close()
            logger.info("Using Bilibili cookie from env config")
            return tmp.name
        from config import BASE_DIR
        path = BASE_DIR / "cookie_bilibili.txt"
        if path.exists():
            return str(path)
        return None

    def download_audio(self, url: str, output_dir: str) -> DownloadResult:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": output_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",
                }
            ],
            "noplaylist": True,
            "quiet": True,
        }
        if self._cookie_path:
            ydl_opts["cookiefile"] = self._cookie_path

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "")
            title = info.get("title", "Unknown")
            duration = info.get("duration", 0)
            audio_path = os.path.join(output_dir, "audio.mp3")

        return DownloadResult(
            file_path=audio_path,
            title=title,
            video_id=video_id,
            duration=duration,
            platform="bilibili",
            raw_info=info,
        )

    def extract_subtitles(self, url: str) -> Optional[str]:
        try:
            info = self.extract_info(url)
            subtitles = info.get("subtitles", {})
            auto_captions = info.get("automatic_captions", {})

            for lang in LANG_PRIORITY:
                for source in (subtitles, auto_captions):
                    if lang in source and source[lang]:
                        sub_url = source[lang][0].get("url")
                        if sub_url:
                            import requests
                            resp = requests.get(sub_url, timeout=10)
                            resp.raise_for_status()
                            return self._parse_subtitle(resp.text)
            return None
        except Exception as e:
            logger.warning(f"Bilibili subtitle extraction failed: {e}")
            return None

    def extract_info(self, url: str) -> dict:
        ydl_opts = {"skip_download": True, "quiet": True, "noplaylist": True}
        if self._cookie_path:
            ydl_opts["cookiefile"] = self._cookie_path
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)

    @staticmethod
    def _parse_subtitle(content: str) -> str:
        # Strip HTML tags first
        cleaned = re.sub(r"<[^>]+>", "", content)
        lines = cleaned.strip().split("\n")
        text_lines = [
            line.strip()
            for line in lines
            if line.strip()
            and not line.startswith(("WEBVTT", "NOTE", "STYLE", "<?xml"))
            and "-->" not in line
            and not line.isdigit()
        ]
        return " ".join(text_lines).strip()
