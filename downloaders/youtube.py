import os
import re
import logging
import tempfile
from typing import Optional

import yt_dlp

from config import DownloaderConfig
from downloaders.base import BaseDownloader, DownloadResult

logger = logging.getLogger(__name__)

LANG_PRIORITY = ["en", "zh-Hans", "zh-CN", "zh", "ja", "ko", "de", "fr"]


class YoutubeDownloader(BaseDownloader):
    def __init__(self):
        self.proxy = DownloaderConfig.YT_PROXY
        self._cookie_path = self._resolve_cookie()

    @staticmethod
    def _resolve_cookie() -> Optional[str]:
        """Cookie 优先级: 环境变量 > cookie_ytb.txt 文件"""
        env_cookie = DownloaderConfig.YOUTUBE_COOKIE
        if env_cookie:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="ytb_cookie_", delete=False
            )
            tmp.write(env_cookie)
            tmp.close()
            logger.info("Using YouTube cookie from env config")
            return tmp.name
        from config import BASE_DIR
        path = BASE_DIR / "cookie_ytb.txt"
        if path.exists():
            return str(path)
        return None

    def _get_base_opts(self) -> dict:
        opts = {"noplaylist": True, "quiet": True}
        if self.proxy:
            opts["proxy"] = self.proxy
        return opts

    def download_audio(self, url: str, output_dir: str) -> DownloadResult:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "audio.%(ext)s")

        strategies = [
            {
                "name": "Web + Cookies",
                "format": "140/251/139/bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
                "use_cookies": True,
            },
            {
                "name": "Web (no cookies)",
                "format": "140/251/139/bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
            },
            {
                "name": "Combined+extract (cookies)",
                "format": "18/91/92/93/94",
                "use_cookies": True,
            },
            {
                "name": "Combined+extract (no cookies)",
                "format": "18/91/92/93/94",
            },
        ]

        info = None
        last_error = None

        for strategy in strategies:
            try:
                logger.info(f"Trying strategy: {strategy['name']}")
                opts = self._get_base_opts()
                opts["format"] = strategy["format"]
                opts["outtmpl"] = output_path
                opts["postprocessors"] = [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",
                }]
                opts["ignoreerrors"] = True

                if strategy.get("use_cookies") and self._cookie_path:
                    opts["cookiefile"] = self._cookie_path

                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if info:
                        logger.info(f"Strategy {strategy['name']} succeeded")
                        break
            except Exception as e:
                last_error = e
                logger.warning(f"Strategy {strategy['name']} failed: {e}")

        if info is None:
            raise RuntimeError(f"All download strategies failed. Last error: {last_error}")

        video_id = info.get("id", "")
        title = info.get("title", "Unknown")
        duration = info.get("duration", 0)

        audio_path = os.path.join(output_dir, "audio.mp3")
        if not os.path.exists(audio_path):
            for ext in ("m4a", "webm", "wav", "mp3"):
                candidate = os.path.join(output_dir, f"audio.{ext}")
                if os.path.exists(candidate):
                    audio_path = candidate
                    break

        return DownloadResult(
            file_path=audio_path,
            title=title,
            video_id=video_id,
            duration=duration,
            platform="youtube",
            raw_info={"tags": info.get("tags", [])},
        )

    def extract_subtitles(self, url: str) -> Optional[str]:
        # Try with cookies first, then without
        for use_cookies in [True, False]:
            try:
                info = self._extract_info_internal(url, use_cookies=use_cookies)
                subtitles = info.get("subtitles", {})
                auto_captions = info.get("automatic_captions", {})

                proxies = None
                if self.proxy:
                    proxies = {"http": self.proxy, "https": self.proxy}

                for lang in LANG_PRIORITY:
                    for source in (subtitles, auto_captions):
                        if lang in source and source[lang]:
                            sub_url = source[lang][0].get("url")
                            if sub_url:
                                import requests
                                resp = requests.get(sub_url, timeout=10, proxies=proxies)
                                resp.raise_for_status()
                                return self._parse_subtitle(resp.text)
                return None
            except Exception as e:
                logger.warning(f"YouTube subtitle extraction (cookies={use_cookies}) failed: {e}")
        return None

    def extract_info(self, url: str) -> dict:
        return self._extract_info_internal(url, use_cookies=True)

    def _extract_info_internal(self, url: str, use_cookies: bool = True) -> dict:
        opts = self._get_base_opts()
        opts["skip_download"] = True
        if use_cookies and self._cookie_path:
            opts["cookiefile"] = self._cookie_path
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    @staticmethod
    def _parse_subtitle(content: str) -> str:
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
