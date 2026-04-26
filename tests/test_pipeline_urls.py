"""Unit tests for URL parsing helpers in pipeline.py."""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Provide minimal stubs for required env vars so config.py doesn't raise.
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest  # noqa: E402

from pipeline import extract_video_id, find_urls, BILIBILI_RE, YOUTUBE_RE  # noqa: E402


# ── extract_video_id ──────────────────────────────────────────────────────────

class TestExtractVideoId:
    def test_bilibili_bv(self):
        url = "https://www.bilibili.com/video/BV1xx411c7mD"
        platform, vid = extract_video_id(url)
        assert platform == "bilibili"
        assert vid == "BV1xx411c7mD"

    def test_bilibili_av(self):
        url = "https://www.bilibili.com/video/av12345678"
        platform, vid = extract_video_id(url)
        assert platform == "bilibili"
        assert vid == "av12345678"

    def test_youtube_long(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        platform, vid = extract_video_id(url)
        assert platform == "youtube"
        assert vid == "dQw4w9WgXcQ"

    def test_youtube_long_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s&list=PL123"
        platform, vid = extract_video_id(url)
        assert platform == "youtube"
        assert vid == "dQw4w9WgXcQ"

    def test_youtube_short(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        platform, vid = extract_video_id(url)
        assert platform == "youtube"
        assert vid == "dQw4w9WgXcQ"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            extract_video_id("https://vimeo.com/123456789")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            extract_video_id("")


# ── find_urls ─────────────────────────────────────────────────────────────────

class TestFindUrls:
    def test_single_bilibili(self):
        msg = "看看这个 https://www.bilibili.com/video/BV1xx411c7mD 很有意思"
        assert find_urls(msg) == ["https://www.bilibili.com/video/BV1xx411c7mD"]

    def test_single_youtube(self):
        msg = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert find_urls(msg) == ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]

    def test_multiple_urls(self):
        msg = (
            "B站 https://www.bilibili.com/video/BV1xx411c7mD "
            "和 YouTube https://youtu.be/dQw4w9WgXcQ"
        )
        urls = find_urls(msg)
        assert len(urls) == 2
        assert any("bilibili" in u for u in urls)
        assert any("youtu.be" in u for u in urls)

    def test_no_urls(self):
        assert find_urls("没有链接的纯文本消息") == []

    def test_vimeo_not_matched(self):
        assert find_urls("https://vimeo.com/123456789") == []
