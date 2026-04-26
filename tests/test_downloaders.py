import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
from downloaders import get_downloader
from downloaders.base import BaseDownloader, DownloadResult
from downloaders.bilibili import BilibiliDownloader
from downloaders.youtube import YoutubeDownloader


def test_get_downloader_bilibili():
    d = get_downloader("bilibili")
    assert isinstance(d, BilibiliDownloader)


def test_get_downloader_youtube():
    d = get_downloader("youtube")
    assert isinstance(d, YoutubeDownloader)


def test_get_downloader_unknown():
    try:
        get_downloader("tiktok")
        assert False, "Should have raised"
    except ValueError as e:
        assert "Unsupported" in str(e)


def test_bilibili_parse_subtitle():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world

00:00:04.000 --> 00:00:06.000
This is a test
"""
    result = BilibiliDownloader._parse_subtitle(vtt)
    assert "Hello world" in result
    assert "This is a test" in result


def test_youtube_parse_subtitle():
    vtt = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000
First line

00:00:04.000 --> 00:00:06.500
Second line
"""
    result = YoutubeDownloader._parse_subtitle(vtt)
    assert "First line" in result
    assert "Second line" in result


def test_bilibili_parse_subtitle_strips_html():
    vtt = """WEBVTT
00:00:01.000 --> 00:00:03.000
<font color="white">Hello world</font>
"""
    result = BilibiliDownloader._parse_subtitle(vtt)
    assert "<font" not in result
    assert "Hello world" in result


@patch("downloaders.bilibili.yt_dlp.YoutubeDL")
def test_bilibili_extract_info(mock_ydl_cls):
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "BV1xx", "title": "Test"}
    mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)

    d = BilibiliDownloader()
    info = d.extract_info("https://bilibili.com/video/BV1xx")
    assert info["id"] == "BV1xx"


@patch("downloaders.youtube.yt_dlp.YoutubeDL")
def test_youtube_extract_info(mock_ydl_cls):
    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {"id": "abc123", "title": "YT Test"}
    mock_ydl_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl_cls.return_value.__exit__ = MagicMock(return_value=False)

    d = YoutubeDownloader()
    info = d.extract_info("https://youtube.com/watch?v=abc123")
    assert info["id"] == "abc123"


if __name__ == "__main__":
    test_get_downloader_bilibili()
    test_get_downloader_youtube()
    test_get_downloader_unknown()
    test_bilibili_parse_subtitle()
    test_youtube_parse_subtitle()
    test_bilibili_parse_subtitle_strips_html()
    test_bilibili_extract_info()
    test_youtube_extract_info()
    print("All downloader tests passed!")
