"""
Unit tests for multi-URL extraction scenarios.
Tests extract_urls() from core.url_parser without network access.

Covers:
  1. Mixed text + single URL
  2. Multiple URLs mixed with text
  3. Multiple URLs only (no surrounding text)

Usage:
  cd /data/code/node/BiliNote/videobot
  python -m pytest tests/test_url_extraction.py -v
  # or
  python tests/test_url_extraction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.url_parser import extract_urls, extract_url


# ── Scenario 1: Mixed text + single URL ──────────────────────────────────────

def test_mixed_text_bilibili_short_link():
    text = "【离谱，Qwen3.6 27B生成速度飙到184t/s,我是怎么做到的？-哔哩哔哩】 https://b23.tv/dgxwkiN"
    result = extract_urls(text)
    assert result == ["https://b23.tv/dgxwkiN"]


def test_mixed_text_youtube_link():
    text = "check this video https://www.youtube.com/watch?v=abc123 it's great"
    result = extract_urls(text)
    assert result == ["https://www.youtube.com/watch?v=abc123"]


def test_mixed_text_bilibili_long_link():
    text = "推荐 https://www.bilibili.com/video/BV1xx411c7mD 很好看"
    result = extract_urls(text)
    assert result == ["https://www.bilibili.com/video/BV1xx411c7mD"]


def test_mixed_text_youtu_be():
    text = "看这个 https://youtu.be/xyz789 值得一看"
    result = extract_urls(text)
    assert result == ["https://youtu.be/xyz789"]


def test_mixed_text_no_valid_url():
    text = "这是一段没有链接的文字"
    result = extract_urls(text)
    assert result == []


def test_mixed_text_unsupported_url():
    text = "看看这个 https://www.google.com/search?q=test"
    result = extract_urls(text)
    assert result == []


# ── Scenario 2: Multiple URLs mixed with text ────────────────────────────────

def test_two_bilibili_urls_with_text():
    text = (
        "【离谱，Qwen3.6 27B生成速度飙到184t/s,我是怎么做到的？-哔哩哔哩】 https://b23.tv/dgxwkiN "
        "【我真在 Mac 上本地部署了 DeepSeek V4！-哔哩哔哩】 https://b23.tv/X7QBak8"
    )
    result = extract_urls(text)
    assert len(result) == 2
    assert "https://b23.tv/dgxwkiN" in result
    assert "https://b23.tv/X7QBak8" in result


def test_mixed_platform_urls_with_text():
    text = "对比这两个 https://b23.tv/aaa 和 https://www.youtube.com/watch?v=bbb"
    result = extract_urls(text)
    assert len(result) == 2
    assert "https://b23.tv/aaa" in result
    assert "https://www.youtube.com/watch?v=bbb" in result


def test_three_urls_with_numbered_list():
    text = (
        "整理几个视频：\n"
        "1. https://www.bilibili.com/video/BV1xx411c7mD\n"
        "2. https://youtu.be/abc123\n"
        "3. https://b23.tv/xyz"
    )
    result = extract_urls(text)
    assert len(result) == 3
    assert "https://www.bilibili.com/video/BV1xx411c7mD" in result
    assert "https://youtu.be/abc123" in result
    assert "https://b23.tv/xyz" in result


def test_duplicate_urls_deduplicated():
    text = "这个 https://b23.tv/aaa 还有这个 https://b23.tv/aaa 是同一个"
    result = extract_urls(text)
    assert len(result) == 1
    assert result[0] == "https://b23.tv/aaa"


def test_mixed_supported_and_unsupported():
    text = "支持 https://b23.tv/aaa 不支持 https://www.google.com 还有 https://youtu.be/bbb"
    result = extract_urls(text)
    assert len(result) == 2
    assert "https://b23.tv/aaa" in result
    assert "https://youtu.be/bbb" in result


# ── Scenario 3: URL list only (no surrounding text) ──────────────────────────

def test_single_url_only():
    text = "https://www.youtube.com/watch?v=abc123"
    result = extract_urls(text)
    assert result == ["https://www.youtube.com/watch?v=abc123"]


def test_multiple_bilibili_urls_only():
    text = (
        "https://www.bilibili.com/video/BV12To9BiEp4/?spm_id_from=333.1007.tianma.1-3-3.click "
        "https://www.bilibili.com/video/BV1BadQBmEny?spm_id_from=333.788.videopod.sections&vd_source=12e5345ff662e428baa991d1d4be34a3"
    )
    result = extract_urls(text)
    assert len(result) == 2


def test_urls_separated_by_spaces():
    text = "https://b23.tv/aaa https://youtu.be/bbb https://www.youtube.com/watch?v=ccc"
    result = extract_urls(text)
    assert len(result) == 3


def test_urls_separated_by_newlines():
    text = "https://b23.tv/aaa\nhttps://youtu.be/bbb\nhttps://b23.tv/ccc"
    result = extract_urls(text)
    assert len(result) == 3


def test_urls_with_markdown_links():
    text = (
        "[英特尔ARC下代没游戏独显_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV12To9BiEp4/?spm_id_from=333.1007.tianma.1-3-3.click) "
        "[长鑫HBM3进展缓慢_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1BadQBmEny?spm_id_from=333.788.videopod.sections&vd_source=12e5345ff662e428baa991d1d4be34a3)"
    )
    result = extract_urls(text)
    assert len(result) == 2
    assert "https://www.bilibili.com/video/BV12To9BiEp4/?spm_id_from=333.1007.tianma.1-3-3.click" in result
    assert "https://www.bilibili.com/video/BV1BadQBmEny?spm_id_from=333.788.videopod.sections&vd_source=12e5345ff662e428baa991d1d4be34a3" in result


def test_empty_string():
    result = extract_urls("")
    assert result == []


def test_none_like_input():
    result = extract_urls("   ")
    assert result == []


# ── Backward compatibility ───────────────────────────────────────────────────

def test_extract_url_still_works():
    text = "check https://www.youtube.com/watch?v=abc123 out"
    assert extract_url(text) == "https://www.youtube.com/watch?v=abc123"

    text = "no url here"
    assert extract_url(text) is None


# ── Main runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_mixed_text_bilibili_short_link()
    test_mixed_text_youtube_link()
    test_mixed_text_bilibili_long_link()
    test_mixed_text_youtu_be()
    test_mixed_text_no_valid_url()
    test_mixed_text_unsupported_url()
    test_two_bilibili_urls_with_text()
    test_mixed_platform_urls_with_text()
    test_three_urls_with_numbered_list()
    test_duplicate_urls_deduplicated()
    test_mixed_supported_and_unsupported()
    test_single_url_only()
    test_multiple_bilibili_urls_only()
    test_urls_separated_by_spaces()
    test_urls_separated_by_newlines()
    test_urls_with_markdown_links()
    test_empty_string()
    test_none_like_input()
    test_extract_url_still_works()
    print("All URL extraction tests passed!")
