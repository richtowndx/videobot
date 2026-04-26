import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
from core.url_parser import parse_platform, url_to_task_id, is_video_url, extract_url


def test_parse_platform_youtube():
    assert parse_platform("https://www.youtube.com/watch?v=abc123") == "youtube"
    assert parse_platform("https://youtu.be/abc123") == "youtube"
    assert parse_platform("https://m.youtube.com/watch?v=abc123") == "youtube"


def test_parse_platform_bilibili():
    assert parse_platform("https://www.bilibili.com/video/BV1xx411c7mD") == "bilibili"
    assert parse_platform("https://b23.tv/abc123") == "bilibili"


def test_parse_platform_unknown():
    assert parse_platform("https://www.google.com/search?q=test") is None
    assert parse_platform("not a url") is None


def test_url_to_task_id():
    url = "https://www.youtube.com/watch?v=abc123"
    expected = hashlib.md5(url.encode()).hexdigest()
    assert url_to_task_id(url) == expected
    assert len(url_to_task_id(url)) == 32


def test_url_to_task_id_deterministic():
    url = "https://www.youtube.com/watch?v=abc123"
    assert url_to_task_id(url) == url_to_task_id(url)


def test_url_to_task_id_different_urls():
    url1 = "https://www.youtube.com/watch?v=abc"
    url2 = "https://www.youtube.com/watch?v=def"
    assert url_to_task_id(url1) != url_to_task_id(url2)


def test_is_video_url():
    assert is_video_url("check this https://www.youtube.com/watch?v=abc")
    assert is_video_url("https://bilibili.com/video/BV1xx")
    assert not is_video_url("https://www.google.com")
    assert not is_video_url("hello world")


def test_extract_url():
    text = "check https://www.youtube.com/watch?v=abc123 out"
    assert extract_url(text) == "https://www.youtube.com/watch?v=abc123"

    text = "no url here"
    assert extract_url(text) is None

    text = "https://www.google.com/search?q=test"
    assert extract_url(text) is None


if __name__ == "__main__":
    test_parse_platform_youtube()
    test_parse_platform_bilibili()
    test_parse_platform_unknown()
    test_url_to_task_id()
    test_url_to_task_id_deterministic()
    test_url_to_task_id_different_urls()
    test_is_video_url()
    test_extract_url()
    print("All url_parser tests passed!")
