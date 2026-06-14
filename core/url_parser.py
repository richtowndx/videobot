import hashlib
import re
from typing import Optional
from urllib.parse import urlparse


PLATFORM_MAP = {
    "bilibili.com": "bilibili",
    "b23.tv": "bilibili",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
}


def parse_platform(url: str) -> Optional[str]:
    """Extract platform name from URL domain."""
    try:
        domain = urlparse(url).hostname or ""
        for key, platform in PLATFORM_MAP.items():
            if key in domain:
                return platform
    except Exception:
        return None
    return None


def url_to_task_id(url: str) -> str:
    """Generate MD5 hash of URL as unique task ID."""
    return hashlib.md5(url.strip().encode()).hexdigest()


def is_video_url(text: str) -> bool:
    """Check if text contains a supported video URL."""
    supported = "|".join(re.escape(k) for k in PLATFORM_MAP)
    return bool(re.search(supported, text))


def extract_urls(text: str) -> list[str]:
    """Extract all supported video URLs from text, preserving order and deduplicating."""
    url_pattern = re.compile(r"https?://[^\s<>\"'\]\)]+")
    results = []
    seen = set()
    for match in url_pattern.finditer(text):
        url = match.group(0).rstrip(".,;:!?)")
        if url in seen:
            continue
        if parse_platform(url):
            results.append(url)
            seen.add(url)
    return results


def extract_url(text: str) -> Optional[str]:
    """Extract the first supported video URL from text."""
    urls = extract_urls(text)
    return urls[0] if urls else None
