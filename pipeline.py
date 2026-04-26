"""Video-processing pipeline.

Responsibilities
----------------
* URL detection for Bilibili and YouTube.
* Orchestrate transcript → summarise → cache steps with per-step
  checkpoint files so a failed run can resume from where it left off.
* Wrap execution with up to ``MAX_RETRIES`` attempts and exponential
  back-off between retries.
"""

from __future__ import annotations

import logging
import re
import time

import yt_dlp

import cache
import config
from transcribe import get_transcript
from summarize import summarize

logger = logging.getLogger(__name__)

# ── URL patterns ──────────────────────────────────────────────────────────────

#: Matches Bilibili video pages and returns the BV / AV id in group 1.
BILIBILI_RE = re.compile(
    r"https?://(?:www\.)?bilibili\.com/video/([A-Za-z0-9]+)"
)

#: Matches long-form and short YouTube URLs.
#: Group 1 → long form ``?v=…``, group 2 → short form ``youtu.be/…``.
YOUTUBE_RE = re.compile(
    r"https?://(?:www\.)?youtube\.com/watch\?(?:[^\s&]*&)*v=([A-Za-z0-9_-]+)"
    r"|https?://youtu\.be/([A-Za-z0-9_-]+)"
)

#: Combined pattern used by the bot to find any supported URL in a message.
SUPPORTED_URL_RE = re.compile(
    r"https?://(?:"
    r"(?:www\.)?bilibili\.com/video/[A-Za-z0-9]+[^\s]*"
    r"|(?:www\.)?youtube\.com/watch\?[^\s]*"
    r"|youtu\.be/[A-Za-z0-9_-]+[^\s]*"
    r")"
)


# ── URL parsing ───────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> tuple[str, str]:
    """Return ``(platform, video_id)`` for a supported URL.

    Raises ``ValueError`` for unsupported URLs.
    """
    m = BILIBILI_RE.search(url)
    if m:
        return "bilibili", m.group(1)

    m = YOUTUBE_RE.search(url)
    if m:
        return "youtube", m.group(1) or m.group(2)

    raise ValueError(f"Unsupported or unrecognised URL: {url!r}")


def find_urls(text: str) -> list[str]:
    """Return all supported video URLs found in *text*."""
    return SUPPORTED_URL_RE.findall(text)


# ── Metadata ──────────────────────────────────────────────────────────────────

def _fetch_title(url: str) -> str:
    """Return the video title via yt-dlp, or an empty string on failure."""
    ydl_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    if config.PROXY_URL:
        ydl_opts["proxy"] = config.PROXY_URL
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "") if info else ""
    except Exception as exc:
        logger.warning("could not fetch title for %s: %s", url, exc)
        return ""


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _process_once(url: str) -> str:
    """Single (non-retried) pipeline execution for *url*.

    Steps
    -----
    1. Parse URL → video_id
    2. Return cached note if fresh (cache.py)
    3. Extract subtitles or transcribe audio (transcribe.py, checkpointed)
    4. Fetch video title for richer LLM context
    5. Summarise transcript (summarize.py)
    6. Persist note to cache
    """
    _platform, video_id = extract_video_id(url)

    # Step 2 – cache check
    cached = cache.get_cached_note(video_id)
    if cached:
        logger.info("cache hit for %s", video_id)
        return cached

    # Step 3 – transcript (subtitle fast-path or Whisper fallback)
    transcript = get_transcript(url, video_id)

    # Step 4 – metadata
    title = _fetch_title(url)

    # Step 5 – summarise
    note = summarize(transcript, title)

    # Step 6 – persist
    cache.save_note(video_id, note)

    return note


def process_url(url: str) -> str:
    """Process *url* with up to ``MAX_RETRIES`` attempts.

    Each failure is logged and followed by exponential back-off before the
    next attempt.  Raises ``RuntimeError`` if every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return _process_once(url)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "attempt %d/%d failed for %s: %s",
                attempt,
                config.MAX_RETRIES,
                url,
                exc,
            )
            if attempt < config.MAX_RETRIES:
                time.sleep(2**attempt)  # 2 s, 4 s, …

    raise RuntimeError(
        f"Processing failed after {config.MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc
