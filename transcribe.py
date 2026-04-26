"""Subtitle extraction and audio transcription with checkpoint support.

Fast path
---------
``extract_subtitles`` uses yt-dlp to pull any available subtitles (Chinese
preferred, English fallback).  The cleaned plain-text is written to
``data/tasks/{video_id}/subtitles.txt`` so subsequent calls skip the network.

Slow path
---------
``transcribe_audio`` downloads the best audio track with yt-dlp, then sends
the file to the OpenAI-compatible ``/audio/transcriptions`` endpoint (Whisper).
The result is cached to ``data/tasks/{video_id}/transcript.txt``.

``get_transcript`` tries the fast path first and falls back to the slow path
automatically.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yt_dlp
import openai

import config

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tasks_dir(video_id: str) -> Path:
    d = config.DATA_DIR / "tasks" / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean_subtitle_text(content: str) -> str:
    """Strip timestamps, cue identifiers, HTML tags, and deduplicate lines."""
    lines: list[str] = []
    seen: set[str] = set()
    for raw in content.splitlines():
        line = raw.strip()
        # Skip blank lines, WEBVTT header, header metadata (e.g. "Kind: captions"),
        # NOTE blocks, SRT sequence numbers, and timestamp lines.
        if (
            not line
            or line.startswith("WEBVTT")
            or line.startswith("NOTE")
            or re.match(r"^\d+$", line)
            or "-->" in line
            or re.match(r"^[A-Za-z][\w-]*:\s*\S", line)
        ):
            continue
        # Strip inline HTML / VTT tags  (<b>, <v Speaker>, <00:00:01.000>, …)
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_subtitles(url: str, video_id: str) -> str | None:
    """Try to download subtitles for *url*.

    Returns cleaned plain-text on success, ``None`` when no subtitles are
    available.  Results are checkpointed in ``subtitles.txt``.
    """
    task_dir = _tasks_dir(video_id)
    subs_path = task_dir / "subtitles.txt"

    if subs_path.exists():
        logger.info("checkpoint: using cached subtitles for %s", video_id)
        return subs_path.read_text(encoding="utf-8")

    ydl_opts: dict = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        # Prefer Chinese, fall back to English
        "subtitleslangs": ["zh-Hans", "zh", "zh-CN", "zh-TW", "en"],
        "skip_download": True,
        "outtmpl": str(task_dir / "video"),
        "quiet": True,
        "no_warnings": True,
    }
    if config.PROXY_URL:
        ydl_opts["proxy"] = config.PROXY_URL

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        logger.warning("subtitle download failed for %s: %s", video_id, exc)

    # yt-dlp writes  video.{lang}.{ext}  or  video.{ext}
    for candidate in sorted(task_dir.glob("video.*")):
        if candidate.suffix in (".vtt", ".srt", ".ass", ".lrc"):
            raw = candidate.read_text(encoding="utf-8", errors="replace")
            text = _clean_subtitle_text(raw)
            if text.strip():
                subs_path.write_text(text, encoding="utf-8")
                logger.info("subtitles extracted for %s (%d chars)", video_id, len(text))
                return text

    return None


def transcribe_audio(url: str, video_id: str) -> str:
    """Download audio and transcribe via Whisper API.

    The transcript is checkpointed in ``transcript.txt``.
    Raises on any unrecoverable error.
    """
    task_dir = _tasks_dir(video_id)
    transcript_path = task_dir / "transcript.txt"

    if transcript_path.exists():
        logger.info("checkpoint: using cached transcript for %s", video_id)
        return transcript_path.read_text(encoding="utf-8")

    # --- Download audio ---
    audio_template = str(task_dir / "audio.%(ext)s")
    ydl_opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": audio_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    if config.PROXY_URL:
        ydl_opts["proxy"] = config.PROXY_URL

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Locate the downloaded audio file (preferably .mp3)
    audio_file: Path | None = None
    for candidate in task_dir.glob("audio.*"):
        if candidate.suffix == ".mp3":
            audio_file = candidate
            break
    if audio_file is None:
        for candidate in task_dir.glob("audio.*"):
            audio_file = candidate
            break
    if audio_file is None or not audio_file.exists():
        raise FileNotFoundError(f"Audio download produced no file in {task_dir}")

    # --- Transcribe ---
    client = openai.OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
    )
    with audio_file.open("rb") as fh:
        response = client.audio.transcriptions.create(
            model=config.WHISPER_MODEL,
            file=fh,
        )

    transcript: str = response.text
    transcript_path.write_text(transcript, encoding="utf-8")
    logger.info("transcript saved for %s (%d chars)", video_id, len(transcript))
    return transcript


def get_transcript(url: str, video_id: str) -> str:
    """Return transcript text, trying subtitles first then Whisper."""
    text = extract_subtitles(url, video_id)
    if text:
        logger.info("using subtitle text for %s", video_id)
        return text
    logger.info("no subtitles for %s – falling back to Whisper", video_id)
    return transcribe_audio(url, video_id)
