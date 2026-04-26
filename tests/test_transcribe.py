"""Unit tests for subtitle text cleaning in transcribe.py."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from transcribe import _clean_subtitle_text  # noqa: E402


class TestCleanSubtitleText:
    def test_vtt_basic(self):
        vtt = """\
WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world

00:00:03.000 --> 00:00:05.000
Second line
"""
        result = _clean_subtitle_text(vtt)
        assert "Hello world" in result
        assert "Second line" in result
        assert "WEBVTT" not in result
        assert "-->" not in result

    def test_removes_html_tags(self):
        vtt = """\
WEBVTT

00:00:01.000 --> 00:00:02.000
<v Speaker><b>Bold text</b></v>
"""
        result = _clean_subtitle_text(vtt)
        assert result.strip() == "Bold text"

    def test_deduplicates_lines(self):
        vtt = """\
WEBVTT

00:00:01.000 --> 00:00:02.000
Repeated line

00:00:02.000 --> 00:00:03.000
Repeated line

00:00:03.000 --> 00:00:04.000
Unique line
"""
        result = _clean_subtitle_text(vtt)
        lines = [l for l in result.splitlines() if l.strip()]
        assert lines.count("Repeated line") == 1
        assert "Unique line" in result

    def test_srt_format(self):
        srt = """\
1
00:00:01,000 --> 00:00:03,000
First subtitle

2
00:00:03,000 --> 00:00:05,000
Second subtitle
"""
        result = _clean_subtitle_text(srt)
        assert "First subtitle" in result
        assert "Second subtitle" in result
        # Sequence numbers should be gone
        assert "1" not in result.splitlines()

    def test_empty_input(self):
        assert _clean_subtitle_text("") == ""

    def test_only_headers(self):
        assert _clean_subtitle_text("WEBVTT\nKind: captions\nLanguage: zh\n").strip() == ""
