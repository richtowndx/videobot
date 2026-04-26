"""Integration-style tests for the pipeline with all external I/O mocked."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest  # noqa: E402

import config   # noqa: E402
import cache    # noqa: E402
from pipeline import process_url  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "MAX_RETRIES", 3)
    monkeypatch.setattr(config, "NOTE_EXPIRE_DAYS", 30)


BILIBILI_URL = "https://www.bilibili.com/video/BV1xx411c7mD"
YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


class TestProcessUrlCacheHit:
    def test_returns_cached_note(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        cache.save_note("BV1xx411c7mD", "# Cached Note")
        result = process_url(BILIBILI_URL)
        assert result == "# Cached Note"


class TestProcessUrlFull:
    @patch("pipeline._fetch_title", return_value="Test Video Title")
    @patch("pipeline.get_transcript", return_value="This is a transcript.")
    @patch("pipeline.summarize", return_value="# Summary\nContent")
    def test_full_pipeline_bilibili(self, mock_sum, mock_trans, mock_title, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        result = process_url(BILIBILI_URL)
        assert result == "# Summary\nContent"
        mock_trans.assert_called_once()
        mock_sum.assert_called_once_with("This is a transcript.", "Test Video Title")
        # Note should be cached
        assert cache.get_cached_note("BV1xx411c7mD") == "# Summary\nContent"

    @patch("pipeline._fetch_title", return_value="YouTube Video")
    @patch("pipeline.get_transcript", return_value="YouTube transcript.")
    @patch("pipeline.summarize", return_value="# YT Summary")
    def test_full_pipeline_youtube(self, mock_sum, mock_trans, mock_title, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        result = process_url(YOUTUBE_URL)
        assert result == "# YT Summary"
        assert cache.get_cached_note("dQw4w9WgXcQ") == "# YT Summary"


class TestProcessUrlRetry:
    @patch("pipeline._fetch_title", return_value="")
    @patch("pipeline.summarize", return_value="# Final")
    @patch(
        "pipeline.get_transcript",
        side_effect=[RuntimeError("network error"), "OK transcript"],
    )
    def test_retries_on_failure(self, mock_trans, mock_sum, mock_title, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        monkeypatch.setattr(config, "MAX_RETRIES", 3)
        # Patch sleep so tests run fast
        with patch("pipeline.time.sleep"):
            result = process_url(BILIBILI_URL)
        assert result == "# Final"
        assert mock_trans.call_count == 2

    @patch("pipeline.get_transcript", side_effect=RuntimeError("always fails"))
    def test_raises_after_max_retries(self, mock_trans, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        monkeypatch.setattr(config, "MAX_RETRIES", 2)
        with patch("pipeline.time.sleep"):
            with pytest.raises(RuntimeError, match="Processing failed"):
                process_url(BILIBILI_URL)
        assert mock_trans.call_count == 2
