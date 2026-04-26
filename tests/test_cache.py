"""Unit tests for cache.py."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest  # noqa: E402

import config  # noqa: E402
import cache   # noqa: E402


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR to a temporary directory for each test."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "NOTE_EXPIRE_DAYS", 30)


class TestCache:
    def test_missing_returns_none(self):
        assert cache.get_cached_note("nonexistent") is None

    def test_save_and_retrieve(self):
        cache.save_note("BV123", "# My Note\nContent here")
        result = cache.get_cached_note("BV123")
        assert result == "# My Note\nContent here"

    def test_note_path_uses_data_dir(self):
        path = cache.note_path("BV456")
        assert str(config.DATA_DIR) in str(path)
        assert path.name == "BV456.md"

    def test_expired_note_returns_none(self, monkeypatch):
        cache.save_note("BVold", "stale content")
        # Pretend NOTE_EXPIRE_DAYS is 0 so any note is expired
        monkeypatch.setattr(config, "NOTE_EXPIRE_DAYS", 0)
        assert cache.get_cached_note("BVold") is None
        # File should have been deleted
        assert not cache.note_path("BVold").exists()

    def test_fresh_note_survives(self):
        cache.save_note("BVfresh", "fresh content")
        assert cache.get_cached_note("BVfresh") == "fresh content"

    def test_overwrite(self):
        cache.save_note("BVdup", "first version")
        cache.save_note("BVdup", "second version")
        assert cache.get_cached_note("BVdup") == "second version"
