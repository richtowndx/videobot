"""Unit tests for cleanup.py."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest  # noqa: E402

import config   # noqa: E402
import cleanup  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "NOTE_EXPIRE_DAYS", 30)
    monkeypatch.setattr(config, "TASK_EXPIRE_DAYS", 1)


def _make_file(path, age_seconds: float = 0.0) -> None:
    """Create *path* and back-date its mtime by *age_seconds*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data")
    old_mtime = time.time() - age_seconds
    os.utime(path, (old_mtime, old_mtime))


class TestCleanupNotes:
    def test_fresh_note_kept(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        note = tmp_path / "notes" / "BV1.md"
        _make_file(note, age_seconds=10)
        cleanup.cleanup_notes()
        assert note.exists()

    def test_expired_note_deleted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        monkeypatch.setattr(config, "NOTE_EXPIRE_DAYS", 0)
        note = tmp_path / "notes" / "BV2.md"
        _make_file(note, age_seconds=1)  # older than 0 days
        removed = cleanup.cleanup_notes()
        assert removed == 1
        assert not note.exists()

    def test_no_notes_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        # notes/ dir doesn't exist – should not raise
        assert cleanup.cleanup_notes() == 0


class TestCleanupTasks:
    def test_fresh_task_kept(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        task_file = tmp_path / "tasks" / "BV3" / "subtitles.txt"
        _make_file(task_file, age_seconds=10)
        cleanup.cleanup_tasks()
        assert task_file.exists()

    def test_expired_task_deleted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        monkeypatch.setattr(config, "TASK_EXPIRE_DAYS", 0)
        task_file = tmp_path / "tasks" / "BV4" / "subtitles.txt"
        _make_file(task_file, age_seconds=1)
        removed = cleanup.cleanup_tasks()
        assert removed == 1
        assert not (tmp_path / "tasks" / "BV4").exists()

    def test_no_tasks_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        assert cleanup.cleanup_tasks() == 0


class TestRunCleanup:
    def test_run_cleanup_no_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        cleanup.run_cleanup()  # Should not raise
