import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import shutil
import tempfile

from core.task_manager import TaskManager, TaskState, Task


def setup_temp_data(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="test_videobot_")
    from config import DataConfig
    DataConfig.DATA_DIR = Path(tmp)
    DataConfig.TASKS_DIR = Path(tmp) / "tasks"
    DataConfig.NOTES_DIR = Path(tmp) / "notes"
    DataConfig.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    return tmp


def teardown_temp_data(tmp):
    shutil.rmtree(tmp, ignore_errors=True)


def test_create_task():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        assert task.task_id == "abc123"
        assert task.url == "https://example.com"
        assert task.platform == "youtube"
        assert task.state == TaskState.PENDING
    finally:
        teardown_temp_data(tmp)


def test_get_task():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        mgr.create_task("abc123", "https://example.com", "youtube")
        task = mgr.get_task("abc123")
        assert task is not None
        assert task.task_id == "abc123"
    finally:
        teardown_temp_data(tmp)


def test_get_task_not_found():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.get_task("nonexistent")
        assert task is None
    finally:
        teardown_temp_data(tmp)


def test_update_state():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        mgr.update_state(task, TaskState.DOWNLOADING)
        assert task.state == TaskState.DOWNLOADING

        loaded = mgr.get_task("abc123")
        assert loaded.state == TaskState.DOWNLOADING
    finally:
        teardown_temp_data(tmp)


def test_get_or_create_new():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task, is_cached = mgr.get_or_create("https://youtube.com/watch?v=abc", "youtube")
        assert not is_cached
        assert task.state == TaskState.PENDING
    finally:
        teardown_temp_data(tmp)


def test_get_or_create_cached():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task1, cached1 = mgr.get_or_create("https://youtube.com/watch?v=abc", "youtube")
        assert not cached1

        task2, cached2 = mgr.get_or_create("https://youtube.com/watch?v=abc", "youtube")
        assert cached2
        assert task2.task_id == task1.task_id
    finally:
        teardown_temp_data(tmp)


def test_completed_note_cache():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task_id = "abc123"
        note_file = Path(tmp) / "notes" / f"{task_id}.md"
        note_file.write_text("# Test Summary", encoding="utf-8")

        task, is_cached = mgr.get_or_create("https://example.com", "youtube")
        # The task_id depends on URL hash, so test with explicit note file
        mgr.save_note(Task(task_id=task_id, url="x", platform="youtube"), "# Test Summary")
        result = mgr.get_cached_result(task_id)
        assert result == "# Test Summary"
    finally:
        teardown_temp_data(tmp)


def test_transcript_save_load():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        mgr.save_transcript(task, "Hello world transcript")
        loaded = mgr.load_transcript(task)
        assert loaded == "Hello world transcript"
    finally:
        teardown_temp_data(tmp)


def test_cleanup_task_files():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        task_dir = task.task_dir
        assert task_dir.exists()
        mgr.cleanup_task_files(task)
        assert not task_dir.exists()
    finally:
        teardown_temp_data(tmp)


def test_has_audio():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        assert not mgr.has_audio(task)
        # Create audio file
        (task.task_dir / "audio.mp3").write_bytes(b"fake audio")
        assert mgr.has_audio(task)
    finally:
        teardown_temp_data(tmp)


def test_resume_logic():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")

        # No progress yet
        assert not mgr.can_resume_from_transcription(task)
        assert not mgr.can_resume_from_summarization(task)

        # Audio exists - can resume from transcription
        (task.task_dir / "audio.mp3").write_bytes(b"fake")
        assert mgr.can_resume_from_transcription(task)

        # Transcript exists - can resume from summarization
        mgr.save_transcript(task, "text")
        assert mgr.can_resume_from_summarization(task)
    finally:
        teardown_temp_data(tmp)


if __name__ == "__main__":
    test_create_task()
    test_get_task()
    test_get_task_not_found()
    test_update_state()
    test_get_or_create_new()
    test_get_or_create_cached()
    test_completed_note_cache()
    test_transcript_save_load()
    test_cleanup_task_files()
    test_has_audio()
    test_resume_logic()
    print("All task_manager tests passed!")
