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
        # 写入中间文件，模拟下载后状态
        (task_dir / "audio.mp3").write_bytes(b"fake audio")
        mgr.save_transcript(task, "some transcript")
        assert (task_dir / "audio.mp3").exists()
        assert (task_dir / "transcript.json").exists()
        mgr.cleanup_task_files(task)
        # cleanup 保留 status.json（重建 task_dir），但中间文件被清
        assert task_dir.exists()
        assert task.status_file.exists()
        assert not (task_dir / "audio.mp3").exists()
        assert not (task_dir / "transcript.json").exists()
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

        # Transcript exists but NOT enough to resume summarization (need corrected)
        mgr.save_transcript(task, "text")
        assert not mgr.can_resume_from_summarization(task)

        # Corrected exists -> can resume from summarization
        mgr.save_corrected(task, "corrected text")
        assert mgr.can_resume_from_summarization(task)
    finally:
        teardown_temp_data(tmp)


def test_corrected_save_load():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        assert not mgr.has_corrected(task)
        mgr.save_corrected(task, "纠错后的文本")
        assert mgr.has_corrected(task)
        assert mgr.load_corrected(task) == "纠错后的文本"
    finally:
        teardown_temp_data(tmp)


def test_corrected_model_name_persisted():
    """save_corrected 带 model_name 时持久化；load_corrected_model 可读回；旧格式缺字段返回 None。"""
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")

        # 带 model_name 保存 -> 可读回
        mgr.save_corrected(task, "纠错后的文本", model_name="step-3.7-flash")
        assert mgr.load_corrected(task) == "纠错后的文本"
        assert mgr.load_corrected_model(task) == "step-3.7-flash"

        # 不带 model_name 保存（旧格式）-> 返回 None
        task2 = mgr.create_task("def456", "https://example.com/2", "youtube")
        mgr.save_corrected(task2, "另一段文本")
        assert mgr.load_corrected(task2) == "另一段文本"
        assert mgr.load_corrected_model(task2) is None
    finally:
        teardown_temp_data(tmp)


def test_resume_logic_now_requires_corrected():
    """新语义：仅有 transcript 不再算可续传至总结，需要 corrected（或 subtitle）。"""
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        mgr.save_transcript(task, "raw text")
        assert not mgr.can_resume_from_summarization(task), "仅有 transcript 不应能续传至总结"
        mgr.save_corrected(task, "corrected text")
        assert mgr.can_resume_from_summarization(task), "有 corrected 后应能续传至总结"
    finally:
        teardown_temp_data(tmp)


def test_load_backward_compat_without_correction_failed():
    """老 status.json 没有 correction_failed 字段时，加载后默认 False。"""
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        import json as _json
        data = _json.loads(task.status_file.read_text(encoding="utf-8"))
        del data["correction_failed"]
        task.status_file.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        loaded = mgr.get_task("abc123")
        assert loaded is not None
        assert loaded.correction_failed is False
    finally:
        teardown_temp_data(tmp)


def test_update_state_persists_correction_failed():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        mgr.update_state(task, TaskState.SUMMARIZING, correction_failed=True)
        assert task.correction_failed is True
        assert mgr.get_task("abc123").correction_failed is True
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
    test_corrected_save_load()
    test_resume_logic_now_requires_corrected()
    test_load_backward_compat_without_correction_failed()
    test_update_state_persists_correction_failed()
    print("All task_manager tests passed!")
