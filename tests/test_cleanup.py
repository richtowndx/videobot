import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import os
import shutil
import tempfile
import time

from utils.cleanup import _cleanup_notes, _cleanup_stale_tasks


def setup_temp_data():
    tmp = tempfile.mkdtemp(prefix="test_cleanup_")
    from config import DataConfig
    DataConfig.NOTES_DIR = Path(tmp) / "notes"
    DataConfig.TASKS_DIR = Path(tmp) / "tasks"
    DataConfig.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    return tmp


def teardown(tmp):
    shutil.rmtree(tmp, ignore_errors=True)


def test_cleanup_old_notes():
    tmp = setup_temp_data()
    try:
        from config import DataConfig

        # Create an old note (>30 days)
        old_note = DataConfig.NOTES_DIR / "old.md"
        old_note.write_text("old note", encoding="utf-8")
        old_time = time.time() - (31 * 86400)
        os.utime(old_note, (old_time, old_time))

        # Create a recent note
        recent_note = DataConfig.NOTES_DIR / "recent.md"
        recent_note.write_text("recent note", encoding="utf-8")

        asyncio.get_event_loop().run_until_complete(_cleanup_notes())

        assert not old_note.exists()
        assert recent_note.exists()
    finally:
        teardown(tmp)


def test_cleanup_stale_tasks():
    tmp = setup_temp_data()
    try:
        from config import DataConfig

        # Create an old task dir (>1 day)
        old_task = DataConfig.TASKS_DIR / "old_task"
        old_task.mkdir()
        (old_task / "status.json").write_text("{}", encoding="utf-8")
        old_time = time.time() - (2 * 86400)
        os.utime(old_task, (old_time, old_time))

        # Create a recent task dir
        recent_task = DataConfig.TASKS_DIR / "recent_task"
        recent_task.mkdir()
        (recent_task / "status.json").write_text("{}", encoding="utf-8")

        asyncio.get_event_loop().run_until_complete(_cleanup_stale_tasks())

        assert not old_task.exists()
        assert recent_task.exists()
    finally:
        teardown(tmp)


def test_cleanup_nothing_to_clean():
    tmp = setup_temp_data()
    try:
        # Should not raise on empty dirs
        asyncio.get_event_loop().run_until_complete(_cleanup_notes())
        asyncio.get_event_loop().run_until_complete(_cleanup_stale_tasks())
    finally:
        teardown(tmp)


if __name__ == "__main__":
    test_cleanup_old_notes()
    test_cleanup_stale_tasks()
    test_cleanup_nothing_to_clean()
    print("All cleanup tests passed!")
