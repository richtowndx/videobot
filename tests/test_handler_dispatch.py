import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile, shutil
from unittest import mock


def _setup_handler_tmp():
    tmp = tempfile.mkdtemp(prefix="test_handler_")
    from config import DataConfig
    DataConfig.TASKS_DIR = Path(tmp) / "tasks"
    DataConfig.NOTES_DIR = Path(tmp) / "notes"
    DataConfig.AUDIO_DIR = Path(tmp) / "audio"
    for d in (DataConfig.TASKS_DIR, DataConfig.NOTES_DIR, DataConfig.AUDIO_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return tmp


def test_handler_returns_cached_without_pipeline():
    """return_cached：发缓存笔记，不调用 pipeline.process。"""
    from bot import handler
    tmp = _setup_handler_tmp()
    try:
        task = mock.MagicMock(task_id="abc", title="T", platform="bilibili", url="u")
        with mock.patch.object(handler, "task_manager") as tm, \
             mock.patch.object(handler, "pipeline") as pl, \
             mock.patch.object(handler, "_send_note", new=mock.AsyncMock()) as send_note, \
             mock.patch.object(handler, "parse_platform", return_value="bilibili"):
            tm.get_or_create.return_value = (task, True)
            tm.get_resume_action.return_value = "return_cached"
            tm.get_cached_result.return_value = "CACHED"
            tm.get_task.return_value = None
            ok, err = asyncio.run(handler._process_single_url(mock.MagicMock(), mock.MagicMock(), "u", 1, 1))
        assert ok is True and err == ""
        send_note.assert_awaited_once()
        pl.process.assert_not_called()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_handler_runs_pipeline_on_reprocess():
    """reprocess：调用 pipeline.process 并发送其结果。"""
    from bot import handler
    tmp = _setup_handler_tmp()
    try:
        task = mock.MagicMock(task_id="abc", title="T", platform="bilibili", url="u")
        with mock.patch.object(handler, "task_manager") as tm, \
             mock.patch.object(handler, "pipeline") as pl, \
             mock.patch.object(handler, "_send_note", new=mock.AsyncMock()) as send_note, \
             mock.patch.object(handler, "parse_platform", return_value="bilibili"):
            tm.get_or_create.return_value = (task, True)
            tm.get_resume_action.return_value = "reprocess"
            pl.process.return_value = mock.MagicMock(title="T", content="# New", model_name="m")
            status_msg = mock.MagicMock()
            status_msg.edit_text = mock.AsyncMock()
            ok, err = asyncio.run(handler._process_single_url(mock.MagicMock(), status_msg, "u", 1, 1))
        assert ok is True and err == ""
        pl.process.assert_called_once()
        send_note.assert_awaited_once()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_handler_returns_cached_without_pipeline()
    test_handler_runs_pipeline_on_reprocess()
    print("All handler dispatch tests passed!")
