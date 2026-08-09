"""
Test pipeline resume from intermediate steps.

Verifies that when a task fails at an intermediate step:
1. Intermediate files from completed steps are preserved
2. Re-processing the same URL resumes from the failed step
3. Already-completed steps (download, transcription) are NOT re-executed

Pipeline steps: DOWNLOADING -> TRANSCRIBING -> SUMMARIZING -> COMPLETED
Resume mechanism: file-existence checks (has_audio / has_transcript / has_subtitle)

Usage:
  cd /data/code/node/BiliNote/videobot
  python -m pytest tests/test_pipeline_resume.py -v
  python tests/test_pipeline_resume.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
import tempfile
from unittest import mock

from core.task_manager import TaskManager, TaskState
from core.pipeline import Pipeline


BILIBILI_URL = "https://www.bilibili.com/video/BV1aDD3B1Ea4/"


def _setup(tmp_dir):
    """Point DataConfig to a temp directory for test isolation."""
    from config import DataConfig
    DataConfig.DATA_DIR = Path(tmp_dir)
    DataConfig.TASKS_DIR = Path(tmp_dir) / "tasks"
    DataConfig.NOTES_DIR = Path(tmp_dir) / "notes"
    DataConfig.AUDIO_DIR = Path(tmp_dir) / "audio"
    DataConfig.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _mock_downloader(title="Test Video"):
    """Create a mock downloader that simulates audio download by writing a file."""
    dl = mock.MagicMock()
    dl.extract_info.return_value = {"title": title}
    dl.extract_subtitles.return_value = None  # Force audio-download path

    def fake_download(url, output_dir):
        audio_path = Path(output_dir) / "audio.mp3"
        audio_path.write_bytes(b"fake audio content for testing")
        result = mock.MagicMock()
        result.title = title
        result.file_path = str(audio_path)
        return result

    dl.download_audio.side_effect = fake_download
    return dl


# ── Test 1: Resume from transcription step ──────────────────────────────────

def test_resume_from_transcription():
    """
    Phase 1: download audio succeeds, transcription FAILS
    Phase 2: re-submit same URL -> skip download, redo transcription
    """
    tmp = tempfile.mkdtemp(prefix="test_resume_")
    _setup(tmp)
    try:
        url = BILIBILI_URL
        platform = "bilibili"
        task_manager = TaskManager()
        pipeline = Pipeline()

        task, _ = task_manager.get_or_create(url, platform)

        # ── Phase 1: Download OK, Transcription FAILS ──
        mock_transcriber_fail = mock.MagicMock()
        mock_transcriber_fail.transcript.side_effect = RuntimeError("Whisper engine crashed")
        pipeline._transcriber = mock_transcriber_fail
        pipeline._summarizer = mock.MagicMock()

        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is None, "Phase 1 should return None"
        assert task.state == TaskState.FAILED
        assert task_manager.has_audio(task), "Audio file should exist after download"
        assert not task_manager.has_transcript(task), "No transcript (transcription failed)"
        assert task.error == "Whisper engine crashed"
        print("  Phase 1: download OK, transcription FAILED ✓")

        # ── Phase 2: Resume -> skip download, redo transcription ──
        mock_transcriber_ok = mock.MagicMock()
        mock_transcriber_ok.transcript.return_value = "Transcribed video content for testing."
        pipeline._transcriber = mock_transcriber_ok

        mock_summarizer = mock.MagicMock()
        mock_summarizer.correct.return_value = "Corrected transcript."
        mock_summarizer.summarize.return_value = "# Video Summary\n\nTest content."
        pipeline._summarizer = mock_summarizer

        mock_dl = _mock_downloader()
        with mock.patch("core.pipeline.get_downloader", return_value=mock_dl):
            task = task_manager.get_task(task.task_id)
            result = pipeline.process(url, task)

        assert result is not None, "Phase 2 should succeed"
        assert task.state == TaskState.COMPLETED

        # KEY: download was SKIPPED (audio already existed)
        mock_dl.download_audio.assert_not_called()
        # Transcription RAN (was the failed step)
        mock_transcriber_ok.transcript.assert_called_once()
        # Summarization RAN
        mock_summarizer.summarize.assert_called_once()

        print("  Phase 2: download SKIPPED, transcription + summarization OK ✓")
        print("test_resume_from_transcription PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 2: Resume from summarization step ──────────────────────────────────

def test_resume_from_summarization():
    """
    Phase 1: download + transcription succeed, summarization FAILS
    Phase 2: re-submit same URL -> skip download AND transcription, redo summarization
    """
    tmp = tempfile.mkdtemp(prefix="test_resume_")
    _setup(tmp)
    try:
        url = BILIBILI_URL
        platform = "bilibili"
        task_manager = TaskManager()
        pipeline = Pipeline()

        task, _ = task_manager.get_or_create(url, platform)

        # ── Phase 1: Download OK, Transcription OK, Summarization FAILS ──
        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcript.return_value = "Transcribed text for summarization."
        pipeline._transcriber = mock_transcriber

        mock_summarizer_fail = mock.MagicMock()
        mock_summarizer_fail.correct.return_value = "Corrected transcript."
        mock_summarizer_fail.summarize.side_effect = RuntimeError("LLM API timeout")
        pipeline._summarizer = mock_summarizer_fail

        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is None, "Phase 1 should fail at summarization"
        assert task.state == TaskState.FAILED
        assert task_manager.has_audio(task), "Audio should exist"
        assert task_manager.has_transcript(task), "Transcript should exist"
        assert not task_manager.get_cached_result(task.task_id), "No final note"
        assert task.error == "LLM API timeout"
        print("  Phase 1: download + transcription OK, summarization FAILED ✓")

        # ── Phase 2: Resume -> skip download AND transcription ──
        mock_summarizer_ok = mock.MagicMock()
        mock_summarizer_ok.summarize.return_value = "# Resumed Summary\n\nFinal content."
        pipeline._summarizer = mock_summarizer_ok

        mock_dl = _mock_downloader()
        with mock.patch("core.pipeline.get_downloader", return_value=mock_dl):
            task = task_manager.get_task(task.task_id)
            result = pipeline.process(url, task)

        assert result is not None, "Phase 2 should succeed"
        assert task.state == TaskState.COMPLETED

        # Download SKIPPED
        mock_dl.download_audio.assert_not_called()
        # Transcriber called only once TOTAL (from Phase 1, NOT again in Phase 2)
        mock_transcriber.transcript.assert_called_once()
        # Summarization RAN (was the failed step)
        mock_summarizer_ok.summarize.assert_called_once()

        print("  Phase 2: download + transcription SKIPPED, summarization OK ✓")
        print("test_resume_from_summarization PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 3: Multi-step progressive resume ────────────────────────────────────

def test_multi_step_resume():
    """
    Full lifecycle across 3 failures:
      Phase 1: fail at transcription        -> audio exists
      Phase 2: resume, fail at summarization -> audio + transcript exist
      Phase 3: resume, full success          -> all done
      Phase 4: verify cached result
    """
    tmp = tempfile.mkdtemp(prefix="test_resume_")
    _setup(tmp)
    try:
        url = BILIBILI_URL
        platform = "bilibili"
        task_manager = TaskManager()
        pipeline = Pipeline()

        task, _ = task_manager.get_or_create(url, platform)

        # ── Phase 1: Fail at transcription ──
        mock_t1 = mock.MagicMock()
        mock_t1.transcript.side_effect = RuntimeError("Whisper OOM")
        pipeline._transcriber = mock_t1
        pipeline._summarizer = mock.MagicMock()

        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is None
        assert task.state == TaskState.FAILED
        assert task_manager.has_audio(task), "Audio downloaded"
        assert not task_manager.has_transcript(task), "No transcript yet"
        print("  Phase 1: FAILED at transcription, audio exists ✓")

        # ── Phase 2: Transcription OK, summarization FAILS ──
        mock_t2 = mock.MagicMock()
        mock_t2.transcript.return_value = "Phase 2 transcript text."
        pipeline._transcriber = mock_t2

        mock_s2 = mock.MagicMock()
        mock_s2.correct.return_value = "Corrected transcript."
        mock_s2.summarize.side_effect = RuntimeError("API rate limit exceeded")
        pipeline._summarizer = mock_s2

        mock_dl2 = _mock_downloader()
        with mock.patch("core.pipeline.get_downloader", return_value=mock_dl2):
            task = task_manager.get_task(task.task_id)
            result = pipeline.process(url, task)

        assert result is None
        assert task.state == TaskState.FAILED
        mock_dl2.download_audio.assert_not_called()  # Download still skipped
        assert task_manager.has_transcript(task), "Transcript now exists"
        assert task_manager.has_audio(task), "Audio still exists"
        print("  Phase 2: download SKIPPED, FAILED at summarization ✓")

        # ── Phase 3: Full success ──
        mock_s3 = mock.MagicMock()
        mock_s3.summarize.return_value = "# Final Summary\n\nDone."
        pipeline._summarizer = mock_s3

        mock_dl3 = _mock_downloader()
        with mock.patch("core.pipeline.get_downloader", return_value=mock_dl3):
            task = task_manager.get_task(task.task_id)
            result = pipeline.process(url, task)

        assert result is not None
        assert task.state == TaskState.COMPLETED
        mock_dl3.download_audio.assert_not_called()  # Download STILL skipped
        mock_t2.transcript.assert_called_once()  # Transcriber from Phase 2 only called once
        mock_s3.summarize.assert_called_once()  # Summarization succeeded now
        print("  Phase 3: download + transcription SKIPPED, summarization OK ✓")

        # ── Phase 4: Cached result ──
        cached_task, is_cached = task_manager.get_or_create(url, platform)
        assert is_cached
        assert cached_task.state == TaskState.COMPLETED
        cached = task_manager.get_cached_result(cached_task.task_id)
        assert cached == "# Final Summary\n\nDone."
        print("  Phase 4: Cache verified ✓")

        print("test_multi_step_resume PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 4: Verify intermediate file state at each failure point ─────────────

def test_intermediate_file_states():
    """
    After each failure, verify exactly which files exist and which don't.
    This ensures the resume logic has the right signals to detect progress.
    """
    tmp = tempfile.mkdtemp(prefix="test_resume_")
    _setup(tmp)
    try:
        url = BILIBILI_URL
        platform = "bilibili"
        task_manager = TaskManager()
        pipeline = Pipeline()

        task, _ = task_manager.get_or_create(url, platform)
        task_id = task.task_id

        # State 0: Initial
        assert not task_manager.has_audio(task)
        assert not task_manager.has_transcript(task)
        assert not task_manager.has_subtitle(task)
        assert not task_manager.get_cached_result(task_id)
        print("  State 0: empty task directory ✓")

        # State 1: After download failure (simulate by creating audio, no transcript)
        from config import DataConfig
        audio_dir = DataConfig.AUDIO_DIR / task_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "audio.mp3").write_bytes(b"fake")
        task_manager.update_state(task, TaskState.FAILED, error="transcription crash")

        assert task_manager.has_audio(task)
        assert not task_manager.has_transcript(task)
        assert task_manager.can_resume_from_transcription(task)
        assert not task_manager.can_resume_from_summarization(task)
        print("  State 1: audio only -> can_resume_from_transcription ✓")

        # State 2: After transcription (simulate transcript file)
        task_manager.save_transcript(task, "Transcribed content.")
        task_manager.update_state(task, TaskState.FAILED, error="summarization crash")

        assert task_manager.has_audio(task)
        assert task_manager.has_transcript(task)
        assert task_manager.can_resume_from_transcription(task)
        assert not task_manager.can_resume_from_summarization(task), "仅有 transcript 不应能续传至总结"
        print("  State 2: audio + transcript -> still needs correction ✓")

        # State 2b: After correction (simulate corrected file)
        task_manager.save_corrected(task, "Corrected content.")
        assert task_manager.can_resume_from_summarization(task)
        print("  State 2b: + corrected -> can_resume_from_summarization ✓")

        # State 3: After completion (simulate note + cleanup)
        task_manager.save_note(task, "# Summary")
        task_manager.update_state(task, TaskState.COMPLETED)
        task_manager.cleanup_task_files(task)

        assert task_manager.has_audio(task), "完成后音频应在 AUDIO_DIR 保留"
        assert not task_manager.has_transcript(task), "transcript 被清"
        assert not (task.task_dir / "audio.mp3").exists(), "task_dir 内不再有音频"
        assert task_manager.get_cached_result(task_id)  # note persists
        print("  State 3: completed -> audio retained in AUDIO_DIR, task_dir cleaned, note cached ✓")

        print("test_intermediate_file_states PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 5: Correction failure falls back to raw transcript ────────────────

def test_correction_failure_falls_back_to_raw():
    """纠错抛错时，流水线降级用原始转写总结、不写 .mp3.md、标记 correction_failed。"""
    tmp = tempfile.mkdtemp(prefix="test_corr_fail_")
    _setup(tmp)
    try:
        from config import DataConfig
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcript.return_value = "RAW TRANSCRIPT TEXT"
        pipeline._transcriber = mock_transcriber

        mock_summarizer = mock.MagicMock()
        mock_summarizer.correct.side_effect = RuntimeError("all models failed")
        mock_summarizer.summarize.return_value = "# Summary"
        pipeline._summarizer = mock_summarizer

        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is not None, "纠错失败不应阻断流水线"
        assert task.state == TaskState.COMPLETED
        assert task.correction_failed is True
        # 总结吃的是原始转写（未纠错）
        mock_summarizer.summarize.assert_called_once()
        summarize_args = mock_summarizer.summarize.call_args
        text_arg = summarize_args.args[1] if len(summarize_args.args) > 1 else summarize_args.args[0]
        assert text_arg == "RAW TRANSCRIPT TEXT"
        # 未产出 .mp3.md
        assert not (DataConfig.NOTES_DIR / f"{task.task_id}.mp3.md").exists()
        print("test_correction_failure_falls_back_to_raw PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 6: Correction writes .mp3.md and feeds summary ─────────────────────

def test_correction_writes_mp3md_and_feeds_summary():
    """正常纠错：写出 notes/{id}.mp3.md，总结吃纠错稿而非原始转写。"""
    tmp = tempfile.mkdtemp(prefix="test_corr_ok_")
    _setup(tmp)
    try:
        from config import DataConfig
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcript.return_value = "RAW TEXT"
        pipeline._transcriber = mock_transcriber

        mock_summarizer = mock.MagicMock()
        mock_summarizer.correct.return_value = "CORRECTED TEXT"
        mock_summarizer._last_model_name = "step-3.7-flash"
        mock_summarizer.summarize.return_value = "# Summary"
        pipeline._summarizer = mock_summarizer

        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is not None
        assert task.state == TaskState.COMPLETED
        # .mp3.md 已写入并含纠错正文
        mp3 = DataConfig.NOTES_DIR / f"{task.task_id}.mp3.md"
        assert mp3.exists(), "纠错稿 .mp3.md 应已写入"
        mp3_content = mp3.read_text(encoding="utf-8")
        assert "CORRECTED TEXT" in mp3_content
        # 头部含纠错所用模型名（来源之后）
        assert "AI 模型：step-3.7-flash" in mp3_content
        assert "CORRECTED TEXT" in mp3.read_text(encoding="utf-8")
        # 总结吃的是纠错稿
        text_arg = mock_summarizer.summarize.call_args.args[1]
        assert text_arg == "CORRECTED TEXT"
        print("test_correction_writes_mp3md_and_feeds_summary PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 7: Subtitle path skips correction ──────────────────────────────────

def test_subtitle_path_skips_correction():
    """字幕路径不纠错、不写 .mp3.md，但仍走总结。"""
    tmp = tempfile.mkdtemp(prefix="test_sub_")
    _setup(tmp)
    try:
        from config import DataConfig
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        # downloader 返回字幕、不下载音频
        dl = mock.MagicMock()
        dl.extract_info.return_value = {"title": "Subtitled Video"}
        dl.extract_subtitles.return_value = "SUBTITLE TEXT"

        mock_summarizer = mock.MagicMock()
        mock_summarizer.summarize.return_value = "# Summary"
        pipeline._summarizer = mock_summarizer

        with mock.patch("core.pipeline.get_downloader", return_value=dl):
            result = pipeline.process(url, task)

        assert result is not None
        mock_summarizer.correct.assert_not_called()
        mock_summarizer.summarize.assert_called_once()
        assert mock_summarizer.summarize.call_args.args[1] == "SUBTITLE TEXT"
        assert not (DataConfig.NOTES_DIR / f"{task.task_id}.mp3.md").exists()
        print("test_subtitle_path_skips_correction PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 8: Resume recreates missing .mp3.md ─────────────────────────────────

def test_resume_recreates_missing_mp3md():
    """resume 时若 corrected.json 存在但 .mp3.md 缺失（落盘缝隙），应重新补写交付物。"""
    tmp = tempfile.mkdtemp(prefix="test_resume_mp3_")
    _setup(tmp)
    try:
        from config import DataConfig
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        # 构造 resume 状态：转录与纠错已完成（corrected.json 存在），但 .mp3.md 缺失
        task_manager.save_transcript(task, "RAW TRANSCRIPT")
        task_manager.save_corrected(task, "CORRECTED BODY")

        mock_transcriber = mock.MagicMock()  # has_transcript 为真 → 不会被调用
        pipeline._transcriber = mock_transcriber

        mock_summarizer = mock.MagicMock()
        mock_summarizer.summarize.return_value = "# Summary"
        pipeline._summarizer = mock_summarizer

        mp3 = DataConfig.NOTES_DIR / f"{task.task_id}.mp3.md"
        assert not mp3.exists(), "前置：.mp3.md 尚不存在"

        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is not None
        assert task.state == TaskState.COMPLETED
        mock_transcriber.transcript.assert_not_called()
        mock_summarizer.correct.assert_not_called()
        mock_summarizer.summarize.assert_called_once()
        assert mock_summarizer.summarize.call_args.args[1] == "CORRECTED BODY"
        assert mp3.exists(), "resume 后 .mp3.md 应被补写"
        assert "CORRECTED BODY" in mp3.read_text(encoding="utf-8")
        print("test_resume_recreates_missing_mp3md PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Test 9: Correction sets CORRECTING state ────────────────────────────────────────

def test_correction_sets_correcting_state():
    """音频路径纠错时，update_state 应被以 CORRECTING 调用过。"""
    tmp = tempfile.mkdtemp(prefix="test_corr_state_")
    _setup(tmp)
    try:
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        pipeline._transcriber = mock.MagicMock()
        pipeline._transcriber.transcript.return_value = "RAW TEXT"
        pipeline._summarizer = mock.MagicMock()
        pipeline._summarizer.correct.return_value = "CORRECTED"
        pipeline._summarizer._last_model_name = "test-model"
        pipeline._summarizer.summarize.return_value = "# Summary"

        # Capture state transitions by monkey-patching the method
        seen_states = []
        original_update_state = pipeline.task_manager.update_state.__func__  # Get unbound method
        
        def patched_update_state(self_mgr, task_obj, state, **kwargs):
            if isinstance(state, TaskState):
                seen_states.append(state)
            return original_update_state(self_mgr, task_obj, state, **kwargs)
        
        # Temporarily replace the method
        import types
        pipeline.task_manager.update_state = types.MethodType(patched_update_state, pipeline.task_manager)

        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is not None
        assert TaskState.CORRECTING in seen_states, f"期望经过 CORRECTING，实际：{seen_states}"
        print("test_correction_sets_correcting_state PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Main runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_resume_from_transcription()
    test_resume_from_summarization()
    test_multi_step_resume()
    test_intermediate_file_states()
    test_correction_failure_falls_back_to_raw()
    test_correction_writes_mp3md_and_feeds_summary()
    test_subtitle_path_skips_correction()
    test_resume_recreates_missing_mp3md()
    test_correction_sets_correcting_state()
    print("All pipeline resume tests passed!")
