import gc
import ctypes
import logging
import resource
import threading
import time
from dataclasses import dataclass
from typing import Optional

from config import DataConfig, PipelineConfig
from core.task_manager import TaskManager, Task, TaskState
from downloaders import get_downloader
from summarizer.llm import LLMSummarizer

logger = logging.getLogger(__name__)

# glibc: release freed heap pages back to OS
try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _malloc_trim():
    if _libc:
        _libc.malloc_trim(0)


def _mem_mb():
    """Get current RSS memory in MB (not peak)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # Linux: KB -> MB
    except Exception:
        pass
    # macOS: ru_maxrss is in bytes, Linux kernel reports in KB
    import sys
    if sys.platform == "darwin":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # bytes -> MB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB


@dataclass
class PipelineResult:
    task_id: str
    title: str
    content: str
    model_name: Optional[str] = None


class Pipeline:
    def __init__(self):
        self.task_manager = TaskManager()
        self.data_config = DataConfig
        self._summarizer = None
        self._transcriber = None
        self._transcriber_lock = threading.Lock()
        self._processing = False
        self._processing_lock = threading.Lock()

    @property
    def summarizer(self):
        if self._summarizer is None:
            self._summarizer = LLMSummarizer()
        return self._summarizer

    @property
    def transcriber(self):
        if self._transcriber is None:
            with self._transcriber_lock:
                if self._transcriber is None:
                    from transcriber.whisper import WhisperTranscriber
                    self._transcriber = WhisperTranscriber()
        return self._transcriber

    def is_busy(self) -> bool:
        with self._processing_lock:
            return self._processing

    def set_processing(self, value: bool):
        with self._processing_lock:
            self._processing = value

    def process(self, url: str, task: Task) -> Optional[PipelineResult]:
        mem_start = _mem_mb()
        logger.info(f"[MEM] Pipeline start for {task.task_id} [mem: {mem_start:.0f}MB]")

        # Memory check: gc + malloc_trim + wait loop
        threshold = PipelineConfig.MEM_THRESHOLD_MB
        if mem_start > threshold:
            logger.warning(f"[MEM] Memory {mem_start:.0f}MB exceeds {threshold}MB threshold, waiting up to {PipelineConfig.MEM_WAIT_SECONDS}s...")
            waited = 0
            interval = PipelineConfig.MEM_WAIT_INTERVAL
            deadline = PipelineConfig.MEM_WAIT_SECONDS
            while waited < deadline:
                gc.collect()
                _malloc_trim()
                cur = _mem_mb()
                if cur <= threshold:
                    logger.info(f"[MEM] Memory recovered to {cur:.0f}MB after {waited}s wait")
                    break
                time.sleep(interval)
                waited += interval
            else:
                cur = _mem_mb()
                logger.warning(f"[MEM] Memory still {cur:.0f}MB after {deadline}s wait, proceeding anyway")
                gc.collect()
                _malloc_trim()

        self.set_processing(True)
        try:
            return self._do_process(url, task, mem_start)
        except Exception as e:
            mem_err = _mem_mb()
            logger.error(f"Pipeline failed for {task.task_id}: {e} [mem: {mem_err:.0f}MB, delta: {mem_err-mem_start:.0f}MB]", exc_info=True)
            self.task_manager.update_state(task, TaskState.FAILED, error=str(e))
            return None
        finally:
            self.set_processing(False)

    def _do_process(self, url: str, task: Task, mem_start: float) -> Optional[PipelineResult]:
        self._reset_if_failed(task)
        audio_cached = self.task_manager.has_cached_audio(task.task_id)
        downloader = get_downloader(task.platform)

        self._ensure_title(task, downloader, url, audio_cached)
        text = self._acquire_text(task, downloader, url, audio_cached)
        if not text:
            logger.error("No text content available for summarization")
            self.task_manager.update_state(task, TaskState.FAILED, error="No text content")
            return None

        markdown, model_name = self._summarize(task, text)
        self._finalize(task, markdown, model_name, mem_start)
        return PipelineResult(task_id=task.task_id, title=task.title, content=markdown, model_name=model_name)

    # ── 编排子函数 ──────────────────────────────────────────────────────────

    def _reset_if_failed(self, task: Task):
        if task.state == TaskState.FAILED:
            logger.info(f"Retrying previously failed task {task.task_id}")
            task.error = None
            self.task_manager.update_state(task, TaskState.PENDING)

    def _ensure_title(self, task: Task, downloader, url: str, audio_cached: bool):
        if task.title:
            return
        if audio_cached:
            task.title = self.task_manager.load_audio_meta(task.task_id) or "Unknown Video"
            return
        try:
            info = downloader.extract_info(url)
            task.title = info.get("title", "Unknown Video")
        except Exception as e:
            logger.warning(f"Failed to extract info: {e}")
            task.title = "Unknown Video"

    def _acquire_text(self, task: Task, downloader, url: str, audio_cached: bool) -> str:
        """取总结用文本：字幕优先，否则音频转写 + 纠错。"""
        subtitle = self._maybe_get_subtitle(task, downloader, url, audio_cached)
        if subtitle:
            return subtitle
        self._ensure_audio(task, downloader, url)
        transcript = self._transcribe(task)
        if not transcript:
            return ""
        return self._correct(task, url, transcript) or transcript

    def _maybe_get_subtitle(self, task: Task, downloader, url: str, audio_cached: bool) -> Optional[str]:
        if audio_cached:
            return None
        if self.task_manager.has_subtitle(task):
            return self.task_manager.load_subtitle(task)
        self.task_manager.update_state(task, TaskState.DOWNLOADING)
        try:
            subtitle = downloader.extract_subtitles(url)
            if subtitle:
                self.task_manager.save_subtitle(task, subtitle)
                logger.info(f"Subtitles extracted ({len(subtitle)} chars)")
            return subtitle
        except Exception as e:
            logger.warning(f"Subtitle extraction failed: {e}")
            return None

    def _ensure_audio(self, task: Task, downloader, url: str):
        if self.task_manager.has_audio(task):
            return
        self.task_manager.update_state(task, TaskState.DOWNLOADING)
        from config import DataConfig
        audio_dir = DataConfig.AUDIO_DIR / task.task_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        result = downloader.download_audio(url, str(audio_dir))
        task.title = result.title or task.title
        self.task_manager.save_audio_meta(task.task_id, task.title or "Unknown Video")
        logger.info(f"Audio downloaded: {result.file_path}")

    def _transcribe(self, task: Task) -> str:
        if self.task_manager.has_transcript(task):
            return self.task_manager.load_transcript(task)
        self.task_manager.update_state(task, TaskState.TRANSCRIBING)
        self._log_mem("Before transcription")
        audio_text = self.transcriber.transcript(str(task.audio_file))
        self.task_manager.save_transcript(task, audio_text)
        logger.info(f"Transcription done ({len(audio_text)} chars)")
        self._reclaim_mem("After transcription (GC+trim)")
        return audio_text

    def _correct(self, task: Task, url: str, transcript: str) -> Optional[str]:
        if self.task_manager.has_corrected(task):
            corrected = self.task_manager.load_corrected(task)
            model = self.task_manager.load_corrected_model(task)
            self._ensure_mp3_md(task, url, corrected, model)
            return corrected
        self.task_manager.update_state(task, TaskState.CORRECTING)
        self._reclaim_mem("Before correction")
        try:
            corrected = self.summarizer.correct(transcript)
            model = self.summarizer._last_model_name
            self.task_manager.save_corrected(task, corrected, model_name=model)
            self._write_mp3_md(task, url, corrected, model)
            logger.info(f"Corrected transcript ({len(corrected)} chars)")
            self._reclaim_mem("After correction (GC+trim)")
            return corrected
        except Exception as e:
            logger.warning(f"Transcript correction failed, falling back to raw transcript: {e}", exc_info=True)
            self.task_manager.update_state(task, task.state, correction_failed=True)
            return None

    def _write_mp3_md(self, task: Task, url: str, text: str, model):
        from bot.formatter import build_transcript_markdown
        md = build_transcript_markdown(task.title or "video", url, text, model_name=model)
        path = self.data_config.NOTES_DIR / f"{task.task_id}.mp3.md"
        path.write_text(md, encoding="utf-8")
        logger.info(f"Corrected transcript -> {path}")

    def _ensure_mp3_md(self, task: Task, url: str, text: str, model):
        path = self.data_config.NOTES_DIR / f"{task.task_id}.mp3.md"
        if not path.exists():
            self._write_mp3_md(task, url, text, model)

    def _summarize(self, task: Task, text: str):
        self.task_manager.update_state(task, TaskState.SUMMARIZING)
        self._log_mem("Before summarization")
        markdown = self.summarizer.summarize(task.title, text)
        model = self.summarizer._last_model_name
        self._log_mem("After summarization")
        return markdown, (model if isinstance(model, str) else None)

    def _finalize(self, task: Task, markdown: str, model_name, mem_start: float):
        task.model_name = model_name
        self.task_manager.save_note(task, markdown)
        self.task_manager.update_state(task, TaskState.COMPLETED)
        self.task_manager.cleanup_task_files(task)
        gc.collect()
        _malloc_trim()
        mem_end = _mem_mb()
        logger.info(f"[MEM] Pipeline completed for {task.task_id} [mem: {mem_end:.0f}MB, delta: {mem_end-mem_start:.0f}MB]")

    def _log_mem(self, label: str):
        logger.info(f"[MEM] {label} [mem: {_mem_mb():.0f}MB]")

    def _reclaim_mem(self, label: str):
        gc.collect()
        _malloc_trim()
        logger.info(f"[MEM] {label} [mem: {_mem_mb():.0f}MB]")
