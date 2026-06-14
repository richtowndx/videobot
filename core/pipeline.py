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
        # Step 1: Check if already completed
        cached = self.task_manager.get_cached_result(task.task_id)
        if cached:
            logger.info(f"Returning cached result for {task.task_id}")
            return PipelineResult(task_id=task.task_id, title=task.title or "video", content=cached)

        # Reset failed task so pipeline can retry from last successful step
        if task.state == TaskState.FAILED:
            logger.info(f"Retrying previously failed task {task.task_id}")
            task.error = None
            self.task_manager.update_state(task, TaskState.PENDING)

        downloader = get_downloader(task.platform)

        # Step 2: Extract video info for title
        try:
            info = downloader.extract_info(url)
            task.title = info.get("title", "Unknown Video")
        except Exception as e:
            logger.warning(f"Failed to extract info: {e}")
            task.title = "Unknown Video"

        # Step 3: Try subtitle extraction (fast path)
        subtitle_text = None
        if not self.task_manager.has_subtitle(task):
            self.task_manager.update_state(task, TaskState.DOWNLOADING)
            try:
                subtitle_text = downloader.extract_subtitles(url)
                if subtitle_text:
                    self.task_manager.save_subtitle(task, subtitle_text)
                    logger.info(f"Subtitles extracted ({len(subtitle_text)} chars)")
            except Exception as e:
                logger.warning(f"Subtitle extraction failed: {e}")
        else:
            subtitle_text = self.task_manager.load_subtitle(task)

        # Step 4: Download audio if no subtitles
        audio_text = None
        if not subtitle_text:
            if not self.task_manager.has_audio(task):
                self.task_manager.update_state(task, TaskState.DOWNLOADING)
                result = downloader.download_audio(url, str(task.task_dir))
                task.title = result.title or task.title
                logger.info(f"Audio downloaded: {result.file_path}")

            # Step 5: Transcribe audio (Whisper model stays loaded)
            if not self.task_manager.has_transcript(task):
                self.task_manager.update_state(task, TaskState.TRANSCRIBING)
                gc.collect()
                mem_before = _mem_mb()
                logger.info(f"[MEM] Before transcription [mem: {mem_before:.0f}MB]")
                audio_path = str(task.audio_file)
                audio_text = self.transcriber.transcript(audio_path)
                mem_after = _mem_mb()
                logger.info(f"[MEM] After transcription [mem: {mem_after:.0f}MB, delta: {mem_after-mem_before:.0f}MB]")
                self.task_manager.save_transcript(task, audio_text)
                logger.info(f"Transcription done ({len(audio_text)} chars)")
                # Free transcription intermediates
                del audio_text
                audio_text = self.task_manager.load_transcript(task)
                gc.collect()
                _malloc_trim()
                logger.info(f"[MEM] After GC+trim [mem: {_mem_mb():.0f}MB]")
            else:
                audio_text = self.task_manager.load_transcript(task)

        # Step 6: Summarize
        self.task_manager.update_state(task, TaskState.SUMMARIZING)
        mem_before_summary = _mem_mb()
        logger.info(f"[MEM] Before summarization [mem: {mem_before_summary:.0f}MB]")
        text_content = subtitle_text or audio_text or ""
        if not text_content:
            logger.error("No text content available for summarization")
            self.task_manager.update_state(task, TaskState.FAILED, error="No text content")
            return None

        markdown = self.summarizer.summarize(task.title, text_content)
        # Free source text immediately
        del text_content, subtitle_text, audio_text
        mem_after_summary = _mem_mb()
        logger.info(f"[MEM] After summarization [mem: {mem_after_summary:.0f}MB, delta: {mem_after_summary-mem_before_summary:.0f}MB]")

        # Step 7: Save and cleanup task files
        model_name = self.summarizer._last_model_name
        task.model_name = model_name if isinstance(model_name, str) else None
        self.task_manager.save_note(task, markdown)
        self.task_manager.update_state(task, TaskState.COMPLETED)
        self.task_manager.cleanup_task_files(task)

        # Aggressive cleanup between tasks
        gc.collect()
        _malloc_trim()

        mem_end = _mem_mb()
        logger.info(f"[MEM] Pipeline completed for {task.task_id} [mem: {mem_end:.0f}MB, delta: {mem_end-mem_start:.0f}MB]")
        return PipelineResult(task_id=task.task_id, title=task.title, content=markdown, model_name=self.summarizer._last_model_name)
