import gc
import logging
import resource
import threading
from dataclasses import dataclass

from config import DataConfig
from core.task_manager import TaskManager, Task, TaskState
from downloaders import get_downloader
from summarizer.llm import LLMSummarizer

logger = logging.getLogger(__name__)


def _mem_mb():
    """Get current RSS memory in MB (not peak)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@dataclass
class PipelineResult:
    task_id: str
    title: str
    markdown: str


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
                # Double-check after acquiring lock
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

    def process(self, url: str, task: Task) -> PipelineResult | None:
        mem_start = _mem_mb()
        logger.info(f"[MEM] Pipeline start for {task.task_id} [mem: {mem_start:.0f}MB]")

        # Memory check
        if mem_start > 1024:
            logger.warning(f"[MEM] Memory {mem_start:.0f}MB exceeds 1024MB threshold, rejecting task")
            self.task_manager.update_state(task, TaskState.FAILED, error="Server memory too high, try later")
            return None

        self.set_processing(True)
        try:
            # Step 1: Check if already completed
            cached = self.task_manager.get_cached_result(task.task_id)
            if cached:
                logger.info(f"Returning cached result for {task.task_id}")
                return PipelineResult(task_id=task.task_id, title=task.title or "video", markdown=cached)

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

                # Step 5: Transcribe audio
                if not self.task_manager.has_transcript(task):
                    self.task_manager.update_state(task, TaskState.TRANSCRIBING)
                    gc.collect()
                    mem_before_transcribe = _mem_mb()
                    logger.info(f"[MEM] Before transcription [mem: {mem_before_transcribe:.0f}MB]")
                    audio_path = str(task.audio_file)
                    audio_text = self.transcriber.transcript(audio_path)
                    mem_after_transcribe = _mem_mb()
                    logger.info(f"[MEM] After transcription [mem: {mem_after_transcribe:.0f}MB, delta: {mem_after_transcribe-mem_before_transcribe:.0f}MB]")
                    self.task_manager.save_transcript(task, audio_text)
                    logger.info(f"Transcription done ({len(audio_text)} chars)")
                    gc.collect()
                    mem_after_gc = _mem_mb()
                    logger.info(f"[MEM] After GC [mem: {mem_after_gc:.0f}MB]")
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
            if self.summarizer._last_model_name:
                markdown = f"*AI 模型：{self.summarizer._last_model_name}*\n\n---\n\n" + markdown
            mem_after_summary = _mem_mb()
            logger.info(f"[MEM] After summarization [mem: {mem_after_summary:.0f}MB, delta: {mem_after_summary-mem_before_summary:.0f}MB]")

            # Step 7: Save and cleanup
            self.task_manager.save_note(task, markdown)
            self.task_manager.update_state(task, TaskState.COMPLETED)
            mem_before_cleanup = _mem_mb()
            logger.info(f"[MEM] Before cleanup [mem: {mem_before_cleanup:.0f}MB]")
            self.task_manager.cleanup_task_files(task)

            # Unload Whisper model to free memory between tasks
            if self._transcriber is not None:
                del self._transcriber
                self._transcriber = None
                logger.info("Whisper model unloaded")
            gc.collect()

            mem_after_cleanup = _mem_mb()
            logger.info(f"[MEM] After cleanup [mem: {mem_after_cleanup:.0f}MB, released: {mem_before_cleanup-mem_after_cleanup:.0f}MB]")

            mem_end = _mem_mb()
            logger.info(f"[MEM] Pipeline completed for {task.task_id} [mem: {mem_end:.0f}MB, delta: {mem_end-mem_start:.0f}MB]")
            return PipelineResult(task_id=task.task_id, title=task.title, markdown=markdown)

        except Exception as e:
            mem_err = _mem_mb()
            logger.error(f"Pipeline failed for {task.task_id}: {e} [mem: {mem_err:.0f}MB, delta: {mem_err-mem_start:.0f}MB]", exc_info=True)
            self.task_manager.update_state(task, TaskState.FAILED, error=str(e))
            return None
        finally:
            self.set_processing(False)
