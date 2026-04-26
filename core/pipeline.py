import gc
import logging
from dataclasses import dataclass

from config import DataConfig
from core.task_manager import TaskManager, Task, TaskState
from downloaders import get_downloader
from summarizer.llm import LLMSummarizer

logger = logging.getLogger(__name__)


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

    @property
    def summarizer(self):
        if self._summarizer is None:
            self._summarizer = LLMSummarizer()
        return self._summarizer

    @property
    def transcriber(self):
        if self._transcriber is None:
            from transcriber.whisper import WhisperTranscriber
            self._transcriber = WhisperTranscriber()
        return self._transcriber

    def process(self, url: str, task: Task) -> PipelineResult | None:
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
                    audio_path = str(task.audio_file)
                    audio_text = self.transcriber.transcript(audio_path)
                    self.task_manager.save_transcript(task, audio_text)
                    logger.info(f"Transcription done ({len(audio_text)} chars)")
                    gc.collect()
                else:
                    audio_text = self.task_manager.load_transcript(task)

            # Step 6: Summarize
            self.task_manager.update_state(task, TaskState.SUMMARIZING)
            text_content = subtitle_text or audio_text or ""
            if not text_content:
                logger.error("No text content available for summarization")
                self.task_manager.update_state(task, TaskState.FAILED, error="No text content")
                return None

            markdown = self.summarizer.summarize(task.title, text_content)

            # Step 7: Save and cleanup
            self.task_manager.save_note(task, markdown)
            self.task_manager.update_state(task, TaskState.COMPLETED)
            self.task_manager.cleanup_task_files(task)

            logger.info(f"Pipeline completed for {task.task_id}")
            return PipelineResult(task_id=task.task_id, title=task.title, markdown=markdown)

        except Exception as e:
            logger.error(f"Pipeline failed for {task.task_id}: {e}", exc_info=True)
            self.task_manager.update_state(task, TaskState.FAILED, error=str(e))
            return None
