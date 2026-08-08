import json
import os
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from config import DataConfig


class TaskState(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    url: str
    platform: str
    state: TaskState = TaskState.PENDING
    title: Optional[str] = None
    error: Optional[str] = None
    model_name: Optional[str] = None
    correction_failed: bool = False

    @property
    def task_dir(self) -> Path:
        return DataConfig.TASKS_DIR / self.task_id

    @property
    def status_file(self) -> Path:
        return self.task_dir / "status.json"

    @property
    def audio_file(self) -> Path:
        for ext in ("mp3", "m4a", "wav", "webm"):
            p = self.task_dir / f"audio.{ext}"
            if p.exists():
                return p
        return self.task_dir / "audio.mp3"

    @property
    def transcript_file(self) -> Path:
        return self.task_dir / "transcript.json"

    @property
    def corrected_file(self) -> Path:
        return self.task_dir / "corrected.json"

    @property
    def subtitle_file(self) -> Path:
        return self.task_dir / "subtitle.txt"

    @property
    def note_file(self) -> Path:
        return DataConfig.NOTES_DIR / f"{self.task_id}.md"


class TaskManager:
    def get_task(self, task_id: str) -> Optional[Task]:
        status_file = DataConfig.TASKS_DIR / task_id / "status.json"
        if not status_file.exists():
            return None
        return self._load(status_file)

    def create_task(self, task_id: str, url: str, platform: str) -> Task:
        task = Task(task_id=task_id, url=url, platform=platform)
        task.task_dir.mkdir(parents=True, exist_ok=True)
        self._save(task)
        return task

    def get_or_create(self, url: str, platform: str) -> tuple[Task, bool]:
        from core.url_parser import url_to_task_id
        task_id = url_to_task_id(url)

        note_file = DataConfig.NOTES_DIR / f"{task_id}.md"
        if note_file.exists():
            task = Task(task_id=task_id, url=url, platform=platform, state=TaskState.COMPLETED)
            existing = self.get_task(task_id)
            if existing and existing.title:
                task.title = existing.title
            return task, True

        # Check existing task
        existing = self.get_task(task_id)
        if existing:
            return existing, True

        # Create new task
        return self.create_task(task_id, url, platform), False

    def update_state(self, task: Task, state: TaskState, **kwargs):
        task.state = state
        if "title" in kwargs:
            task.title = kwargs["title"]
        if "error" in kwargs:
            task.error = kwargs["error"]
        if "correction_failed" in kwargs:
            task.correction_failed = kwargs["correction_failed"]
        self._save(task)

    def get_cached_result(self, task_id: str) -> Optional[str]:
        note_file = DataConfig.NOTES_DIR / f"{task_id}.md"
        if note_file.exists():
            return note_file.read_text(encoding="utf-8")
        return None

    def has_audio(self, task: Task) -> bool:
        return task.audio_file.exists()

    def has_transcript(self, task: Task) -> bool:
        return task.transcript_file.exists()

    def has_subtitle(self, task: Task) -> bool:
        return task.subtitle_file.exists()

    def can_resume_from_transcription(self, task: Task) -> bool:
        return self.has_audio(task) or self.has_subtitle(task)

    def can_resume_from_summarization(self, task: Task) -> bool:
        return self.has_corrected(task) or self.has_subtitle(task)

    def save_transcript(self, task: Task, text: str):
        data = {"full_text": text}
        task.transcript_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_transcript(self, task: Task) -> Optional[str]:
        if not task.transcript_file.exists():
            return None
        data = json.loads(task.transcript_file.read_text(encoding="utf-8"))
        return data.get("full_text")

    def has_corrected(self, task: Task) -> bool:
        return task.corrected_file.exists()

    def save_corrected(self, task: Task, text: str, model_name: Optional[str] = None):
        data = {"full_text": text}
        if model_name:
            data["model_name"] = model_name
        task.corrected_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_corrected(self, task: Task) -> Optional[str]:
        if not task.corrected_file.exists():
            return None
        data = json.loads(task.corrected_file.read_text(encoding="utf-8"))
        return data.get("full_text")

    def load_corrected_model(self, task: Task) -> Optional[str]:
        """返回纠错所用模型名（旧版 corrected.json 无此字段则返回 None）。"""
        if not task.corrected_file.exists():
            return None
        data = json.loads(task.corrected_file.read_text(encoding="utf-8"))
        return data.get("model_name")

    def save_subtitle(self, task: Task, text: str):
        task.subtitle_file.write_text(text, encoding="utf-8")

    def load_subtitle(self, task: Task) -> Optional[str]:
        if not task.subtitle_file.exists():
            return None
        return task.subtitle_file.read_text(encoding="utf-8")

    def save_note(self, task: Task, content: str):
        note_path = DataConfig.NOTES_DIR / f"{task.task_id}.md"
        note_path.write_text(content, encoding="utf-8")

    def cleanup_task_files(self, task: Task):
        """Delete intermediate files (audio, video, transcript) but keep status."""
        import shutil
        if task.task_dir.exists():
            status = task.status_file
            status_data = status.read_text(encoding="utf-8") if status.exists() else None
            shutil.rmtree(task.task_dir, ignore_errors=True)
            if status_data:
                task.task_dir.mkdir(parents=True, exist_ok=True)
                status.write_text(status_data, encoding="utf-8")

    def _save(self, task: Task):
        task.task_dir.mkdir(parents=True, exist_ok=True)
        data = asdict(task)
        data["state"] = task.state.value
        task.status_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self, path: Path) -> Task:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["state"] = TaskState(data["state"])
        data.setdefault("model_name", None)
        data.setdefault("correction_failed", False)
        return Task(**data)
