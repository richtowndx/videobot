"""Persistent note cache stored in ``data/notes/``.

A cached note is keyed by *video_id* and stored as ``{video_id}.md``.
Any note whose file modification-time is older than ``NOTE_EXPIRE_DAYS``
is treated as expired and automatically removed on access.
"""

from __future__ import annotations

import time
from pathlib import Path

import config


def _notes_dir() -> Path:
    d = config.DATA_DIR / "notes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def note_path(video_id: str) -> Path:
    return _notes_dir() / f"{video_id}.md"


def get_cached_note(video_id: str) -> str | None:
    """Return the cached note for *video_id*, or ``None`` if absent / expired."""
    path = note_path(video_id)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > config.NOTE_EXPIRE_DAYS * 86400:
        path.unlink(missing_ok=True)
        return None
    return path.read_text(encoding="utf-8")


def save_note(video_id: str, note: str) -> None:
    """Persist *note* to disk for future cache hits."""
    note_path(video_id).write_text(note, encoding="utf-8")
