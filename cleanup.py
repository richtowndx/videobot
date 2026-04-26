"""Background cleanup of stale data files.

Two sweep types run on a configurable schedule:

* **Notes** – ``data/notes/*.md`` files whose mtime is older than
  ``NOTE_EXPIRE_DAYS`` days are deleted.
* **Tasks** – ``data/tasks/{video_id}/`` directories whose newest-file
  mtime is older than ``TASK_EXPIRE_DAYS`` days are removed entirely.

``start_cleanup_thread()`` launches a daemon thread that loops forever,
sleeping ``CLEANUP_INTERVAL`` seconds between sweeps.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

import config

logger = logging.getLogger(__name__)

#: How often (seconds) the cleanup loop runs.
CLEANUP_INTERVAL: int = int(config.__dict__.get("CLEANUP_INTERVAL", 3600))


# ── Individual sweeps ─────────────────────────────────────────────────────────

def cleanup_notes() -> int:
    """Delete expired note files.  Returns the number of files removed."""
    notes_dir = config.DATA_DIR / "notes"
    if not notes_dir.exists():
        return 0

    cutoff = time.time() - config.NOTE_EXPIRE_DAYS * 86400
    removed = 0
    for note_file in notes_dir.glob("*.md"):
        try:
            if note_file.stat().st_mtime < cutoff:
                note_file.unlink()
                removed += 1
                logger.info("deleted expired note: %s", note_file.name)
        except OSError as exc:
            logger.warning("could not delete note %s: %s", note_file, exc)
    return removed


def cleanup_tasks() -> int:
    """Delete expired task directories.  Returns the number removed."""
    tasks_dir = config.DATA_DIR / "tasks"
    if not tasks_dir.exists():
        return 0

    cutoff = time.time() - config.TASK_EXPIRE_DAYS * 86400
    removed = 0
    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir():
            continue
        try:
            mtimes = [f.stat().st_mtime for f in task_dir.rglob("*") if f.is_file()]
            newest = max(mtimes) if mtimes else 0.0
            if newest < cutoff:
                shutil.rmtree(task_dir, ignore_errors=True)
                removed += 1
                logger.info("deleted expired task dir: %s", task_dir.name)
        except OSError as exc:
            logger.warning("could not clean task dir %s: %s", task_dir, exc)
    return removed


def run_cleanup() -> None:
    """Run both sweeps once."""
    n_notes = cleanup_notes()
    n_tasks = cleanup_tasks()
    if n_notes or n_tasks:
        logger.info("cleanup: removed %d note(s) and %d task dir(s)", n_notes, n_tasks)


# ── Background thread ─────────────────────────────────────────────────────────

def _loop() -> None:
    while True:
        try:
            run_cleanup()
        except Exception as exc:
            logger.error("cleanup loop error: %s", exc, exc_info=True)
        time.sleep(CLEANUP_INTERVAL)


def start_cleanup_thread() -> threading.Thread:
    """Start (and return) the background cleanup daemon thread."""
    t = threading.Thread(target=_loop, daemon=True, name="cleanup-thread")
    t.start()
    logger.info("background cleanup thread started (interval=%ds)", CLEANUP_INTERVAL)
    return t
