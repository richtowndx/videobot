import os
import time
import logging
import shutil
from pathlib import Path

import asyncio

from config import DataConfig

logger = logging.getLogger(__name__)

NOTES_MAX_AGE_DAYS = 30
TASKS_MAX_AGE_DAYS = 1


async def cleanup_scheduler():
    """Background task that periodically cleans up old files."""
    while True:
        try:
            await _cleanup_notes()
            await _cleanup_stale_tasks()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        await asyncio.sleep(3600)  # Run every hour


async def _cleanup_notes():
    """Delete notes older than NOTES_MAX_AGE_DAYS."""
    notes_dir = DataConfig.NOTES_DIR
    if not notes_dir.exists():
        return

    cutoff = time.time() - (NOTES_MAX_AGE_DAYS * 86400)
    count = 0

    for f in notes_dir.glob("*.md"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            count += 1

    if count:
        logger.info(f"Cleaned up {count} old notes (> {NOTES_MAX_AGE_DAYS} days)")


async def _cleanup_stale_tasks():
    """Delete task directories older than TASKS_MAX_AGE_DAYS."""
    tasks_dir = DataConfig.TASKS_DIR
    if not tasks_dir.exists():
        return

    cutoff = time.time() - (TASKS_MAX_AGE_DAYS * 86400)
    count = 0

    for d in tasks_dir.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            count += 1

    if count:
        logger.info(f"Cleaned up {count} stale task dirs (> {TASKS_MAX_AGE_DAYS} day)")


def cleanup_sync():
    """Synchronous cleanup for testing."""
    import asyncio
    asyncio.get_event_loop().run_until_complete(_cleanup_notes())
    asyncio.get_event_loop().run_until_complete(_cleanup_stale_tasks())
