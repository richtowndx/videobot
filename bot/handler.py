import asyncio
import logging
import os

from aiogram import types
from aiogram.filters import Command
from aiogram import Router

from config import BotConfig
from core.url_parser import extract_urls, parse_platform
from core.task_manager import TaskManager, TaskState
from core.pipeline import Pipeline
from bot.formatter import build_markdown, save_temp_markdown

logger = logging.getLogger(__name__)
router = Router()
task_manager = TaskManager()
pipeline = Pipeline()

MAX_RETRIES = 3

# Task queue: serializes pipeline processing so requests don't get rejected
_queue: asyncio.Queue | None = None
_worker_started = False


async def _ensure_worker():
    """Start the queue worker if not already running."""
    global _queue, _worker_started
    if _worker_started:
        return
    _worker_started = True
    _queue = asyncio.Queue()
    asyncio.create_task(_queue_worker())


async def _queue_worker():
    """Process queued requests sequentially."""
    while True:
        message, urls, status_msg = await _queue.get()
        try:
            total = len(urls)
            errors = []

            for i, url in enumerate(urls, start=1):
                success, error = await _process_single_url(message, status_msg, url, i, total)
                if not success:
                    errors.append(error)

            if not errors:
                await status_msg.edit_text(
                    f"Done! Processed {total} link(s) successfully." if total > 1 else "Done!"
                )
            elif len(errors) == total:
                await status_msg.edit_text("All links failed:\n" + "\n".join(errors))
            else:
                succeeded = total - len(errors)
                await status_msg.edit_text(
                    f"Processed {succeeded}/{total}. Failures:\n" + "\n".join(errors)
                )
        except Exception as e:
            logger.error(f"Queue worker error: {e}", exc_info=True)
            try:
                await status_msg.edit_text(f"Error: {e}")
            except Exception:
                pass
        finally:
            _queue.task_done()


def _is_authorized(message: types.Message) -> bool:
    return message.from_user.id == BotConfig.AUTH_USER_ID


@router.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Send me a video link (Bilibili or YouTube), "
        "and I'll generate a structured summary as a Markdown file."
    )


@router.message(Command("status"))
async def cmd_status(message: types.Message):
    if not _is_authorized(message):
        return
    tasks_dir = os.listdir(pipeline.data_config.TASKS_DIR) if pipeline.data_config.TASKS_DIR.exists() else []
    notes_count = len(list(pipeline.data_config.NOTES_DIR.glob("*.md"))) if pipeline.data_config.NOTES_DIR.exists() else 0
    q_size = _queue.qsize() if _queue else 0
    await message.answer(f"Queue: {q_size} pending\nActive tasks: {len(tasks_dir)}\nCompleted notes: {notes_count}")


async def _process_single_url(
    message: types.Message,
    status_msg: types.Message,
    url: str,
    index: int,
    total: int,
) -> tuple[bool, str]:
    """Process one URL with retry logic. Returns (success, error_message)."""
    platform = parse_platform(url)
    if not platform:
        return False, f"Unsupported platform: {url}"

    task, is_cached = task_manager.get_or_create(url, platform)

    if is_cached and task.state == TaskState.COMPLETED:
        cached = task_manager.get_cached_result(task.task_id)
        if cached:
            await _send_note(message, task.title or "video", task.platform, url, cached)
            return True, ""

    prefix = f"[{index}/{total}] " if total > 1 else ""
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            retry_hint = f" (attempt {attempt}/{MAX_RETRIES})" if attempt > 1 else ""
            await status_msg.edit_text(f"{prefix}Processing{retry_hint}...")

            result = await asyncio.get_event_loop().run_in_executor(
                None, pipeline.process, url, task
            )

            if result:
                await _send_note(message, result.title, platform, url, result.markdown)
                return True, ""
            else:
                last_error = task.error or "Pipeline returned no result"
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"{prefix}Attempt {attempt} failed for {url}: {last_error}, retrying..."
                    )
        except Exception as e:
            last_error = str(e)
            logger.error(
                f"{prefix}Attempt {attempt} error for {url}: {e}", exc_info=True
            )

    return False, f"{prefix}Failed after {MAX_RETRIES} attempts: {last_error[:300]}"


@router.message()
async def handle_message(message: types.Message):
    if not _is_authorized(message):
        await message.answer("Unauthorized. This bot is for private use only.")
        return

    await _ensure_worker()

    urls = extract_urls(message.text)
    if not urls:
        await message.answer("Please send a valid Bilibili or YouTube link.")
        return

    q_size = _queue.qsize()
    if q_size > 0:
        status_msg = await message.answer(
            f"Queued ({q_size} ahead). Will process {len(urls)} link(s) when ready..."
        )
    else:
        status_msg = await message.answer(
            f"Processing {len(urls)} link(s)..." if len(urls) > 1 else "Processing your video..."
        )

    await _queue.put((message, urls, status_msg))


async def _send_note(message: types.Message, title: str, platform: str, url: str, markdown: str):
    content = build_markdown(title, platform, url, markdown)
    file_path = save_temp_markdown(title, content)

    try:
        doc = types.FSInputFile(file_path, filename=f"{title[:50]}_summary.md")
        await message.answer_document(doc, caption=f"Summary: {title}", parse_mode=None)
    finally:
        try:
            os.remove(file_path)
            os.rmdir(os.path.dirname(file_path))
        except OSError:
            pass
