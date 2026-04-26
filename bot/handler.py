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
    await message.answer(f"Active tasks: {len(tasks_dir)}\nCompleted notes: {notes_count}")


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

    # Check if pipeline is busy processing another task
    if pipeline.is_busy():
        await message.answer("Server is busy processing another request. Please try again later.")
        return

    urls = extract_urls(message.text)
    if not urls:
        await message.answer("Please send a valid Bilibili or YouTube link.")
        return

    total = len(urls)
    status_msg = await message.answer(
        f"Found {total} link(s). Processing..." if total > 1 else "Processing your video..."
    )

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


async def _send_note(message: types.Message, title: str, platform: str, url: str, markdown: str):
    content = build_markdown(title, platform, url, markdown)
    file_path = save_temp_markdown(title, content)

    try:
        doc = types.FSInputFile(file_path, filename=f"{title[:50]}_summary.md")
        await message.answer_document(doc, caption=f"Summary: {title}")
    finally:
        try:
            os.remove(file_path)
            os.rmdir(os.path.dirname(file_path))
        except OSError:
            pass
