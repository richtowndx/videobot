"""
Integration test for the full VideoBot pipeline.

Tests the complete flow:
  URL received -> parse platform -> download -> transcribe -> summarize -> save note
  + cached result verification on second run

Usage:
  cd /data/code/node/BiliNote/videobot
  python -m pytest tests/test_integration.py -v -s
  # or
  python tests/test_integration.py
"""

import sys
import time
import logging
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.url_parser import extract_url, parse_platform, url_to_task_id
from core.task_manager import TaskManager, TaskState
from core.pipeline import Pipeline, PipelineResult
from bot.formatter import build_markdown, save_temp_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("integration_test")

# ── Test URLs ────────────────────────────────────────────────────────────────
BILIBILI_URL = (
    "https://www.bilibili.com/video/BV1aDD3B1Ea4/"
    "?spm_id_from=333.1007.tianma.3-4-10.click"
    "&vd_source=12e5345ff662e428baa991d1d4be34a3"
)
YOUTUBE_URL = "https://www.youtube.com/watch?v=edHNTFt5jYk"

TEST_CASES = [
    {"name": "Bilibili", "url": BILIBILI_URL, "platform": "bilibili"},
    {"name": "YouTube", "url": YOUTUBE_URL, "platform": "youtube"},
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def simulate_bot_receive(raw_text: str):
    """Simulate the bot receiving a message with a URL."""
    url = extract_url(raw_text)
    assert url, f"No URL extracted from: {raw_text}"
    platform = parse_platform(url)
    assert platform, f"Unsupported platform for URL: {url}"
    task_id = url_to_task_id(url)
    logger.info(f"Bot received URL: {url}  platform={platform}  task_id={task_id}")
    return url, platform, task_id


def run_pipeline_once(url: str, platform: str) -> PipelineResult:
    """Create task and run the full pipeline once."""
    pipeline = Pipeline()
    task_manager = TaskManager()

    task, is_cached = task_manager.get_or_create(url, platform)
    logger.info(
        f"Task {task.task_id}: state={task.state} is_cached={is_cached}"
    )

    result = pipeline.process(url, task)
    assert result is not None, f"Pipeline returned None for {url}"
    assert result.markdown, f"Pipeline returned empty markdown for {url}"
    assert result.title, f"Pipeline returned empty title for {url}"

    logger.info(f"Pipeline result: title='{result.title}' markdown_len={len(result.markdown)}")
    return result


def simulate_bot_send_note(title: str, platform: str, url: str, markdown: str) -> Path:
    """Simulate the bot formatting and saving the note file for sending."""
    content = build_markdown(title, platform, url, markdown)
    file_path = save_temp_markdown(title, content)

    saved = Path(file_path)
    assert saved.exists(), f"Note file not created: {file_path}"
    content_text = saved.read_text(encoding="utf-8")
    assert title in content_text, f"Title not in note file"
    assert platform in content_text, f"Platform not in note file"
    assert url in content_text, f"URL not in note file"

    logger.info(f"Note file created: {file_path} ({len(content_text)} chars)")
    return saved


def verify_cached_result(url: str, platform: str, original_result: PipelineResult):
    """Verify that re-processing the same URL returns the cached result."""
    task_manager = TaskManager()

    task, is_cached = task_manager.get_or_create(url, platform)
    assert is_cached, "Second run should detect cached task"
    assert task.state == TaskState.COMPLETED, (
        f"Expected COMPLETED, got {task.state}"
    )

    cached = task_manager.get_cached_result(task.task_id)
    assert cached, "Cached result should exist"
    assert cached == original_result.markdown, (
        "Cached markdown should match original"
    )
    logger.info(f"Cache verified for task {task.task_id}")


# ── Test Cases ───────────────────────────────────────────────────────────────

def test_bilibili_full_pipeline():
    """Full integration test for Bilibili video processing."""
    tc = TEST_CASES[0]
    logger.info(f"=== Starting {tc['name']} full pipeline test ===")

    url, platform, task_id = simulate_bot_receive(tc["url"])
    assert platform == "bilibili"

    result = run_pipeline_once(url, platform)
    note_path = simulate_bot_send_note(result.title, platform, url, result.markdown)

    logger.info(f"{tc['name']} pipeline passed. Note: {note_path}")

    # Cleanup temp file
    try:
        note_path.unlink()
        note_path.parent.rmdir()
    except OSError:
        pass

    return result


def test_youtube_full_pipeline():
    """Full integration test for YouTube video processing."""
    tc = TEST_CASES[1]
    logger.info(f"=== Starting {tc['name']} full pipeline test ===")

    url, platform, task_id = simulate_bot_receive(tc["url"])
    assert platform == "youtube"

    result = run_pipeline_once(url, platform)
    note_path = simulate_bot_send_note(result.title, platform, url, result.markdown)

    logger.info(f"{tc['name']} pipeline passed. Note: {note_path}")

    try:
        note_path.unlink()
        note_path.parent.rmdir()
    except OSError:
        pass

    return result


def test_bilibili_cached_result():
    """Verify Bilibili re-processing uses cached result."""
    tc = TEST_CASES[0]
    logger.info(f"=== Starting {tc['name']} cache verification ===")

    url, platform, _ = simulate_bot_receive(tc["url"])
    result = run_pipeline_once(url, platform)
    verify_cached_result(url, platform, result)

    logger.info(f"{tc['name']} cache test passed")


def test_youtube_cached_result():
    """Verify YouTube re-processing uses cached result."""
    tc = TEST_CASES[1]
    logger.info(f"=== Starting {tc['name']} cache verification ===")

    url, platform, _ = simulate_bot_receive(tc["url"])
    result = run_pipeline_once(url, platform)
    verify_cached_result(url, platform, result)

    logger.info(f"{tc['name']} cache test passed")


# ── Main Runner ──────────────────────────────────────────────────────────────

def main():
    results = {}

    # Phase 1: Full pipeline for both platforms
    logger.info("=" * 60)
    logger.info("Phase 1: Full pipeline tests (download -> transcribe -> summarize)")
    logger.info("=" * 60)

    for tc in TEST_CASES:
        logger.info("")
        logger.info(f"--- Processing {tc['name']}: {tc['url'][:80]}... ---")
        start = time.time()
        try:
            url, platform, _ = simulate_bot_receive(tc["url"])
            result = run_pipeline_once(url, platform)
            note_path = simulate_bot_send_note(
                result.title, platform, url, result.markdown
            )
            elapsed = time.time() - start
            results[tc["name"]] = {
                "status": "PASS",
                "title": result.title,
                "note_path": str(note_path),
                "markdown_len": len(result.markdown),
                "elapsed": elapsed,
            }
            logger.info(
                f"  {tc['name']} PASS  title='{result.title}'  "
                f"len={len(result.markdown)}  time={elapsed:.1f}s"
            )
        except Exception as e:
            elapsed = time.time() - start
            results[tc["name"]] = {"status": "FAIL", "error": str(e), "elapsed": elapsed}
            logger.error(f"  {tc['name']} FAIL: {e}")

    # Phase 2: Cached result verification
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: Cache verification (re-submit same URLs)")
    logger.info("=" * 60)

    for tc in TEST_CASES:
        if results.get(tc["name"], {}).get("status") != "PASS":
            logger.warning(f"Skipping cache test for {tc['name']} (pipeline failed)")
            continue

        logger.info("")
        logger.info(f"--- Cache test for {tc['name']} ---")
        start = time.time()
        try:
            url, platform, _ = simulate_bot_receive(tc["url"])
            result = run_pipeline_once(url, platform)
            verify_cached_result(url, platform, result)
            elapsed = time.time() - start
            results[f"{tc['name']}_cache"] = {
                "status": "PASS",
                "elapsed": elapsed,
            }
            logger.info(f"  {tc['name']} cache PASS  time={elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start
            results[f"{tc['name']}_cache"] = {
                "status": "FAIL",
                "error": str(e),
                "elapsed": elapsed,
            }
            logger.error(f"  {tc['name']} cache FAIL: {e}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    all_pass = True
    for name, info in results.items():
        status = info["status"]
        elapsed = info.get("elapsed", 0)
        if status == "PASS":
            extra = ""
            if "title" in info:
                extra = f"  title='{info['title']}'  len={info['markdown_len']}"
            logger.info(f"  [PASS] {name}  ({elapsed:.1f}s){extra}")
        else:
            all_pass = False
            logger.error(f"  [FAIL] {name}  ({elapsed:.1f}s)  {info.get('error', '')}")

    logger.info("")
    if all_pass:
        logger.info("ALL TESTS PASSED")
    else:
        logger.error("SOME TESTS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
