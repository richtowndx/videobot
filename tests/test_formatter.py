import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from bot.formatter import build_markdown, save_temp_markdown


def test_build_markdown():
    result = build_markdown("Test Video", "youtube", "https://youtube.com/watch?v=abc", "## Summary\n\nContent here")
    assert "# Test Video" in result
    assert "youtube" in result
    assert "https://youtube.com/watch?v=abc" in result
    assert "## Summary" in result
    assert "---" in result


def test_build_markdown_with_special_chars():
    result = build_markdown("Test <Video>", "bilibili", "https://bilibili.com", "Content")
    assert "# Test <Video>" in result
    assert "bilibili" in result


def test_save_temp_markdown():
    path = save_temp_markdown("My Video Title", "# Summary\n\nContent")
    assert os.path.exists(path)
    assert path.endswith("_summary.md")
    assert "My Video Title" in os.path.basename(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "# Summary" in content

    # Cleanup
    os.remove(path)
    os.rmdir(os.path.dirname(path))


def test_save_temp_markdown_sanitizes_filename():
    path = save_temp_markdown("Video: A/B/C?D=E&F", "# Summary")
    basename = os.path.basename(path)
    assert "/" not in basename
    assert ":" not in basename

    os.remove(path)
    os.rmdir(os.path.dirname(path))


if __name__ == "__main__":
    test_build_markdown()
    test_build_markdown_with_special_chars()
    test_save_temp_markdown()
    test_save_temp_markdown_sanitizes_filename()
    print("All formatter tests passed!")
