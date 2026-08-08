import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from bot.formatter import build_markdown, save_temp_markdown, build_transcript_markdown, collect_deliverables


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


def test_save_temp_markdown_custom_suffix():
    path = save_temp_markdown("My Video", "# 内容", suffix=".mp3")
    assert path.endswith(".mp3.md")
    assert "My Video" in os.path.basename(path)
    os.remove(path)
    os.rmdir(os.path.dirname(path))


def test_build_transcript_markdown():
    result = build_transcript_markdown("测试视频", "https://example.com/v/1", "纠错后的正文")
    assert "# 测试视频" in result
    assert "https://example.com/v/1" in result
    assert "---" in result
    assert "纠错后的正文" in result


def test_collect_deliverables_with_and_without_mp3():
    import tempfile
    from pathlib import Path
    notes_dir = Path(tempfile.mkdtemp(prefix="test_notes_"))
    try:
        # 无 .mp3.md -> 仅总结
        items = collect_deliverables("T", "youtube", "https://u", "正文", None, "id1", notes_dir)
        assert len(items) == 1
        assert items[0][0] == "_summary"

        # 有 .mp3.md -> 两份
        (notes_dir / "id1.mp3.md").write_text("# T\n\n> 来源：https://u\n\n---\n\n正文", encoding="utf-8")
        items = collect_deliverables("T", "youtube", "https://u", "正文", None, "id1", notes_dir)
        assert len(items) == 2
        assert items[0][0] == "_summary"
        assert items[1][0] == ".mp3"
        assert "正文" in items[1][1]
    finally:
        import shutil
        shutil.rmtree(notes_dir, ignore_errors=True)


if __name__ == "__main__":
    test_build_markdown()
    test_build_markdown_with_special_chars()
    test_save_temp_markdown()
    test_save_temp_markdown_sanitizes_filename()
    test_save_temp_markdown_custom_suffix()
    test_build_transcript_markdown()
    test_collect_deliverables_with_and_without_mp3()
    print("All formatter tests passed!")
