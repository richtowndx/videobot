import os
import tempfile
from datetime import datetime
from typing import Optional


def build_markdown(title: str, platform: str, url: str, summary: str, model_name: Optional[str] = None) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_line = f"\n> AI 模型：{model_name}" if model_name else ""
    header = f"""# {title}

> Platform: {platform}
> URL: {url}
> Generated: {generated}{model_line}

---
"""
    return header + summary


def save_temp_markdown(title: str, content: str, suffix: str = "_summary") -> str:
    """Save markdown to a temp file and return path. Caller should clean up."""
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
    safe_name = safe_name[:80] + f"{suffix}.md"
    temp_dir = tempfile.mkdtemp(prefix="videobot_md_")
    path = os.path.join(temp_dir, safe_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def build_transcript_markdown(title: str, url: str, text: str) -> str:
    """纠错转写稿交付物（.mp3.md）的内容：极简头 + 纠错正文。"""
    return f"# {title}\n\n> 来源：{url}\n\n---\n\n{text}"


def collect_deliverables(title, platform, url, summary, model_name=None, task_id=None, notes_dir=None):
    """返回待推送交付物列表 [(suffix, content), ...]。
    总结恒有；若 notes_dir/{task_id}.mp3.md 存在则追加纠错稿。"""
    from pathlib import Path
    items = [("_summary", build_markdown(title, platform, url, summary, model_name))]
    if task_id and notes_dir:
        mp3 = Path(notes_dir) / f"{task_id}.mp3.md"
        if mp3.exists():
            items.append((".mp3", mp3.read_text(encoding="utf-8")))
    return items
