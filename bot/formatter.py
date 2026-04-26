import os
import tempfile
from datetime import datetime


def build_markdown(title: str, platform: str, url: str, summary: str) -> str:
    header = f"""# {title}

> Platform: {platform}
> URL: {url}
> Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
    return header + summary


def save_temp_markdown(title: str, content: str) -> str:
    """Save markdown to a temp file and return path. Caller should clean up."""
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
    safe_name = safe_name[:80] + "_summary.md"
    temp_dir = tempfile.mkdtemp(prefix="videobot_md_")
    path = os.path.join(temp_dir, safe_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
