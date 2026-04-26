import os
import sys
import platform
import logging
from pathlib import Path
from urllib.request import urlretrieve

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)


def _ensure_deno():
    """Auto-detect or download deno for yt-dlp EJS challenge solving."""
    # Already in PATH?
    from shutil import which
    if which("deno"):
        return

    # Check common install locations
    candidates = [
        Path.home() / ".deno" / "bin" / "deno",
        Path.home() / ".local" / "bin" / "deno",
        BASE_DIR / "bin" / "deno",
    ]
    for p in candidates:
        if p.exists():
            _add_to_path(p.parent)
            return

    # Download deno to project bin/
    deno_dir = BASE_DIR / "bin"
    deno_path = deno_dir / "deno"
    if deno_path.exists():
        _add_to_path(deno_dir)
        return

    logger.info("Downloading deno runtime for yt-dlp...")
    deno_dir.mkdir(parents=True, exist_ok=True)

    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        arch = machine

    if sys_name == "linux":
        suffix = "zip"
        file_tag = f"deno-{arch}-unknown-linux-gnu"
    elif sys_name == "darwin":
        suffix = "zip"
        file_tag = f"deno-{arch}-apple-darwin"
    elif sys_name == "windows":
        suffix = "zip"
        file_tag = f"deno-{arch}-pc-windows-msvc"
    else:
        logger.warning(f"Unsupported OS: {sys_name}, skipping deno install")
        return

    url = f"https://github.com/denoland/deno/releases/latest/download/{file_tag}.{suffix}"

    try:
        import tempfile, zipfile
        tmp = tempfile.mkdtemp(prefix="deno_dl_")
        archive = os.path.join(tmp, f"deno.{suffix}")
        urlretrieve(url, archive)

        with zipfile.ZipFile(archive, "r") as zf:
            # Find deno binary in the archive
            for name in zf.namelist():
                if name.endswith("deno") or name == "deno.exe":
                    with zf.open(name) as src, open(deno_path, "wb") as dst:
                        dst.write(src.read())
                    break

        os.chmod(deno_path, 0o755)

        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

        _add_to_path(deno_dir)
        logger.info(f"Deno installed: {deno_path}")
    except Exception as e:
        logger.warning(f"Failed to install deno: {e}. YouTube downloads may require it.")


def _add_to_path(directory: Path):
    p = str(directory)
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{p}:{os.environ.get('PATH', '')}"


class BotConfig:
    TOKEN = os.environ.get("BOT_TOKEN", "")
    AUTH_USER_ID = int(os.environ.get("AUTH_USER_ID", "0"))
    PROXY = os.environ.get("BOT_PROXY", "") or None


class DownloaderConfig:
    BILIBILI_COOKIE = os.environ.get("BILIBILI_COOKIE", "") or None
    YOUTUBE_COOKIE = os.environ.get("YOUTUBE_COOKIE", "") or None
    YT_PROXY = os.environ.get("YT_PROXY", "") or None


class AIConfig:
    API_KEY = os.environ.get("AI_API_KEY", "")
    MODEL_NAME = os.environ.get("AI_MODEL_NAME", "qwen-plus")
    API_URL = os.environ.get("AI_API_URL", "")


class WhisperConfig:
    MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"


class DataConfig:
    DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
    TASKS_DIR = DATA_DIR / "tasks"
    NOTES_DIR = DATA_DIR / "notes"
    MODELS_DIR = BASE_DIR / "models"

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Auto-ensure deno on import
_ensure_deno()
