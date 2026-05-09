import os
import sys
import tomllib
import platform
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

BASE_DIR = Path(__file__).parent

_cfg_path = BASE_DIR / "config.toml"
with open(_cfg_path, "rb") as _f:
    _cfg = tomllib.load(_f)
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
    _bot = _cfg.get("bot", {})
    TOKEN = _bot.get("token", "")
    AUTH_USER_ID = int(_bot.get("auth_user_id", 0))
    PROXY = _bot.get("proxy", "") or None


class DownloaderConfig:
    _dl = _cfg.get("downloader", {})
    BILIBILI_COOKIE = _dl.get("bilibili_cookie", "") or None
    YOUTUBE_COOKIE = _dl.get("youtube_cookie", "") or None
    YT_PROXY = _dl.get("yt_proxy", "") or None


@dataclass
class ModelConfig:
    name: str
    url: str
    key: str
    max_context_tokens: int = 64000


class AIConfig:
    _ai = _cfg.get("ai", {})
    MAX_CONTEXT_TOKENS = int(_ai.get("max_context_tokens", 64000))

    @staticmethod
    def load_models() -> list[ModelConfig]:
        """Load model list from config.toml [ai] section."""
        ai = _cfg.get("ai", {})
        models_cfg = ai.get("models", [])
        if models_cfg:
            models = []
            for item in models_cfg:
                models.append(ModelConfig(
                    name=item["name"],
                    url=item["url"],
                    key=item["key"],
                    max_context_tokens=item.get("max_context_tokens", AIConfig.MAX_CONTEXT_TOKENS),
                ))
            logger.info(f"Loaded {len(models)} AI model(s): {', '.join(m.name for m in models)}")
            return models

        # Fallback to single-model config
        name = ai.get("model_name", "qwen-plus")
        url = ai.get("api_url", "")
        key = ai.get("api_key", "")
        if name and url and key:
            logger.info(f"Using single AI model: {name}")
            return [ModelConfig(name=name, url=url, key=key, max_context_tokens=AIConfig.MAX_CONTEXT_TOKENS)]

        raise ValueError("No AI models configured. Set [ai.models] or ai.model_name/api_url/api_key in config.toml.")


class WhisperConfig:
    _whisper = _cfg.get("whisper", {})
    MODEL_SIZE = _whisper.get("model_size", "tiny")
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"


class DataConfig:
    DATA_DIR = Path(_cfg.get("data", {}).get("dir", "./data"))
    TASKS_DIR = DATA_DIR / "tasks"
    NOTES_DIR = DATA_DIR / "notes"
    MODELS_DIR = BASE_DIR / "models"

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Auto-ensure deno on import
_ensure_deno()
