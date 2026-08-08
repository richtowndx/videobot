"""uploader 测试：HTTP PUT 远端文件名按交付物 suffix 区分（总结 vs 纠错稿）。

回归背景：修复前 _clean_filename 无条件返回 {title}-summary.md，
导致纠错稿（.mp3.md）上传时与总结撞名，远端追加时间戳后缀。
"""
import asyncio
import os
from unittest import mock

import pytest

from uploaders.base import BaseUploader, UploadResult
from uploaders.http_put import HttpPutUploader, _clean_filename
from uploaders import UploaderManager


# ---------- _clean_filename：纯函数，按 suffix 生成远端文件名 ----------

def test_clean_filename_summary_default():
    # 默认 suffix -> -summary.md（旧行为保持不变）
    assert _clean_filename("闪客一小时") == "闪客一小时-summary.md"


def test_clean_filename_summary_explicit():
    assert _clean_filename("闪客一小时", suffix="_summary") == "闪客一小时-summary.md"


def test_clean_filename_mp3():
    # 纠错稿 suffix .mp3 -> .mp3.md（与本地 notes/{id}.mp3.md 命名一致）
    assert _clean_filename("闪客一小时", suffix=".mp3") == "闪客一小时.mp3.md"


def test_clean_filename_mp3_same_sanitization_as_summary():
    # 与 summary 走同样的清洗：空格/标点被去掉
    assert _clean_filename("Transformer AI入门06", suffix=".mp3") == "TransformerAI入门06.mp3.md"


def test_clean_filename_summary_same_sanitization_as_before():
    # 复现日志里的实际标题清洗结果（空间/标点剥离）
    title = "闪客一小时从函数到Transformer p0202计算神经网络的参数"
    assert _clean_filename(title) == "闪客一小时从函数到Transformerp0202计算神经网络的参数-summary.md"


def test_clean_filename_empty_title_summary():
    assert _clean_filename("") == "summary.md"


def test_clean_filename_empty_title_mp3():
    assert _clean_filename("", suffix=".mp3") == "transcript.mp3.md"


def test_clean_filename_unknown_suffix_falls_back_to_summary():
    # 未知 suffix 兜底为 summary 命名（不崩）
    assert _clean_filename("hello", suffix="something") == "hello-summary.md"


# ---------- HttpPutUploader.upload：suffix 贯穿到上传 URL ----------

def _fake_aiohttp_session(url_holder):
    """构造一个假的 aiohttp.ClientSession，捕获 PUT 的 URL。"""

    class _FakeResp:
        status = 200

        async def text(self):
            return '{"code":"ok","message":"saved"}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _FakeSession:
        def put(self, url, **kwargs):
            url_holder["url"] = url
            return _FakeResp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _FakeClientSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return _FakeSession()

        async def __aexit__(self, *exc):
            return False

    return _FakeClientSession


def test_httpput_upload_summary_suffix_in_url(tmp_path):
    fp = tmp_path / "note.md"
    fp.write_text("# summary", encoding="utf-8")
    url_holder = {}
    uploader = HttpPutUploader(url="https://example.com/", token="t")
    captured = _fake_aiohttp_session(url_holder)

    async def go():
        with mock.patch("uploaders.http_put.aiohttp.ClientSession", captured):
            return await uploader.upload(str(fp), "闪客一小时", suffix="_summary")

    result = asyncio.run(go())
    assert result.success
    assert url_holder["url"].endswith("/闪客一小时-summary.md")


def test_httpput_upload_mp3_suffix_in_url(tmp_path):
    """关键回归：纠错稿上传 URL 必须以 .mp3.md 结尾，而非 -summary.md。"""
    fp = tmp_path / "note.md"
    fp.write_text("# 转写稿", encoding="utf-8")
    url_holder = {}
    uploader = HttpPutUploader(url="https://example.com/", token="t")
    captured = _fake_aiohttp_session(url_holder)

    async def go():
        with mock.patch("uploaders.http_put.aiohttp.ClientSession", captured):
            return await uploader.upload(str(fp), "闪客一小时", suffix=".mp3")

    result = asyncio.run(go())
    assert result.success
    assert url_holder["url"].endswith("/闪客一小时.mp3.md")
    assert "-summary" not in url_holder["url"]


# ---------- UploaderManager.upload：suffix 透传给每个 uploader ----------

def test_uploader_manager_propagates_suffix(tmp_path):
    fp = tmp_path / "note.md"
    fp.write_text("body", encoding="utf-8")
    received = {}

    class _SpyUploader(BaseUploader):
        name = "spy"

        async def upload(self, file_path, title, suffix="_summary"):
            received["suffix"] = suffix
            received["title"] = title
            return UploadResult(uploader="spy", success=True, message="ok")

    mgr = UploaderManager.__new__(UploaderManager)  # 跳过 _build（不读配置）
    mgr._uploaders = [_SpyUploader()]

    async def go():
        return await mgr.upload(str(fp), "标题", suffix=".mp3")

    results = asyncio.run(go())
    assert received["suffix"] == ".mp3"
    assert len(results) == 1 and results[0].success


def test_uploader_manager_default_suffix_is_summary(tmp_path):
    """handler 以外若有调用不带 suffix，默认按 summary，保持向后兼容。"""
    fp = tmp_path / "note.md"
    fp.write_text("body", encoding="utf-8")
    received = {}

    class _SpyUploader(BaseUploader):
        name = "spy"

        async def upload(self, file_path, title, suffix="_summary"):
            received["suffix"] = suffix
            return UploadResult(uploader="spy", success=True, message="ok")

    mgr = UploaderManager.__new__(UploaderManager)
    mgr._uploaders = [_SpyUploader()]

    async def go():
        return await mgr.upload(str(fp), "标题")

    asyncio.run(go())
    assert received["suffix"] == "_summary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
