# 音频缓存与从音频重跑 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让处理过的视频保留 audio.mp3（7 天 TTL），第二个请求从本地音频重跑「转写→纠错→总结→上报」而不再联网下载；同时把高圈复杂度的 `_do_process` 重构为薄编排器 + 职责单一的子函数。

**架构：** 音频从 task 目录搬到独立的 `data/audio/{task_id}/`（带 `meta.json` 存标题），与 task 目录清理（1 天）解耦。新增 `TaskManager.get_resume_action()` 三态决策（reprocess / return_cached / full）驱动入口；pipeline 在音频缓存命中时跳过一切联网步骤；新增 `_cleanup_old_audio()` 挂到现有 `cleanup_scheduler`。

**技术栈：** Python 3.11、stdlib `unittest.mock`、pytest（测试也可 `python tests/xxx.py` 直接跑）、aiogram（handler）、faster-whisper、OpenAI 兼容 LLM。

**规格：** `docs/superpowers/specs/2026-08-09-audio-cache-reprocess-design.md`

---

## 文件结构

**修改：**
- `config.py` — `DataConfig` 新增 `AUDIO_DIR`。
- `core/task_manager.py` — `Task.audio_file` 改向 `AUDIO_DIR`；新增 `has_cached_audio` / `get_resume_action` / `save_audio_meta` / `load_audio_meta`；`TaskState` 新增 `CORRECTING`。
- `core/pipeline.py` — `_do_process` 重构为编排器 + 子函数；音频下载到 `AUDIO_DIR`；音频缓存命中时跳过联网；移除缓存快路径。
- `bot/handler.py` — `_process_single_url` 用 `get_resume_action` 决策。
- `utils/cleanup.py` — 新增 `AUDIO_MAX_AGE_DAYS` 与 `_cleanup_old_audio()`，挂到 `cleanup_scheduler` / `cleanup_sync`。

**测试（修改现有）：**
- `tests/test_pipeline_resume.py` — `_setup` 加 `AUDIO_DIR`；`test_intermediate_file_states` 改音频位置与"完成后保留"断言。
- `tests/test_task_manager.py` — `setup_temp_data` 加 `AUDIO_DIR`；`test_has_audio` / `test_resume_logic` 改音频位置。

**测试（新增用例，加在现有文件内 / 新建）：**
- `tests/test_pipeline_resume.py` — reprocess 跳过联网、CORRECTING 状态、标题从 meta.json 恢复。
- `tests/test_task_manager.py` — `get_resume_action` 四象限、`has_cached_audio`、audio meta 存取。
- `tests/test_cleanup.py` — `_cleanup_old_audio`。
- `tests/test_handler_dispatch.py`（新建）— handler return_cached / reprocess 分发。

---

## 任务 1：重构 `_do_process` + 新增 `CORRECTING`（行为不变）

**文件：**
- 修改：`core/task_manager.py`（`TaskState` 枚举）
- 修改：`core/pipeline.py`（`_do_process` 重构）
- 测试：`tests/test_pipeline_resume.py`（新增 characterization 测试）

本任务**不**搬音频、**不**加音频缓存逻辑，保持现有行为。仅拆函数 + 加 `CORRECTING`。保留 `_do_process` 开头的 `get_cached_result` 快路径。

- [ ] **步骤 1：跑基线，确认现有测试全绿**

运行：`python -m pytest tests/test_pipeline_resume.py tests/test_task_manager.py -v`
预期：全 PASS（建立重构安全网）。

- [ ] **步骤 2：`TaskState` 新增 `CORRECTING`**

`core/task_manager.py`：

```python
class TaskState(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    CORRECTING = "correcting"        # 新增：纠错
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"
```

- [ ] **步骤 3：写 characterization 测试（断言纠错时进入 CORRECTING）**

在 `tests/test_pipeline_resume.py` 末尾新增（复用现有 `_setup` / `_mock_downloader` / `BILIBILI_URL`）：

```python
def test_correction_sets_correcting_state():
    """音频路径纠错时，update_state 应被以 CORRECTING 调用过。"""
    tmp = tempfile.mkdtemp(prefix="test_corr_state_")
    _setup(tmp)
    try:
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        pipeline._transcriber = mock.MagicMock()
        pipeline._transcriber.transcript.return_value = "RAW TEXT"
        pipeline._summarizer = mock.MagicMock()
        pipeline._summarizer.correct.return_value = "CORRECTED"
        pipeline._summarizer.summarize.return_value = "# Summary"

        with mock.patch.object(task_manager, "update_state", wraps=task_manager.update_state) as spy, \
             mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is not None
        seen = [c.args[1] for c in spy.call_args_list if len(c.args) >= 2 and isinstance(c.args[1], TaskState)]
        assert TaskState.CORRECTING in seen, f"期望经过 CORRECTING，实际：{seen}"
        print("test_correction_sets_correcting_state PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

并在 `if __name__ == "__main__":` 块追加 `test_correction_sets_correcting_state()`。

- [ ] **步骤 4：运行测试验证失败**

运行：`python -m pytest tests/test_pipeline_resume.py::test_correction_sets_correcting_state -v`
预期：FAIL（pipeline 当前不设 CORRECTING）。

- [ ] **步骤 5：重构 `_do_process` 为编排器 + 子函数**

把 `core/pipeline.py` 中整个 `_do_process` 方法替换为下面的编排器，并新增这些私有方法（放在 `_do_process` 之后）。**保持音频仍在 task 目录、保留开头缓存快路径**：

```python
    def _do_process(self, url: str, task: Task, mem_start: float) -> Optional[PipelineResult]:
        # 已完成：返回缓存（任务 3 将由入口决策接管）
        cached = self.task_manager.get_cached_result(task.task_id)
        if cached:
            logger.info(f"Returning cached result for {task.task_id}")
            return PipelineResult(task_id=task.task_id, title=task.title or "video", content=cached)

        self._reset_if_failed(task)
        downloader = get_downloader(task.platform)

        self._ensure_title(task, downloader, url)
        text = self._acquire_text(task, downloader, url)
        if not text:
            logger.error("No text content available for summarization")
            self.task_manager.update_state(task, TaskState.FAILED, error="No text content")
            return None

        markdown, model_name = self._summarize(task, text)
        self._finalize(task, markdown, model_name, mem_start)
        return PipelineResult(task_id=task.task_id, title=task.title, content=markdown, model_name=model_name)

    # ── 编排子函数 ──────────────────────────────────────────────

    def _reset_if_failed(self, task: Task):
        if task.state == TaskState.FAILED:
            logger.info(f"Retrying previously failed task {task.task_id}")
            task.error = None
            self.task_manager.update_state(task, TaskState.PENDING)

    def _ensure_title(self, task: Task, downloader, url: str):
        """任务 1：始终联网取标题（与原行为一致）。任务 3 改为按 audio_cached 跳过。"""
        try:
            info = downloader.extract_info(url)
            task.title = info.get("title", "Unknown Video")
        except Exception as e:
            logger.warning(f"Failed to extract info: {e}")
            task.title = "Unknown Video"

    def _acquire_text(self, task: Task, downloader, url: str) -> str:
        """取总结用文本：字幕优先，否则音频转写 + 纠错。"""
        subtitle = self._maybe_get_subtitle(task, downloader, url)
        if subtitle:
            return subtitle
        self._ensure_audio(task, downloader, url)
        transcript = self._transcribe(task)
        if not transcript:
            return ""
        return self._correct(task, url, transcript) or transcript

    def _maybe_get_subtitle(self, task: Task, downloader, url: str) -> Optional[str]:
        if self.task_manager.has_subtitle(task):
            return self.task_manager.load_subtitle(task)
        self.task_manager.update_state(task, TaskState.DOWNLOADING)
        try:
            subtitle = downloader.extract_subtitles(url)
            if subtitle:
                self.task_manager.save_subtitle(task, subtitle)
                logger.info(f"Subtitles extracted ({len(subtitle)} chars)")
            return subtitle
        except Exception as e:
            logger.warning(f"Subtitle extraction failed: {e}")
            return None

    def _ensure_audio(self, task: Task, downloader, url: str):
        """任务 1：仍下载到 task 目录。任务 2 改为下载到 AUDIO_DIR。"""
        if self.task_manager.has_audio(task):
            return
        self.task_manager.update_state(task, TaskState.DOWNLOADING)
        result = downloader.download_audio(url, str(task.task_dir))
        task.title = result.title or task.title
        logger.info(f"Audio downloaded: {result.file_path}")

    def _transcribe(self, task: Task) -> str:
        if self.task_manager.has_transcript(task):
            return self.task_manager.load_transcript(task)
        self.task_manager.update_state(task, TaskState.TRANSCRIBING)
        self._log_mem("Before transcription")
        audio_text = self.transcriber.transcript(str(task.audio_file))
        self.task_manager.save_transcript(task, audio_text)
        logger.info(f"Transcription done ({len(audio_text)} chars)")
        self._reclaim_mem("After transcription (GC+trim)")
        return audio_text

    def _correct(self, task: Task, url: str, transcript: str) -> Optional[str]:
        if self.task_manager.has_corrected(task):
            corrected = self.task_manager.load_corrected(task)
            model = self.task_manager.load_corrected_model(task)
            self._ensure_mp3_md(task, url, corrected, model)
            return corrected
        self.task_manager.update_state(task, TaskState.CORRECTING)
        self._reclaim_mem("Before correction")
        try:
            corrected = self.summarizer.correct(transcript)
            model = self.summarizer._last_model_name
            self.task_manager.save_corrected(task, corrected, model_name=model)
            self._write_mp3_md(task, url, corrected, model)
            logger.info(f"Corrected transcript ({len(corrected)} chars)")
            self._reclaim_mem("After correction (GC+trim)")
            return corrected
        except Exception as e:
            logger.warning(f"Transcript correction failed, falling back to raw transcript: {e}", exc_info=True)
            self.task_manager.update_state(task, task.state, correction_failed=True)
            return None

    def _write_mp3_md(self, task: Task, url: str, text: str, model):
        from bot.formatter import build_transcript_markdown
        md = build_transcript_markdown(task.title or "video", url, text, model_name=model)
        path = self.data_config.NOTES_DIR / f"{task.task_id}.mp3.md"
        path.write_text(md, encoding="utf-8")
        logger.info(f"Corrected transcript -> {path}")

    def _ensure_mp3_md(self, task: Task, url: str, text: str, model):
        path = self.data_config.NOTES_DIR / f"{task.task_id}.mp3.md"
        if not path.exists():
            self._write_mp3_md(task, url, text, model)

    def _summarize(self, task: Task, text: str):
        self.task_manager.update_state(task, TaskState.SUMMARIZING)
        self._log_mem("Before summarization")
        markdown = self.summarizer.summarize(task.title, text)
        model = self.summarizer._last_model_name
        self._log_mem("After summarization")
        return markdown, (model if isinstance(model, str) else None)

    def _finalize(self, task: Task, markdown: str, model_name, mem_start: float):
        task.model_name = model_name
        self.task_manager.save_note(task, markdown)
        self.task_manager.update_state(task, TaskState.COMPLETED)
        self.task_manager.cleanup_task_files(task)
        gc.collect()
        _malloc_trim()
        mem_end = _mem_mb()
        logger.info(f"[MEM] Pipeline completed for {task.task_id} [mem: {mem_end:.0f}MB, delta: {mem_end-mem_start:.0f}MB]")

    def _log_mem(self, label: str):
        logger.info(f"[MEM] {label} [mem: {_mem_mb():.0f}MB]")

    def _reclaim_mem(self, label: str):
        gc.collect()
        _malloc_trim()
        logger.info(f"[MEM] {label} [mem: {_mem_mb():.0f}MB]")
```

> 说明：相比原代码去掉了 `del audio_text; audio_text = load_transcript()` 这段 cargo-cult（保留真正有用的 `gc.collect()+_malloc_trim()`），语义等价。

- [ ] **步骤 6：运行全部相关测试验证通过**

运行：`python -m pytest tests/test_pipeline_resume.py tests/test_task_manager.py tests/test_corrector.py -v`
预期：全 PASS（含新增 `test_correction_sets_correcting_state`）。

- [ ] **步骤 7：Commit**

```bash
git add core/task_manager.py core/pipeline.py tests/test_pipeline_resume.py
git commit -m "refactor(pipeline): 拆分 _do_process 为子函数，新增 CORRECTING 状态

行为不变。_do_process 降为薄编排器；纠错阶段显式进入 CORRECTING。
去掉无用的 del+reload 内存操作，统一 _log_mem/_reclaim_mem。"
```

## 任务 2：音频迁移到 `data/audio/` + meta.json 标题持久化

**文件：**
- 修改：`config.py`（`DataConfig.AUDIO_DIR`）
- 修改：`core/task_manager.py`（`Task.audio_file`、`has_cached_audio`、`save_audio_meta`、`load_audio_meta`）
- 修改：`core/pipeline.py`（`_ensure_audio` 下载到 AUDIO_DIR + 写 meta.json）
- 测试：`tests/test_task_manager.py`、`tests/test_pipeline_resume.py`（helper + 既有用例适配）

本任务搬音频位置；完成后音频不再被 `cleanup_task_files` 删除（因为它在 AUDIO_DIR）。**仍不**改入口决策——第二个请求此时仍返回缓存笔记（安全中间态）。`meta.json` 本任务写入，任务 3 消费。

- [ ] **步骤 1：`DataConfig` 新增 `AUDIO_DIR`**

`config.py` 的 `DataConfig`：

```python
class DataConfig:
    DATA_DIR = Path(_cfg.get("data", {}).get("dir", "./data"))
    TASKS_DIR = DATA_DIR / "tasks"
    NOTES_DIR = DATA_DIR / "notes"
    AUDIO_DIR = DATA_DIR / "audio"          # 新增
    MODELS_DIR = BASE_DIR / "models"

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)   # 新增
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **步骤 2：写失败测试 — `has_cached_audio` / audio meta 存取**

在 `tests/test_task_manager.py` 新增：

```python
def test_audio_cache_and_meta():
    tmp = setup_temp_data(None)
    try:
        from config import DataConfig
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        assert not mgr.has_cached_audio("abc123")

        # 模拟下载器写入音频到 AUDIO_DIR/{task_id}/
        audio_dir = DataConfig.AUDIO_DIR / "abc123"
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "audio.mp3").write_bytes(b"fake")
        assert mgr.has_cached_audio("abc123")
        assert mgr.has_audio(task), "has_audio 应指向 AUDIO_DIR"

        # meta 存取
        mgr.save_audio_meta("abc123", "My Video Title")
        assert mgr.load_audio_meta("abc123") == "My Video Title"
        assert mgr.load_audio_meta("nope") is None
    finally:
        teardown_temp_data(tmp)
```

并在 `if __name__ == "__main__":` 追加 `test_audio_cache_and_meta()`。

- [ ] **步骤 3：运行验证失败**

运行：`python -m pytest tests/test_task_manager.py::test_audio_cache_and_meta -v`
预期：FAIL（`has_cached_audio` 不存在 / helper 未设 AUDIO_DIR）。

- [ ] **步骤 4：更新测试 helper 设置 `AUDIO_DIR`**

`tests/test_task_manager.py` 的 `setup_temp_data`：

```python
def setup_temp_data(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="test_videobot_")
    from config import DataConfig
    DataConfig.DATA_DIR = Path(tmp)
    DataConfig.TASKS_DIR = Path(tmp) / "tasks"
    DataConfig.NOTES_DIR = Path(tmp) / "notes"
    DataConfig.AUDIO_DIR = Path(tmp) / "audio"          # 新增
    DataConfig.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.AUDIO_DIR.mkdir(parents=True, exist_ok=True)   # 新增
    return tmp
```

`tests/test_pipeline_resume.py` 的 `_setup`：

```python
def _setup(tmp_dir):
    from config import DataConfig
    DataConfig.DATA_DIR = Path(tmp_dir)
    DataConfig.TASKS_DIR = Path(tmp_dir) / "tasks"
    DataConfig.NOTES_DIR = Path(tmp_dir) / "notes"
    DataConfig.AUDIO_DIR = Path(tmp_dir) / "audio"      # 新增
    DataConfig.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.AUDIO_DIR.mkdir(parents=True, exist_ok=True)   # 新增
```

- [ ] **步骤 5：实现 `task_manager` 新方法 + 改 `audio_file`**

`core/task_manager.py` —— `Task.audio_file` 属性改为指向 AUDIO_DIR：

```python
    @property
    def audio_file(self) -> Path:
        audio_dir = DataConfig.AUDIO_DIR / self.task_id
        for ext in ("mp3", "m4a", "wav", "webm"):
            p = audio_dir / f"audio.{ext}"
            if p.exists():
                return p
        return audio_dir / "audio.mp3"
```

`TaskManager` 新增方法（放在 `has_audio` 附近）：

```python
    def has_cached_audio(self, task_id: str) -> bool:
        audio_dir = DataConfig.AUDIO_DIR / task_id
        if not audio_dir.exists():
            return False
        return any((audio_dir / f"audio.{ext}").exists() for ext in ("mp3", "m4a", "wav", "webm"))

    def save_audio_meta(self, task_id: str, title: str):
        meta_path = DataConfig.AUDIO_DIR / task_id / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"title": title}, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_audio_meta(self, task_id: str) -> Optional[str]:
        meta_path = DataConfig.AUDIO_DIR / task_id / "meta.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8")).get("title")
```

- [ ] **步骤 6：`_ensure_audio` 下载到 AUDIO_DIR + 写 meta.json**

`core/pipeline.py`：

```python
    def _ensure_audio(self, task: Task, downloader, url: str):
        if self.task_manager.has_audio(task):
            return
        self.task_manager.update_state(task, TaskState.DOWNLOADING)
        from config import DataConfig
        audio_dir = DataConfig.AUDIO_DIR / task.task_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        result = downloader.download_audio(url, str(audio_dir))
        task.title = result.title or task.title
        self.task_manager.save_audio_meta(task.task_id, task.title or "Unknown Video")
        logger.info(f"Audio downloaded: {result.file_path}")
```

- [ ] **步骤 7：适配既有"音频位置"测试**

`tests/test_task_manager.py::test_has_audio`：

```python
def test_has_audio():
    tmp = setup_temp_data(None)
    try:
        from config import DataConfig
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        assert not mgr.has_audio(task)
        audio_dir = DataConfig.AUDIO_DIR / "abc123"
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "audio.mp3").write_bytes(b"fake audio")
        assert mgr.has_audio(task)
    finally:
        teardown_temp_data(tmp)
```

`tests/test_task_manager.py::test_resume_logic` —— 把音频写到 AUDIO_DIR（替换原 `task.task_dir / "audio.mp3"` 两行）：

```python
        # Audio exists - can resume from transcription
        from config import DataConfig
        audio_dir = DataConfig.AUDIO_DIR / task.task_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "audio.mp3").write_bytes(b"fake")
        assert mgr.can_resume_from_transcription(task)
```

- [ ] **步骤 8：适配 `test_intermediate_file_states`（完成后音频保留）**

`tests/test_pipeline_resume.py::test_intermediate_file_states` —— State 1 把音频放进 AUDIO_DIR：

```python
        # State 1: After download failure (simulate by creating audio, no transcript)
        from config import DataConfig
        audio_dir = DataConfig.AUDIO_DIR / task_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "audio.mp3").write_bytes(b"fake")
        task_manager.update_state(task, TaskState.FAILED, error="transcription crash")

        assert task_manager.has_audio(task)
```

State 3 改为"完成后音频保留在 AUDIO_DIR、task 目录中间文件被清"：

```python
        # State 3: After completion (simulate note + cleanup)
        task_manager.save_note(task, "# Summary")
        task_manager.update_state(task, TaskState.COMPLETED)
        task_manager.cleanup_task_files(task)

        assert task_manager.has_audio(task), "完成后音频应在 AUDIO_DIR 保留"
        assert not task_manager.has_transcript(task), "transcript 被清"
        assert not (task.task_dir / "audio.mp3").exists(), "task_dir 内不再有音频"
        assert task_manager.get_cached_result(task_id)  # note persists
```

- [ ] **步骤 9：运行全部相关测试**

运行：`python -m pytest tests/test_task_manager.py tests/test_pipeline_resume.py -v`
预期：全 PASS。

- [ ] **步骤 10：Commit**

```bash
git add config.py core/task_manager.py core/pipeline.py tests/test_task_manager.py tests/test_pipeline_resume.py
git commit -m "feat(audio): 音频迁移到 data/audio/{task_id}/ 并写 meta.json

音频不再随 cleanup_task_files 删除；title 持久化到 meta.json 供后续重跑恢复。
新增 has_cached_audio/save_audio_meta/load_audio_meta；测试 helper 同步 AUDIO_DIR。"
```

## 任务 3：入口决策 `get_resume_action` + reprocess 跳过联网 + 移除缓存快路径

**文件：**
- 修改：`core/task_manager.py`（`get_resume_action`）
- 修改：`core/pipeline.py`（`_do_process` 移除缓存快路径；`_ensure_title`/`_maybe_get_subtitle`/`_acquire_text` 加 `audio_cached` 跳过联网）
- 修改：`bot/handler.py`（`_process_single_url` 用 `get_resume_action`）
- 测试：`tests/test_task_manager.py`、`tests/test_pipeline_resume.py`、新建 `tests/test_handler_dispatch.py`

核心行为变更：第二请求音频缓存命中 → 重跑转写/纠错/总结，零联网。

- [ ] **步骤 1：写失败测试 — `get_resume_action` 四象限**

`tests/test_task_manager.py` 新增：

```python
def test_get_resume_action_quadrants():
    tmp = setup_temp_data(None)
    try:
        from config import DataConfig
        mgr = TaskManager()
        task = mgr.create_task("abc", "https://example.com", "youtube")

        # 无音频 + 无笔记 -> full
        assert mgr.get_resume_action(task) == "full"

        # 有笔记 + 无音频 -> return_cached
        mgr.save_note(task, "# Note")
        assert mgr.get_resume_action(task) == "return_cached"

        # 有音频(+有笔记) -> reprocess（音频优先）
        audio_dir = DataConfig.AUDIO_DIR / task.task_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "audio.mp3").write_bytes(b"fake")
        assert mgr.get_resume_action(task) == "reprocess"

        # 有音频 + 无笔记 -> reprocess
        (DataConfig.NOTES_DIR / f"{task.task_id}.md").unlink()
        assert mgr.get_resume_action(task) == "reprocess"
    finally:
        teardown_temp_data(tmp)
```

追加到 `__main__` 块。

- [ ] **步骤 2：运行验证失败**

运行：`python -m pytest tests/test_task_manager.py::test_get_resume_action_quadrants -v`
预期：FAIL（`get_resume_action` 不存在）。

- [ ] **步骤 3：实现 `get_resume_action`**

`core/task_manager.py`（`TaskManager` 内）：

```python
    def get_resume_action(self, task: Task) -> str:
        """决定本次请求如何处理：
        'reprocess'     : 音频缓存命中 -> 从音频重跑，跳过一切联网
        'return_cached' : 有笔记无音频 -> 直接返回缓存笔记（不联网）
        'full'          : 无音频 -> 正常全流程（联网下载）
        """
        if self.has_cached_audio(task.task_id):
            return "reprocess"
        if task.note_file.exists():
            return "return_cached"
        return "full"
```

- [ ] **步骤 4：运行验证通过**

运行：`python -m pytest tests/test_task_manager.py::test_get_resume_action_quadrants -v`
预期：PASS。

- [ ] **步骤 5：写失败测试 — reprocess 跳过联网**

`tests/test_pipeline_resume.py` 新增：

```python
def test_reprocess_skips_all_networking():
    """音频缓存命中时重跑：extract_info/extract_subtitles/download_audio 均不被调用。"""
    tmp = tempfile.mkdtemp(prefix="test_reprocess_")
    _setup(tmp)
    try:
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        # Phase 1：首次跑通，音频落到 AUDIO_DIR
        pipeline._transcriber = mock.MagicMock()
        pipeline._transcriber.transcript.return_value = "RAW"
        pipeline._summarizer = mock.MagicMock()
        pipeline._summarizer.correct.return_value = "CORRECTED"
        pipeline._summarizer.summarize.return_value = "# First"
        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader(title="My Title")):
            pipeline.process(url, task)
        assert task_manager.has_cached_audio(task.task_id)

        # Phase 2：重跑 —— 全新 downloader mock，断言零联网
        pipeline._transcriber.transcript.return_value = "RAW2"
        pipeline._summarizer.summarize.return_value = "# Second"
        strict_dl = mock.MagicMock()
        with mock.patch("core.pipeline.get_downloader", return_value=strict_dl):
            task = task_manager.get_task(task.task_id)
            result = pipeline.process(url, task)

        assert result is not None
        assert result.content == "# Second"
        strict_dl.extract_info.assert_not_called()
        strict_dl.extract_subtitles.assert_not_called()
        strict_dl.download_audio.assert_not_called()
        pipeline._transcriber.transcript.assert_called()
        pipeline._summarizer.summarize.assert_called_once()
        print("test_reprocess_skips_all_networking PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

> 移除缓存快路径前，此测试因快路径直接返回 `"# First"` 而 FAIL，正是预期失败。

追加到 `__main__` 块。

- [ ] **步骤 6：运行验证失败**

运行：`python -m pytest tests/test_pipeline_resume.py::test_reprocess_skips_all_networking -v`
预期：FAIL（缓存快路径返回旧结果，`result.content == "# First"` ≠ `"# Second"`）。

- [ ] **步骤 7：改 pipeline 接入 audio_cached + 移除缓存快路径**

`core/pipeline.py` —— `_do_process` 移除缓存快路径并计算 `audio_cached`：

```python
    def _do_process(self, url: str, task: Task, mem_start: float) -> Optional[PipelineResult]:
        self._reset_if_failed(task)
        audio_cached = self.task_manager.has_cached_audio(task.task_id)
        downloader = get_downloader(task.platform)

        self._ensure_title(task, downloader, url, audio_cached)
        text = self._acquire_text(task, downloader, url, audio_cached)
        if not text:
            logger.error("No text content available for summarization")
            self.task_manager.update_state(task, TaskState.FAILED, error="No text content")
            return None

        markdown, model_name = self._summarize(task, text)
        self._finalize(task, markdown, model_name, mem_start)
        return PipelineResult(task_id=task.task_id, title=task.title, content=markdown, model_name=model_name)
```

`_ensure_title` / `_acquire_text` / `_maybe_get_subtitle` 改签名（audio_cached 时跳过联网）：

```python
    def _ensure_title(self, task: Task, downloader, url: str, audio_cached: bool):
        if task.title:
            return
        if audio_cached:
            task.title = self.task_manager.load_audio_meta(task.task_id) or "Unknown Video"
            return
        try:
            info = downloader.extract_info(url)
            task.title = info.get("title", "Unknown Video")
        except Exception as e:
            logger.warning(f"Failed to extract info: {e}")
            task.title = "Unknown Video"

    def _acquire_text(self, task: Task, downloader, url: str, audio_cached: bool) -> str:
        subtitle = self._maybe_get_subtitle(task, downloader, url, audio_cached)
        if subtitle:
            return subtitle
        self._ensure_audio(task, downloader, url)
        transcript = self._transcribe(task)
        if not transcript:
            return ""
        return self._correct(task, url, transcript) or transcript

    def _maybe_get_subtitle(self, task: Task, downloader, url: str, audio_cached: bool) -> Optional[str]:
        if audio_cached:
            return None
        if self.task_manager.has_subtitle(task):
            return self.task_manager.load_subtitle(task)
        self.task_manager.update_state(task, TaskState.DOWNLOADING)
        try:
            subtitle = downloader.extract_subtitles(url)
            if subtitle:
                self.task_manager.save_subtitle(task, subtitle)
                logger.info(f"Subtitles extracted ({len(subtitle)} chars)")
            return subtitle
        except Exception as e:
            logger.warning(f"Subtitle extraction failed: {e}")
            return None
```

`_ensure_audio` 保持任务 2 版本不变（`has_audio` 命中即跳过下载）。

- [ ] **步骤 8：写失败测试 — 标题从 meta.json 恢复**

`tests/test_pipeline_resume.py` 新增（模拟 task 目录被 1 天定时清理后重跑）：

```python
def test_reprocess_recovers_title_from_meta():
    """task 目录被清理（标题丢失）后重跑，从 meta.json 恢复标题且不联网。"""
    tmp = tempfile.mkdtemp(prefix="test_title_meta_")
    _setup(tmp)
    try:
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        pipeline._transcriber = mock.MagicMock()
        pipeline._transcriber.transcript.return_value = "RAW"
        pipeline._summarizer = mock.MagicMock()
        pipeline._summarizer.correct.return_value = "CORRECTED"
        pipeline._summarizer.summarize.return_value = "# Summary"
        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader(title="The Real Title")):
            pipeline.process(url, task)

        # 模拟 task 目录被 1 天定时清理
        import shutil as _sh
        _sh.rmtree(task.task_dir, ignore_errors=True)
        assert task_manager.has_cached_audio(task.task_id)
        assert task_manager.load_audio_meta(task.task_id) == "The Real Title"

        # 重跑：strict downloader 不应联网；标题从 meta 恢复
        pipeline._summarizer.summarize.return_value = "# Summary 2"
        strict_dl = mock.MagicMock()
        with mock.patch("core.pipeline.get_downloader", return_value=strict_dl):
            new_task, _ = task_manager.get_or_create(url, "bilibili")
            result = pipeline.process(url, new_task)

        assert result is not None
        assert result.title == "The Real Title", "标题应从 meta.json 恢复"
        strict_dl.extract_info.assert_not_called()
        print("test_reprocess_recovers_title_from_meta PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

追加到 `__main__` 块。

- [ ] **步骤 9：运行 pipeline 测试验证通过**

运行：`python -m pytest tests/test_pipeline_resume.py -v`
预期：全 PASS（含 reprocess 跳过联网 + 标题恢复）。

- [ ] **步骤 10：handler 改用 `get_resume_action`**

`bot/handler.py` 的 `_process_single_url`，把开头缓存判断替换为：

```python
    task, _existed = task_manager.get_or_create(url, platform)
    action = task_manager.get_resume_action(task)

    if action == "return_cached":
        cached = task_manager.get_cached_result(task.task_id)
        if cached:
            full_task = task_manager.get_task(task.task_id)
            model_name = full_task.model_name if full_task else None
            await _send_note(message, task.title or "video", task.platform, url, cached, model_name, task_id=task.task_id)
            return True, ""
        # 笔记恰好消失 -> 回落到 full 流程
```

> 下面的 `for attempt in range(...)` 重试循环原样保留，对 `reprocess` 和 `full` 都走 `pipeline.process`。删除原本 `if is_cached and task.state == TaskState.COMPLETED:` 那段（已被上面取代）。

- [ ] **步骤 11：写 handler 分发测试**

新建 `tests/test_handler_dispatch.py`：

```python
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile, shutil
from unittest import mock


def _setup_handler_tmp():
    tmp = tempfile.mkdtemp(prefix="test_handler_")
    from config import DataConfig
    DataConfig.TASKS_DIR = Path(tmp) / "tasks"
    DataConfig.NOTES_DIR = Path(tmp) / "notes"
    DataConfig.AUDIO_DIR = Path(tmp) / "audio"
    for d in (DataConfig.TASKS_DIR, DataConfig.NOTES_DIR, DataConfig.AUDIO_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return tmp


def test_handler_returns_cached_without_pipeline():
    """return_cached：发缓存笔记，不调用 pipeline.process。"""
    from bot import handler
    tmp = _setup_handler_tmp()
    try:
        task = mock.MagicMock(task_id="abc", title="T", platform="bilibili", url="u")
        with mock.patch.object(handler, "task_manager") as tm, \
             mock.patch.object(handler, "pipeline") as pl, \
             mock.patch.object(handler, "_send_note", new=mock.AsyncMock()) as send_note, \
             mock.patch.object(handler, "parse_platform", return_value="bilibili"):
            tm.get_or_create.return_value = (task, True)
            tm.get_resume_action.return_value = "return_cached"
            tm.get_cached_result.return_value = "CACHED"
            tm.get_task.return_value = None
            ok, err = asyncio.run(handler._process_single_url(mock.MagicMock(), mock.MagicMock(), "u", 1, 1))
        assert ok is True and err == ""
        send_note.assert_awaited_once()
        pl.process.assert_not_called()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_handler_runs_pipeline_on_reprocess():
    """reprocess：调用 pipeline.process 并发送其结果。"""
    from bot import handler
    tmp = _setup_handler_tmp()
    try:
        task = mock.MagicMock(task_id="abc", title="T", platform="bilibili", url="u")
        with mock.patch.object(handler, "task_manager") as tm, \
             mock.patch.object(handler, "pipeline") as pl, \
             mock.patch.object(handler, "_send_note", new=mock.AsyncMock()) as send_note, \
             mock.patch.object(handler, "parse_platform", return_value="bilibili"):
            tm.get_or_create.return_value = (task, True)
            tm.get_resume_action.return_value = "reprocess"
            pl.process.return_value = mock.MagicMock(title="T", content="# New", model_name="m")
            status_msg = mock.MagicMock()
            status_msg.edit_text = mock.AsyncMock()
            ok, err = asyncio.run(handler._process_single_url(mock.MagicMock(), status_msg, "u", 1, 1))
        assert ok is True and err == ""
        pl.process.assert_called_once()
        send_note.assert_awaited_once()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_handler_returns_cached_without_pipeline()
    test_handler_runs_pipeline_on_reprocess()
    print("All handler dispatch tests passed!")
```

- [ ] **步骤 12：运行 handler + 全量测试**

运行：`python -m pytest tests/test_handler_dispatch.py tests/test_pipeline_resume.py tests/test_task_manager.py -v`
预期：全 PASS。

- [ ] **步骤 13：Commit**

```bash
git add core/task_manager.py core/pipeline.py bot/handler.py tests/test_task_manager.py tests/test_pipeline_resume.py tests/test_handler_dispatch.py
git commit -m "feat(pipeline): 第二请求从缓存音频重跑，跳过一切联网

新增 get_resume_action 三态决策；音频缓存命中时 pipeline 跳过
extract_info/extract_subtitles/download_audio；移除 _do_process 缓存
快路径；handler 按决策分发。标题从 meta.json 恢复。"
```

## 任务 4：音频 TTL 清理 `_cleanup_old_audio`

**文件：**
- 修改：`utils/cleanup.py`（新增 `AUDIO_MAX_AGE_DAYS`、`_cleanup_old_audio()`，挂到 `cleanup_scheduler` 与 `cleanup_sync`）
- 测试：`tests/test_cleanup.py`

- [ ] **步骤 1：写失败测试 — `_cleanup_old_audio`**

`tests/test_cleanup.py` 的 `setup_temp_data` 加上 AUDIO_DIR：

```python
def setup_temp_data():
    tmp = tempfile.mkdtemp(prefix="test_cleanup_")
    from config import DataConfig
    DataConfig.NOTES_DIR = Path(tmp) / "notes"
    DataConfig.TASKS_DIR = Path(tmp) / "tasks"
    DataConfig.AUDIO_DIR = Path(tmp) / "audio"          # 新增
    DataConfig.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    DataConfig.AUDIO_DIR.mkdir(parents=True, exist_ok=True)   # 新增
    return tmp
```

并在文件顶部 import 改为：

```python
from utils.cleanup import _cleanup_notes, _cleanup_stale_tasks, _cleanup_old_audio
```

新增测试：

```python
def test_cleanup_old_audio():
    from utils.cleanup import AUDIO_MAX_AGE_DAYS
    tmp = setup_temp_data()
    try:
        from config import DataConfig

        # 旧音频子目录（>AUDIO_MAX_AGE_DAYS 天）
        old_audio = DataConfig.AUDIO_DIR / "old_task"
        old_audio.mkdir()
        (old_audio / "audio.mp3").write_bytes(b"old")
        (old_audio / "meta.json").write_text("{}", encoding="utf-8")
        old_time = time.time() - ((AUDIO_MAX_AGE_DAYS + 1) * 86400)
        os.utime(old_audio, (old_time, old_time))

        # 新音频子目录
        new_audio = DataConfig.AUDIO_DIR / "new_task"
        new_audio.mkdir()
        (new_audio / "audio.mp3").write_bytes(b"new")

        asyncio.get_event_loop().run_until_complete(_cleanup_old_audio())

        assert not old_audio.exists(), "旧音频应被清理"
        assert new_audio.exists(), "新音频应保留"
    finally:
        teardown(tmp)


def test_cleanup_old_audio_empty_dir():
    tmp = setup_temp_data()
    try:
        asyncio.get_event_loop().run_until_complete(_cleanup_old_audio())  # 不应报错
    finally:
        teardown(tmp)
```

追加到 `if __name__ == "__main__":` 块。

- [ ] **步骤 2：运行验证失败**

运行：`python -m pytest tests/test_cleanup.py::test_cleanup_old_audio -v`
预期：FAIL（`_cleanup_old_audio` 不存在 / import 失败）。

- [ ] **步骤 3：实现 `_cleanup_old_audio` + 挂到调度器**

`utils/cleanup.py`：

```python
NOTES_MAX_AGE_DAYS = 30
TASKS_MAX_AGE_DAYS = 1
AUDIO_MAX_AGE_DAYS = 7
```

```python
async def cleanup_scheduler():
    """Background task that periodically cleans up old files."""
    while True:
        try:
            await _cleanup_notes()
            await _cleanup_stale_tasks()
            await _cleanup_old_audio()        # 新增
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        await asyncio.sleep(3600)  # Run every hour
```

```python
async def _cleanup_old_audio():
    """Delete audio cache subdirs older than AUDIO_MAX_AGE_DAYS."""
    audio_dir = DataConfig.AUDIO_DIR
    if not audio_dir.exists():
        return

    cutoff = time.time() - (AUDIO_MAX_AGE_DAYS * 86400)
    count = 0

    for d in audio_dir.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            count += 1

    if count:
        logger.info(f"Cleaned up {count} audio caches (> {AUDIO_MAX_AGE_DAYS} days)")
```

`cleanup_sync` 也追加：

```python
def cleanup_sync():
    """Synchronous cleanup for testing."""
    import asyncio
    asyncio.get_event_loop().run_until_complete(_cleanup_notes())
    asyncio.get_event_loop().run_until_complete(_cleanup_stale_tasks())
    asyncio.get_event_loop().run_until_complete(_cleanup_old_audio())
```

- [ ] **步骤 4：运行 cleanup 测试验证通过**

运行：`python -m pytest tests/test_cleanup.py -v`
预期：全 PASS。

- [ ] **步骤 5：全量回归**

运行：`python -m pytest tests/ -v --ignore=tests/test_integration.py`
预期：全 PASS（集成测试需网络/模型，单独跑）。

- [ ] **步骤 6：Commit**

```bash
git add utils/cleanup.py tests/test_cleanup.py
git commit -m "feat(cleanup): 新增 _cleanup_old_audio，7 天 TTL 清理音频缓存

挂到 cleanup_scheduler 每小时循环与 cleanup_sync。"
```

---

## 完成标准

- 首次处理（音频路径）：音频落 `data/audio/{task_id}/audio.mp3` + `meta.json`；完成时 task 目录中间文件被清、音频保留；`TaskState` 经历 `DOWNLOADING→TRANSCRIBING→CORRECTING→SUMMARIZING→COMPLETED`。
- 第二请求（音频<7天）：跳过 `extract_info`/`extract_subtitles`/`download_audio`；重跑转写/纠错/总结；note 被覆盖。
- 第二请求（字幕路径 / 音频过期）：返回缓存笔记，不联网。
- task 目录被 1 天清理后（音频还在）重跑：标题从 `meta.json` 恢复，不退化、不联网。
- 音频 >7 天被 `_cleanup_old_audio` 清理。
- `_do_process` 降为薄编排器，全部单测通过。
