# 音频缓存与从音频重跑（Audio Cache & Reprocess-from-Audio）

- 日期：2026-08-09
- 状态：已通过头脑风暴，待编写实现计划
- 涉及模块：`core/pipeline.py`、`core/task_manager.py`、`bot/handler.py`、`utils/cleanup.py`、`config.py`

## 1. 背景与问题

长视频下载耗时长，且二次下载容易被视频平台限流/封禁而失败。当前流程在任务完成后调用 `cleanup_task_files()`，把整个 task 目录（含 `audio.mp3`）清空，只留 `status.json`；定时的 `_cleanup_stale_tasks()` 还会在 1 天后删掉整个 task 目录。

后果：一旦处理失败、或用户想对已成功的任务重跑（刷新纠错/总结），音频已被删除，必须重新联网下载——慢且大概率再次失败。

## 2. 目标

1. **保留音频**：处理完成（无论成功/失败）后，`audio.mp3` 不再被 `cleanup_task_files()` 删除，改为由定时任务按 TTL 删除。
2. **从音频重跑**：第二个请求进来时，若音频缓存仍在，跳过一切联网步骤（`extract_info` / `extract_subtitles` / `download_audio`），直接从本地音频重跑「转写→纠错→总结→上报」。即便该任务之前已成功输出过笔记，也照样重跑（刷新 LLM 产出）。
3. **避免联网**：核心约束——「不要重复向视频服务器发请求」。音频缓存命中即零联网。
4. **复用清理逻辑**：音频 TTL 删除挂到现有 `cleanup_scheduler`，不另起机制。

非目标：不缓存转写稿/纠错稿（每次从音频重跑）；不改变字幕快路径的视频仍返回缓存笔记的行为；不做旧音频迁移（单用户，一次性重下载可接受）。

## 3. 现状关键点

### 3.1 实际业务步骤 vs TaskState

| 实际步骤 | 当前 state |
|---|---|
| 解析 URL / 创建任务 | `PENDING` |
| 提取视频信息(title) | （无对应 state） |
| 抓字幕(快路径) / 下载音频 | `DOWNLOADING`（字幕和音频挤在一起） |
| Whisper 转写 | `TRANSCRIBING` |
| **纠错(correct)** | **（完全没有对应 state）** |
| 总结(summarize) | `SUMMARIZING` |
| 保存笔记 + 上报(_send_note 上传) | `COMPLETED` |
| 出错 | `FAILED` |

"上报" = `bot/handler.py` 的 `_send_note`（上传到 Telegram 等），在 `pipeline.process()` 返回之后执行，**不在 pipeline 内部**——只要 pipeline 重新跑出结果，上报会自动重做。

### 3.2 音频被删的根因

`Pipeline._do_process()` 在 `COMPLETED` 后调用 `TaskManager.cleanup_task_files()`，该方法 `shutil.rmtree(task_dir)` 删除 audio/transcript/corrected，只留 `status.json`。`_cleanup_stale_tasks()`（每小时，>1 天）还会删整个 task 目录。

### 3.3 当前"第二个请求"的行为

`get_or_create` 发现 note 存在就返回 `COMPLETED` 任务，handler 直接返回缓存笔记，**完全不重跑**。本设计要改掉这条。

## 4. 设计决策

| # | 决策 | 选项 | 选定 | 理由 |
|---|---|---|---|---|
| D1 | 第二请求刷新粒度 | A 只缓存音频 / B 缓存音频+转写稿 / C 全缓存 / D 返回缓存笔记 | **A** | 本地 Whisper 可接受重跑；满足"成功过也刷新"诉求 |
| D2 | 音频 TTL | 1/3/7/14/30 天 | **7 天** | 覆盖多数隔天/隔周重跑；单用户磁盘可控 |
| D3 | 字幕路径视频 | A 每次重抓重跑 / B 返回缓存笔记 / C 缓存字幕 | **B** | 不重复向服务器请求，避免封禁 |
| D4 | 音频存储位置 | 方案一 task 目录内+保护 / 方案二 独立 `data/audio/` | **方案二** | 音频生命周期与 task 目录解耦，现有清理零改动 |

### 4.1 避联网原则的统一

D3 的理由"不要重复向服务器发请求"同样适用于音频路径：既然本地有缓存的 mp3，重跑时连 `extract_info`（取标题）和 `extract_subtitles` 都不该再打服务器。因此统一规则：**音频缓存命中 → 跳过一切联网步骤**。

## 5. TaskState 重新设计

补上 `CORRECTING`，并明确语义分工：**state = 当前进度（瞬时）+ 终态**；**跳哪一步 = 看产物文件是否存在**（audio/transcript/corrected）。

```python
class TaskState(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"      # 抓字幕 或 下载音频（联网阶段）
    TRANSCRIBING = "transcribing"    # Whisper 转写
    CORRECTING = "correcting"        # 新增：纠错
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"
```

不采用里程碑式 state（DOWNLOADED/TRANSCRIBED/...）——里程碑信息与"文件存在性"重复，且 `/status` 用现有字段已够展示。保留动词式 + 新增 `CORRECTING` 最小且贴合现实。

## 6. 请求入口决策

新增 `TaskManager.get_resume_action(task) -> str`，三选一（判断优先级：**音频存在优先**）：

| 条件 | 动作 | 含义 |
|---|---|---|
| 音频缓存存在（`data/audio/{task_id}/` 有音频） | `reprocess` | 从音频重跑 转写→纠错→总结→上报，**跳过一切联网** |
| 笔记存在 且 无音频缓存 | `return_cached` | 直接返回缓存笔记，**不联网**（覆盖字幕路径 + 音频已过期） |
| 都没有 | `full` | 正常全流程（联网下载） |

`bot/handler.py::_process_single_url` 改为：
- `return_cached` → 直接发缓存笔记（走 `_send_note`），不进 pipeline；
- `reprocess` / `full` → 进 `pipeline.process`，外层 `MAX_RETRIES` 重试逻辑不变。

边界：`return_cached` 但 `get_cached_result` 返回 None（笔记文件恰好消失）→ 回落到 `full` 进 pipeline。

## 7. Pipeline 重构 + 音频缓存

### 7.1 目标

`_do_process` 从 ~90 行（圈复杂度 ~15-18）重构为薄编排器，每步一个职责单一的私有方法，单方法圈复杂度 ≤4，可独立单测。

### 7.2 编排器（重构后的 `_do_process`）

```python
def _do_process(self, url, task, mem_start):
    self._reset_if_failed(task)
    audio_cached = self.task_manager.has_cached_audio(task.task_id)
    downloader = get_downloader(task.platform)

    self._ensure_title(task, downloader, url, audio_cached)
    text = self._acquire_text(task, downloader, url, audio_cached)
    if not text:
        self.task_manager.update_state(task, TaskState.FAILED, error="No text content")
        return None

    markdown, model = self._summarize(task, text)
    self._finalize(task, markdown, model, mem_start)
    return PipelineResult(task.task_id, task.title, markdown, model)
```

> 原 `_do_process` 开头 `get_cached_result` 直接返回的快路径**删除**——缓存返回已交给入口决策。

### 7.3 拆出的子函数

| 子函数 | 职责 | 音频缓存规则 |
|---|---|---|
| `_reset_if_failed(task)` | FAILED → 重置回 PENDING | — |
| `_ensure_title(task, dl, url, audio_cached)` | 补标题 | 已有标题不动；audio_cached 时不联网置 "Unknown Video"；否则 `extract_info` |
| `_acquire_text(...) -> str` | 取总结用文本：字幕优先，否则音频转写+纠错 | — |
| └ `_maybe_get_subtitle(...) -> str\|None` | 字幕快路径 | audio_cached 时直接返回 None（不联网） |
| └ `_ensure_audio(task, dl, url)` | 音频不存在才下载（含 `mkdir AUDIO_DIR/{task_id}`） | 存在自动跳过 |
| └ `_transcribe(task) -> str` | Whisper 转写；`has_transcript` 复用否则转写保存 | — |
| └ `_correct(task, transcript) -> str\|None` | 纠错；产出 corrected.json + `.mp3.md`；失败置 `correction_failed` 返回 None | — |
| `_summarize(task, text) -> (md, model)` | 总结 | — |
| `_finalize(task, md, model, mem_start)` | 存笔记→COMPLETED→`cleanup_task_files`→回收内存→收尾日志 | 音频在 AUDIO_DIR，cleanup 碰不到 |

辅助：`_log_mem(label)` / `_reclaim_mem()`（统一原先散落各处的 `gc.collect()+_malloc_trim()+logger.info("[MEM]...")`）；顺手去掉原 `del audio_text; audio_text = load_transcript()` 这类 cargo-cult 内存操作（保留真正有用的 gc+trim）。

### 7.4 断点续传与重跑的关系（关键）

`cleanup_task_files()` 在完成时仍会删 transcript/corrected（只不删音频，因为音频已搬到 AUDIO_DIR）。所以：
- **成功过的任务再请求**：transcript/corrected 已被清 → 现有 `if not has_transcript / if not has_corrected` 条件自然为真 → 自动重跑转写/纠错 = D1 的 A 语义。
- **失败任务重试**（transcript 仍在）：复用已有 transcript，不白白重跑 Whisper。

即：**无需拆掉现有断点续传逻辑**，靠"完成时清中间文件"自然达成重跑。

## 8. 音频存储 & 清理（方案二）

### 8.1 文件布局

```
data/
├── tasks/{task_id}/        # status.json + 瞬时中间文件(transcript/corrected/subtitle)
├── audio/{task_id}/audio.mp3   # 新增：持久音频缓存，7 天 TTL
└── notes/{task_id}.md / {task_id}.mp3.md
```

### 8.2 改动点

- `config.py`：`DataConfig` 新增 `AUDIO_DIR`（`data/audio`）。
- `core/task_manager.py`：
  - `Task.audio_file` 属性指向 `AUDIO_DIR / {task_id} / audio.{ext}`（保留多扩展名回退）。
  - 新增 `has_cached_audio(task_id) -> bool`：`AUDIO_DIR/{task_id}` 子目录存在且有音频文件。
  - 新增 `get_resume_action(task) -> str`（见第 6 节）。
- `core/pipeline.py`：`_ensure_audio` 下载时 output_dir 传 `str(DataConfig.AUDIO_DIR / task.task_id)`，并先 `mkdir(parents=True, exist_ok=True)`。**不改下载器**（`download_audio(url, output_dir)` 写 `audio.mp3` 进传入目录）。
- `cleanup_task_files()`：**不改**——它 rmtree 的是 task_dir，音频已搬走，天然保留。
- `_cleanup_stale_tasks()`：**零改动**。

### 8.3 新增音频清理

`utils/cleanup.py` 新增常量与函数，挂到 `cleanup_scheduler`：

```python
AUDIO_MAX_AGE_DAYS = 7

async def _cleanup_old_audio():
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

`cleanup_scheduler()` 循环里追加 `await _cleanup_old_audio()`。`cleanup_sync()` 同步测试版也追加。

### 8.4 标题持久化（跨 task 目录清理）

**问题**：`status.json`（含 `title`）存在 task 目录里，会被 `_cleanup_stale_tasks()` 在 1 天后删掉；但音频缓存有 7 天。第 2~7 天之间重跑时 task 目录已没，`title` 丢失，而重跑又不允许联网补标题 → 标题会退化成 "Unknown Video"。

**修复**：下载音频时在音频目录写 sidecar `data/audio/{task_id}/meta.json`，至少存 `{"title": ...}`（也可含 url/platform 便于诊断）。`_ensure_title` 的回退顺序：① 已有 `task.title` → 不动；② audio_cached 且 status.json 无标题 → 读 `meta.json` 恢复；③ 都没有 → "Unknown Video"。这样标题存活满 7 天音频窗口，与音频同生命周期被 `_cleanup_old_audio` 一并清理。

> `meta.json` 由 `_ensure_audio` 在下载成功后写入（用 `result.title or task.title`）。

## 9. 测试要点

- **首次（音频路径）**：音频落 `data/audio/{task_id}/audio.mp3`；完成时 task_dir 中间文件被清、音频保留；`TaskState` 经历 `DOWNLOADING→TRANSCRIBING→CORRECTING→SUMMARIZING→COMPLETED`。
- **第二次（音频 <7 天，reprocess）**：mock 下载器，断言 `extract_info / extract_subtitles / download_audio` **均未被调用**；transcribe/correct/summarize 重跑；`{task_id}.md` 被覆盖；`get_resume_action` 返回 `reprocess`。
- **第二次（字幕路径，return_cached）**：有笔记、无音频 → 返回缓存笔记，pipeline 不执行，下载器未被调用。
- **第二次（音频已过期，return_cached）**：音频被 TTL 清掉、笔记还在 → 返回缓存笔记，不联网重下载。
- **`get_resume_action` 四象限**：(有音频+有笔记)→reprocess；(有音频+无笔记)→reprocess；(无音频+有笔记)→return_cached；(无音频+无笔记)→full。
- **清理**：音频 >7 天删、<7 天留；`cleanup_task_files()` 后音频仍在 AUDIO_DIR；`_cleanup_old_audio` 只动 `data/audio/`，不碰 task_dir/notes。
- **重构不回归**：纯重构步骤后，原有失败重试、字幕快路径、纠错失败回退（`correction_failed`）行为不变。
- **标题持久化**：第 2~7 天（task_dir 已删、音频还在）重跑 → `_ensure_title` 从 `data/audio/{task_id}/meta.json` 恢复标题，不退化成 "Unknown Video"，且未联网。
- **迁移**：旧 task_dir 里的音频不会被新逻辑识别（一次性，重新下载即可）。

## 10. 实施顺序

1. **Step 1（纯重构，行为不变）**：拆 `_do_process` 为子函数 + 统一内存日志（`_log_mem`/`_reclaim_mem`）+ 新增 `CORRECTING` state。不加音频缓存逻辑。先补/调测试保证现有行为不回归。
2. **Step 2（音频缓存）**：在重构好的结构上叠加 `AUDIO_DIR` + `task.audio_file` 改向 + `has_cached_audio` + `_ensure_audio` 下载到新位置 + `_maybe_get_subtitle`/`_ensure_title` 的 audio_cached 跳过 + 入口决策 `get_resume_action` + handler 改造 + `_cleanup_old_audio`。

重构与功能解耦，review/回滚都干净。
