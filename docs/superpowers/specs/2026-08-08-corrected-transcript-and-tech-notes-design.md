# 设计规格：纠错转写稿 + 技术知识点总结

- **日期**：2026-08-08
- **状态**：已确认，待编写实现计划
- **范围**：单次实现可覆盖（流水线新增"纠错"步骤 + 新增交付物 + 总结提示词替换）

---

## 1. 目标与背景

VideoBot 现有流程：下载音频 → Whisper 转写（原始未纠错文本）→ LLM 总结 → 推送单份 Markdown 笔记。

本次需求两点：

1. **纠错转写稿**：音频转文字之后，把原始转写交给 LLM 纠错，产出一份独立的纠错转写稿 Markdown（`.mp3.md`），与总结笔记一并推送。纠错稿反过来喂给总结，提升总结质量。
2. **技术知识点总结**：总结笔记默认采用"技术知识点梳理"风格，把视频拆解为若干知识点，每个知识点严格包含 4 个字段（说明 / 逻辑细节 / 相关联知识点 / 完备性点评）；对非技术内容软兜底为通用笔记。

每次处理一个视频，最终推送 **2 份** Markdown 文件：
- ① 技术知识点总结笔记（`{title}_summary.md`）
- ② 纠错转写稿（`{title}.mp3.md`，仅 Whisper 路径产出）

---

## 2. 数据流与流水线

### 2.1 Whisper 路径（改造后）

```
audio → Whisper → transcript.json(原始)
       → 【纠错分块调用 LLM】→ corrected.json(断点) + notes/{id}.mp3.md(交付物)
       → summarize(读取 corrected.json) → notes/{id}.md
```

关键变化：
- 在"转录"与"总结"之间插入**纠错**步骤。
- 纠错结果落两处：
  - `data/tasks/{task_id}/corrected.json`：断点续传用，完成后随其他中间文件清理。
  - `data/notes/{task_id}.mp3.md`：持久交付物，与 `notes/{id}.md` 同目录，不清理。
- 总结读取 `corrected.json`（纠错后文本）而非原始 `transcript.json`。

### 2.2 字幕路径（不变 + 提示词替换）

有字幕时跳过 Whisper：
- **不做纠错、不产出 `.mp3.md`**（字幕本身即为干净文本）。
- 总结仍使用新技术知识点提示词（吃 `subtitle.txt`）。

理由：需求明确把纠错绑定在"音频转文字之后"；字幕是平台提供的已清洗文本，无需纠错。

### 2.3 断点续传

新增第三级中间文件，续传顺序：
```
has_subtitle? ──是──→ summarize
       │否
has_transcript? ──否──→ 下载+转写
       │是
has_corrected? ──否──→ 纠错
       │是
summarize(读 corrected)
```
失败任务重试时按现有机制逐级跳过已完成步骤。

---

## 3. 纠错步骤

### 3.1 分块策略

复用 `summarizer/llm.py` 现有的 `_split_chunks` 与 `_calc_chunk_char_limit(MAX_CONTEXT_TOKENS)`，仅做一处调整：

- **关闭 overlap**：纠错是无损拼接，overlap 会产生重复文本。纠错分块 `overlap=0`。
- **块大小复用上下文窗口**：直接采用 `_calc_chunk_char_limit(MAX_CONTEXT_TOKENS)` 算出的字符上限（≈37k 字符），与总结分块同源、同上限。`max_context_tokens` 是唯一的块大小依据。

**不限定输出长度**：纠错调用 LLM 时**不传 `max_tokens`**——请求不限定上下文/输出长度，让模型按自身能力输出完整纠错结果。这与项目既有约定一致（总结/refine 调用也已移除 `max_tokens`）。

**拼接**：各块纠错后按原始顺序直接拼接，不做 refine 合并。

### 3.2 纠错提示词

新增 `CORRECTION_SYSTEM_PROMPT`（要点）：

- 角色：专业文字校对助手，输入是 ASR（语音转文字）原始结果。
- 铁律（最高优先级）：**只纠错、不改写、不删减、不总结**；保留全部信息量、原始语序与段落结构，逐句对照修正。
- 重点修正：同音/近音错别字（如"人工只能"→"人工智能"）、专有名词/技术术语/人名/产品名、明显断句与标点缺失。
- 不确定处保留原样，禁止臆测捏造。
- 不输出解释、批注、标题或 Markdown 格式，只输出校对后的正文。
- 输出语言跟随原文：中文→简体中文，英文→英文。

调用结构沿用 `_call_with_fallback`（多模型回退），不复用总结的 `_wrap_user_content` 语言守卫（纠错要保留原文语言，不能强制中文）。

### 3.3 失败与降级

- **纠错失败**（所有模型都报错）：降级为总结吃原始 `transcript.json`，不阻断整条流水线；不产出 `.mp3.md`；`status.json` 记录 `correction_failed=true`，日志告警。
- **截断/空响应**：复用 `_extract_content` 的既有机制——响应 `finish_reason=length`（输出被截断）或 content 为空时抛 `EmptyLLMResponseError`，由 `_call_with_fallback` 自动切换下一个模型重试；全部模型都失败则触发上一条的降级。不另外加手动启发式。

---

## 4. 纠错转写稿文件（`.mp3.md`）

### 4.1 存储与命名

| 用途 | 路径 | 生命周期 |
|------|------|----------|
| 断点续传 | `data/tasks/{task_id}/corrected.json` | 完成后随中间文件清理 |
| 持久交付物 | `data/notes/{task_id}.mp3.md` | 持久，不清理 |
| 推送文件名 | `{title[:50]}.mp3.md` | 临时文件，推完即删 |

`corrected.json` 结构沿用 `transcript.json`：`{"full_text": "<纠错后全文>"}`。

推送文件名与总结（`{title[:50]}_summary.md`）通过 `.mp3` 后缀区分，表达"音频转写稿"。

### 4.2 文件内容

极简头 + 纠错正文：

```markdown
# {title}

> 来源：{url}

---

{纠错后正文}
```

### 4.3 推送（一并推送两份）

改造 `bot/handler.py` 的 `_send_note`：
- 接收交付物列表 `[(filename, content), ...]`，逐个 `save_temp_markdown` → `uploader_manager.upload`。
- Whisper 路径成功时推送 2 份：总结 + 纠错稿。
- 字幕路径 / 纠错失败 / 老任务无 `.mp3.md`：只推总结。
- `save_temp_markdown` 增加可选 `suffix` 参数（默认 `_summary`，纠错稿传 `.mp3`），避免硬编码文件名。

### 4.4 缓存命中

`get_or_create` 返回已完成的缓存任务时，推送逻辑读取磁盘：
- 总是推 `notes/{task_id}.md`。
- 若 `notes/{task_id}.mp3.md` 存在则一并推；不存在（功能上线前的老任务）则只推总结。

---

## 5. 技术知识点总结提示词

### 5.1 替换默认行为

替换 `summarizer/llm.py` 的 `SYSTEM_PROMPT`（与 `REFINE_SYSTEM_PROMPT` 同步更新），保留现有的：
- 100% 简体中文语言约束（含 `_wrap_user_content` 首/尾守卫）。
- 数学公式 LaTeX、Markdown 表格、末尾 AI 总结。
- 仅返回 Markdown、不包裹代码块。

### 5.2 新增结构

**默认模式 = 技术知识点梳理**：把视频内容拆解为若干技术知识点，**每个知识点严格输出 4 个字段**：

1. **知识点说明** — 这个知识点是什么、解决什么问题。
2. **逻辑细节** — 原理 / 机制 / 实现步骤 / 关键参数等展开。
3. **相关联知识点** — 前置 / 延伸 / 对比概念，指出关联关系。
4. **完备性点评** — 评估视频对该知识点的讲解是否完整，指出遗漏、含糊或可深化处。

### 5.3 软兜底（非技术内容）

明显非技术类视频（访谈 / Vlog / 新闻 / 纯闲聊）找不到有意义的技术知识点时：切换为通用结构化笔记，**不强行凑知识点**，在笔记开头注明一行：

> 本视频非技术类，采用通用笔记格式。

由模型在单次调用内自行判断是否触发兜底，不增加额外调用。

---

## 6. TaskManager 改动

`core/task_manager.py`：

- `Task` 新增属性 `corrected_file -> Path`（`task_dir / "corrected.json"`）。
- 新增方法：`has_corrected(task)`、`save_corrected(task, text)`、`load_corrected(task)`，实现与 `transcript` 系列对称。
- `can_resume_from_summarization` 更新为 `has_corrected or has_subtitle`（纠错成为 summarize 的前置）。**注意语义变化**：原义为"有 transcript 即可直接总结"，新流程下变为"有 corrected（或字幕）才能直接进入总结"——`audio + transcript` 但无 `corrected` 的任务不再算"可从总结阶段续传"，需先补纠错。这会翻转 `test_task_manager.py` / `test_pipeline_resume.py` 里现有断言（见第 8 节）。
- `cleanup_task_files` 行为不变（`corrected.json` 随目录一起被清掉，仅保留 `status.json`）。
- `status.json` 新增可选字段 `correction_failed: bool`（默认不写或 false）。

---

## 7. Pipeline 改动

`core/pipeline.py::_do_process`：

1. 转录后（已有 `audio_text`），若 `not has_corrected(task)`：执行纠错。
2. 纠错：分块 → 逐块调用 LLM → 拼接 → `save_corrected` + 写 `notes/{id}.mp3.md`。
3. 纠错失败：try/except 降级，`text_content = audio_text`（原始），不写 `.mp3.md`，记 `correction_failed`。
4. 总结阶段：`text_content = corrected_text or subtitle_text or audio_text`（优先级：纠错稿 > 字幕 > 原始转写）。

LLMSummarizer 新增 `correct(text) -> str` 方法，封装分块纠错 + 拼接 + 回退，与 `summarize` 平级。

---

## 8. 测试计划

| 测试文件 | 覆盖点 |
|----------|--------|
| `tests/test_corrector.py`（新增） | 分块（overlap=0）、逐块纠错、拼接顺序、截断 warning；mock LLM |
| `tests/test_task_manager.py` | `has/save/load_corrected`；续传三级跳；`cleanup` 清 `corrected.json` 留 `status.json`；**更新**现有 `can_resume_from_summarization` 断言（`audio+transcript` 不再算可续传至总结，需 `audio+transcript+corrected`） |
| `tests/test_summarizer.py` | 新 `SYSTEM_PROMPT`/`REFINE` 含 4 字段 + 兜底条款；`correct()` 方法；纠错失败抛 `RuntimeError` |
| `tests/test_pipeline_resume.py` | 纠错稿缺失→执行纠错；存在→跳过；纠错失败→降级原始转写；字幕路径不纠错；**更新**现有 State 1/State 2 断言以反映新续传语义 |
| `tests/test_formatter.py`（新增/扩展） | `save_temp_markdown(suffix=...)`；多文件推送；老任务无 `.mp3.md` 只推总结 |

集成测试 `tests/test_integration.py` 视情况补一条端到端用例（需网络/模型，默认不强求）。

---

## 9. 不做的事（YAGNI）

- 不为纠错/技术知识点加配置开关（默认全部开启；后续需要再加 `[pipeline] correct_transcript` 等）。
- 不对纠错截断做自动重试/重切分。
- 不让字幕路径产出 `.mp3.md`。
- 不改动现有内存管理、下载器、Whisper 转写逻辑。
- 不引入新的依赖。

---

## 10. 风险

- **模型实际输出能力**：请求不传 `max_tokens`，由 `_extract_content` 在 `finish_reason=length`/空 content 时触发模型回退；若所有配置模型对长纠错块都频繁截断，可在实现时把纠错块调小（如复用 `refine_chunk_char_limit`），无需引入新的输出上限常量。
- **纠错块边界**：句子被切断时可能影响个别纠错质量；`_split_chunks` 已优先在换行处切分，影响可控。
- **老任务兼容**：功能上线前的已完成任务无 `.mp3.md`，推送时按"只推总结"处理，不报错。
