# 纠错转写稿 + 技术知识点总结 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 Whisper 转录与总结之间插入 LLM 纠错步骤，产出独立的纠错转写稿 `.mp3.md` 并与总结笔记一并推送；同时把总结提示词默认升级为"技术知识点梳理（四字段 + 软兜底）"。

**架构：** 流水线变为 `转录(原始) → 纠错 → 总结(吃纠错稿)`。纠错稿落两处：`data/tasks/{id}/corrected.json`（断点续传，完成后清理）+ `data/notes/{id}.mp3.md`（持久交付物）。纠错失败时降级用原始转写、不阻断流水线。`TaskManager` 增加 corrected 三件套；`LLMSummarizer` 增加 `correct()`；`SYSTEM_PROMPT`/`REFINE_SYSTEM_PROMPT` 替换为技术知识点版；`formatter` 支持多文件；`handler` 一并推送两份。

**技术栈：** Python 3.11、faster-whisper、OpenAI 兼容 SDK（多模型回退）、aiogram、pytest。

**对应规格：** `docs/superpowers/specs/2026-08-08-corrected-transcript-and-tech-notes-design.md`

---

## 文件结构

**修改：**
- `core/task_manager.py` — 新增 `corrected_file` 属性、`has/save/load_corrected`、`correction_failed` 字段（含向后兼容加载）、更新 `can_resume_from_summarization` 语义、`update_state` 支持 `correction_failed`。
- `summarizer/llm.py` — 新增 `CORRECTION_SYSTEM_PROMPT`、`correct()` 方法；替换 `SYSTEM_PROMPT`/`REFINE_SYSTEM_PROMPT` 为技术知识点四字段 + 软兜底版。
- `bot/formatter.py` — `save_temp_markdown` 增加 `suffix` 参数；新增 `build_transcript_markdown`、`collect_deliverables`。
- `core/pipeline.py` — 在转录与总结之间插入纠错步骤，写 `notes/{id}.mp3.md`，纠错失败降级，`text_content` 优先级改为 `corrected > subtitle > raw`。
- `bot/handler.py` — `_send_note` 改为多文件推送，读取 `notes/{id}.mp3.md`。

**新建：**
- `tests/test_corrector.py` — 纠错分块/拼接/提示词/输出上限。

**修改测试：**
- `tests/test_task_manager.py`、`tests/test_summarizer.py`、`tests/test_pipeline_resume.py`、`tests/test_formatter.py`。

**依赖关系：** 任务 1、2、3、4 互相独立可并行审查；任务 5 依赖 1+2+4；任务 6 依赖 4+5。

---

## 任务 1：TaskManager 纠错稿支持

**文件：**
- 修改：`core/task_manager.py`
- 测试：`tests/test_task_manager.py`

- [ ] **步骤 1：编写失败测试（新增 + 更新现有断言）**

在 `tests/test_task_manager.py` 末尾、`if __name__ == "__main__":` 之前新增：

```python
def test_corrected_save_load():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        assert not mgr.has_corrected(task)
        mgr.save_corrected(task, "纠错后的文本")
        assert mgr.has_corrected(task)
        assert mgr.load_corrected(task) == "纠错后的文本"
    finally:
        teardown_temp_data(tmp)


def test_resume_logic_now_requires_corrected():
    """新语义：仅有 transcript 不再算可续传至总结，需要 corrected（或 subtitle）。"""
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        mgr.save_transcript(task, "raw text")
        assert not mgr.can_resume_from_summarization(task), "仅有 transcript 不应能续传至总结"
        mgr.save_corrected(task, "corrected text")
        assert mgr.can_resume_from_summarization(task), "有 corrected 后应能续传至总结"
    finally:
        teardown_temp_data(tmp)


def test_load_backward_compat_without_correction_failed():
    """老 status.json 没有 correction_failed 字段时，加载后默认 False。"""
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        import json as _json
        data = _json.loads(task.status_file.read_text(encoding="utf-8"))
        del data["correction_failed"]
        task.status_file.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        loaded = mgr.get_task("abc123")
        assert loaded is not None
        assert loaded.correction_failed is False
    finally:
        teardown_temp_data(tmp)


def test_update_state_persists_correction_failed():
    tmp = setup_temp_data(None)
    try:
        mgr = TaskManager()
        task = mgr.create_task("abc123", "https://example.com", "youtube")
        mgr.update_state(task, TaskState.SUMMARIZING, correction_failed=True)
        assert task.correction_failed is True
        assert mgr.get_task("abc123").correction_failed is True
    finally:
        teardown_temp_data(tmp)
```

同时**更新现有 `test_resume_logic`**，把"transcript 存在即可续传"改为"需要 corrected"：

```python
        # Transcript exists but NOT enough to resume summarization (need corrected)
        mgr.save_transcript(task, "text")
        assert not mgr.can_resume_from_summarization(task)

        # Corrected exists -> can resume from summarization
        mgr.save_corrected(task, "corrected text")
        assert mgr.can_resume_from_summarization(task)
```

并在 `if __name__ == "__main__":` 调用列表追加四个新测试函数名。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_task_manager.py -v`
预期：FAIL，报错 `AttributeError: 'Task' object has no attribute 'corrected_file'` / `'TaskManager' object has no attribute 'has_corrected'`。

- [ ] **步骤 3：实现 TaskManager 改动**

在 `core/task_manager.py` 的 `Task` 数据类中新增字段与属性（紧跟 `model_name` 字段之后、`task_dir` 属性之前）：

```python
    model_name: Optional[str] = None
    correction_failed: bool = False
```

在 `Task` 类的属性区（`transcript_file` 之后）新增：

```python
    @property
    def corrected_file(self) -> Path:
        return self.task_dir / "corrected.json"
```

在 `TaskManager` 中，把 `can_resume_from_summarization` 改为：

```python
    def can_resume_from_summarization(self, task: Task) -> bool:
        return self.has_corrected(task) or self.has_subtitle(task)
```

在 `load_transcript` 方法之后新增三个方法（与 transcript 系列对称）：

```python
    def has_corrected(self, task: Task) -> bool:
        return task.corrected_file.exists()

    def save_corrected(self, task: Task, text: str):
        data = {"full_text": text}
        task.corrected_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_corrected(self, task: Task) -> Optional[str]:
        if not task.corrected_file.exists():
            return None
        data = json.loads(task.corrected_file.read_text(encoding="utf-8"))
        return data.get("full_text")
```

在 `update_state` 中新增 `correction_failed` 处理（紧跟 `error` 分支之后）：

```python
        if "correction_failed" in kwargs:
            task.correction_failed = kwargs["correction_failed"]
```

在 `_load` 中新增向后兼容（紧跟 `model_name` 的 setdefault 之后）：

```python
        data.setdefault("model_name", None)
        data.setdefault("correction_failed", False)
        return Task(**data)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_task_manager.py -v`
预期：PASS（全部用例）。

- [ ] **步骤 5：Commit**

```bash
git add core/task_manager.py tests/test_task_manager.py
git commit -m "feat: TaskManager 新增纠错稿(corrected)持久化与续传支持"
```

---

## 任务 2：LLMSummarizer 纠错能力（correct + 提示词 + 分块）

**文件：**
- 修改：`summarizer/llm.py`
- 测试：`tests/test_corrector.py`（新建）

- [ ] **步骤 1：编写失败测试**

新建 `tests/test_corrector.py`：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
from config import ModelConfig


def _make_summarizer(mock_client):
    with patch("summarizer.llm.AIConfig.load_models") as mock_load, \
         patch("summarizer.llm.OpenAI") as mock_openai_cls:
        mock_load.return_value = [ModelConfig(name="test-model", url="http://test", key="k")]
        mock_client.models.list.return_value = []
        mock_openai_cls.return_value = mock_client
        from summarizer.llm import LLMSummarizer
        return LLMSummarizer()


def _resp(content):
    m = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    m.choices = [choice]
    m.usage = None
    return m


def test_correct_single_chunk_returns_content():
    mc = MagicMock()
    mc.chat.completions.create.return_value = _resp("纠错后的文本")
    s = _make_summarizer(mc)
    assert s.correct("原始文本") == "纠错后的文本"
    mc.chat.completions.create.assert_called_once()


def test_correct_uses_correction_system_prompt():
    mc = MagicMock()
    mc.chat.completions.create.return_value = _resp("x")
    s = _make_summarizer(mc)
    s.correct("text")
    from summarizer.llm import CORRECTION_SYSTEM_PROMPT
    call = mc.chat.completions.create.call_args
    assert call.kwargs["messages"][0]["content"] == CORRECTION_SYSTEM_PROMPT


def test_correct_does_not_set_max_tokens():
    """请求不限定上下文长度：纠错调用不传 max_tokens。"""
    mc = MagicMock()
    mc.chat.completions.create.return_value = _resp("x")
    s = _make_summarizer(mc)
    s.correct("text")
    assert "max_tokens" not in mc.chat.completions.create.call_args.kwargs


def test_correct_empty_returns_empty_without_call():
    mc = MagicMock()
    s = _make_summarizer(mc)
    assert s.correct("") == ""
    mc.chat.completions.create.assert_not_called()


def test_correct_multi_chunk_concatenates_in_order():
    mc = MagicMock()
    mc.chat.completions.create.side_effect = [_resp("A"), _resp("B")]
    s = _make_summarizer(mc)
    s.correction_chunk_char_limit = 5  # 强制分块
    result = s.correct("0123456789")   # 10 chars -> 2 chunks
    assert result == "AB"


def test_correct_no_overlap_reconstructs_original():
    """overlap=0 + 逐字回显 => 拼接结果应能还原原文，无重复。"""
    def echo(*a, **kw):
        body = kw["messages"][1]["content"].split("\n", 1)[1]
        return _resp(body)
    mc = MagicMock()
    mc.chat.completions.create.side_effect = echo
    s = _make_summarizer(mc)
    s.correction_chunk_char_limit = 5
    text = "abcdefghij"
    assert s.correct(text) == text


if __name__ == "__main__":
    test_correct_single_chunk_returns_content()
    test_correct_uses_correction_system_prompt()
    test_correct_does_not_set_max_tokens()
    test_correct_empty_returns_empty_without_call()
    test_correct_multi_chunk_concatenates_in_order()
    test_correct_no_overlap_reconstructs_original()
    print("All corrector tests passed!")
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_corrector.py -v`
预期：FAIL，报错 `ImportError: cannot import name 'CORRECTION_SYSTEM_PROMPT'` / `AttributeError: 'LLMSummarizer' object has no attribute 'correct'`。

- [ ] **步骤 3：实现纠错提示词与方法**

在 `REFINE_SYSTEM_PROMPT` 定义之后新增纠错提示词（**不新增任何 max_tokens / 输出上限常量**——请求不限定上下文长度；截断与空响应交给既有 `_extract_content` + `_call_with_fallback` 模型回退处理）：

```python
CORRECTION_SYSTEM_PROMPT = """你是一个专业的文字校对助手。输入是一段视频语音转文字（ASR）的原始结果，可能包含错别字、同音字错误、标点缺失、专有名词错误等问题。

铁律（最高优先级，必须严格遵守）：
1. **只纠错，不改写、不删减、不总结、不扩写**：保留全部信息量、原始语序与段落结构，逐句对照修正。
2. 重点修正：同音/近音错别字（如"人工只能"→"人工智能"）、专有名词/技术术语/人名/产品名、明显断句与标点缺失。
3. 不确定处保留原样，禁止臆测或捏造内容。
4. 不要输出任何解释、批注、标题或 Markdown 格式符号，只输出校对后的正文。
5. 输出语言跟随原文：中文输出简体中文，英文输出英文，不要翻译。

仅返回校对后的纯文本正文。
"""
```

在 `LLMSummarizer.__init__` 中（`self.refine_chunk_char_limit = ...` 那一行之后）新增纠错块上限（**复用上下文窗口，与总结同源**，仅 overlap 不同）：

```python
        # 纠错块大小复用上下文窗口（与总结同源），仅 overlap 不同
        self.correction_chunk_char_limit = self.chunk_char_limit
        logger.info(f"Correction chunk size: {self.correction_chunk_char_limit} chars")
```

在 `LLMSummarizer` 类中（`summarize` 方法之前）新增 `correct` 与 `_correct_single`：

```python
    def correct(self, text: str) -> str:
        """对 ASR 原始转写做纠错。无损：保留全部内容，仅修正错误。
        复用上下文窗口分块、overlap=0，逐块纠错后按顺序拼接。
        截断/空响应由 _extract_content 抛 EmptyLLMResponseError，经 _call_with_fallback 自动切模型。"""
        if not text:
            return text

        chunks = _split_chunks(text, chunk_size=self.correction_chunk_char_limit, overlap=0)
        logger.info(f"Correcting transcript ({len(text)} chars, {len(chunks)} chunk(s))")

        parts = []
        for i, chunk in enumerate(chunks, start=1):
            corrected = self._correct_single(chunk)
            parts.append(corrected)
            logger.info(f"Corrected chunk {i}/{len(chunks)} ({len(corrected)} chars)")

        return "".join(parts)

    def _correct_single(self, chunk: str) -> str:
        def _fn(client, model):
            user_content = f"原始转写文本：\n{chunk}"
            kwargs = dict(temperature=0.1)
            if _is_reasoning_model(model):
                kwargs["reasoning_effort"] = "low"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                **kwargs,
            )
            return _extract_content(response, "correct")

        return self._call_with_fallback(_fn, "correct")
```

> 说明：`_split_chunks(..., overlap=0)` 复用现有分块函数；其 `start = split_pos - overlap if ... else split_pos` 在 overlap=0 时退化为 `start = split_pos`，正常推进、无重复。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_corrector.py -v`
预期：PASS（6 个用例）。

- [ ] **步骤 5：Commit**

```bash
git add summarizer/llm.py tests/test_corrector.py
git commit -m "feat: LLMSummarizer 新增无损纠错 correct()，复用上下文窗口分块、不传 max_tokens"
```

---

## 任务 3：技术知识点总结提示词（替换默认总结行为）

**文件：**
- 修改：`summarizer/llm.py`（`SYSTEM_PROMPT`、`REFINE_SYSTEM_PROMPT`）
- 测试：`tests/test_summarizer.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/test_summarizer.py` 末尾、`if __name__ == "__main__":` 之前新增（无需 mock，常量为模块级）：

```python
def test_system_prompt_has_four_knowledge_fields():
    from summarizer.llm import SYSTEM_PROMPT
    for field in ("知识点说明", "逻辑细节", "相关联知识点", "完备性点评"):
        assert field in SYSTEM_PROMPT, f"SYSTEM_PROMPT 缺少字段：{field}"


def test_system_prompt_has_soft_fallback():
    from summarizer.llm import SYSTEM_PROMPT
    assert "非技术" in SYSTEM_PROMPT
    assert "通用笔记格式" in SYSTEM_PROMPT


def test_refine_prompt_has_four_knowledge_fields():
    from summarizer.llm import REFINE_SYSTEM_PROMPT
    for field in ("知识点说明", "逻辑细节", "相关联知识点", "完备性点评"):
        assert field in REFINE_SYSTEM_PROMPT, f"REFINE_SYSTEM_PROMPT 缺少字段：{field}"


def test_refine_prompt_has_soft_fallback():
    from summarizer.llm import REFINE_SYSTEM_PROMPT
    assert "非技术" in REFINE_SYSTEM_PROMPT
```

并把上述四个函数名追加到 `if __name__ == "__main__":` 调用列表。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_summarizer.py -v -k "knowledge_fields or soft_fallback"`
预期：FAIL（现有提示词不含"知识点说明"等字段）。

- [ ] **步骤 3：替换两个提示词**

在 `summarizer/llm.py` 中，把现有 `SYSTEM_PROMPT = """..."""` 整体替换为：

```python
SYSTEM_PROMPT = """你是一个专业的技术笔记助手，默认采用「技术知识点梳理」模式整理视频转录内容。

⚠️ 语言要求（最高优先级，必须严格遵守）：
- 笔记正文必须 **100% 使用简体中文** 撰写，严禁用英文书写整句或整段。
- 即使转录原文为英文、繁体中文或中英混杂，也要全部翻译/转换为简体中文。
- 仅允许在以下情况保留英文：专有名词、技术术语、品牌名称、人名、代码、数学公式与变量。

输出说明：
- 仅返回最终的 **Markdown 内容**。
- **不要**将输出包裹在代码块中。

【默认模式：技术知识点梳理】
将视频内容拆解为若干技术知识点，逐个梳理。**每个知识点必须严格包含以下 4 个字段**，缺一不可：

1. **知识点说明**：这个知识点是什么、解决什么问题、用在什么场景。
2. **逻辑细节**：原理、机制、实现步骤、关键参数、数据流向等展开说明。
3. **相关联知识点**：与该知识点相关的前置/延伸/对比概念，并指出它们的关联关系。
4. **完备性点评**：评估视频对该知识点的讲解是否完整，明确指出遗漏、含糊或可深化之处。

建议每个知识点用二/三级标题分隔，4 个字段作为子项；结构化对比信息可用 Markdown 表格呈现。

【软兜底：非技术内容】
如果视频内容明显不是技术类（如访谈、Vlog、新闻、纯闲聊），找不到有意义的技术知识点，则**不要强行凑知识点**，切换为通用结构化笔记，并在笔记最开头注明一行：
> 本视频非技术类，采用通用笔记格式。

其他规则：
1. **完整信息**：记录尽可能多的相关细节。
2. **纠错**：语音转文字可能产生错别字或同音错误，需根据上下文自动修正。
3. **去除无关内容**：省略广告、填充词、问候语。
4. **保留关键细节**：保留重要事实、示例、结论和建议。
5. 视频中提及的数学公式必须保留，并以 LaTeX 语法形式呈现。
6. **Markdown 表格**：对核心观点、关键要点、对比分析、分类信息等结构化内容使用表格。
7. 在笔记末尾添加一段专业的 **AI 总结**，简要概括整个视频的核心内容。
"""
```

把现有 `REFINE_SYSTEM_PROMPT = """..."""` 整体替换为：

```python
REFINE_SYSTEM_PROMPT = """你是一个专业的技术笔记助手，正在分阶段处理一段长视频转录内容。

你已有前一部分内容的笔记，现在需要结合新提供的转录片段，更新和完善笔记。

⚠️ 语言要求（最高优先级，必须严格遵守）：
- 笔记正文必须 **100% 使用简体中文** 撰写，严禁用英文书写整句或整段。
- 即使转录原文为英文、繁体中文或中英混杂，也要全部翻译/转换为简体中文。
- 仅允许在以下情况保留英文：专有名词、技术术语、品牌名称、人名、代码、数学公式与变量。
- **不要**将输出包裹在代码块中。

笔记结构（技术知识点梳理模式）：
- 把内容拆解为若干技术知识点，**每个知识点严格包含 4 个字段**：① 知识点说明 ② 逻辑细节 ③ 相关联知识点 ④ 完备性点评。
- 若内容明显非技术类，切换为通用结构化笔记，并在开头注明「本视频非技术类，采用通用笔记格式」，不要强行凑知识点。

规则：
1. 保留已有笔记中的所有重要信息，不要丢弃。最好原始保留已有内容，除非新片段提供了更准确或更详细的信息。
2. 将新片段中的重要信息整合进对应知识点位置；遇到新知识点就新增条目。
3. 如果新片段与已有内容重复，合并而非简单堆叠。
4. **纠错**：语音转文字可能产生错别字或同音错误，需根据上下文自动修正。
5. 去除广告、填充词、问候语等无关内容。
6. 视频中提及的数学公式必须保留，并以 LaTeX 语法形式呈现。
7. **Markdown 表格**：对结构化内容使用表格。
8. 在笔记末尾保留/更新一段专业的 **AI 总结**。
"""
```

> 不要改动 `LANGUAGE_REQUIREMENT`、`LANGUAGE_GUARD_HEAD/TAIL`、`_wrap_user_content`——它们继续作为 user 消息首尾守卫注入。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_summarizer.py -v`
预期：PASS（含原有 `test_summarize`/`test_summarize_uses_config` 与 4 个新用例）。

- [ ] **步骤 5：Commit**

```bash
git add summarizer/llm.py tests/test_summarizer.py
git commit -m "feat: 总结提示词默认升级为技术知识点梳理（四字段+软兜底）"
```

---

## 任务 4：Formatter 多文件支持

**文件：**
- 修改：`bot/formatter.py`
- 测试：`tests/test_formatter.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/test_formatter.py` 中，把 `from bot.formatter import build_markdown, save_temp_markdown` 改为：

```python
from bot.formatter import build_markdown, save_temp_markdown, build_transcript_markdown, collect_deliverables
```

在文件末尾、`if __name__ == "__main__":` 之前新增：

```python
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
```

把三个新函数名追加到 `if __name__ == "__main__":` 调用列表。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_formatter.py -v`
预期：FAIL，`ImportError: cannot import name 'build_transcript_markdown'`。

- [ ] **步骤 3：实现 formatter 改动**

把 `bot/formatter.py` 的 `save_temp_markdown` 改为接受 `suffix`：

```python
def save_temp_markdown(title: str, content: str, suffix: str = "_summary") -> str:
    """Save markdown to a temp file and return path. Caller should clean up."""
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
    safe_name = safe_name[:80] + f"{suffix}.md"
    temp_dir = tempfile.mkdtemp(prefix="videobot_md_")
    path = os.path.join(temp_dir, safe_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
```

在 `save_temp_markdown` 之后新增两个函数：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_formatter.py -v`
预期：PASS（含原有用例 + 3 个新用例）。

- [ ] **步骤 5：Commit**

```bash
git add bot/formatter.py tests/test_formatter.py
git commit -m "feat: formatter 支持自定义后缀、纠错稿头与多文件交付物收集"
```

---

## 任务 5：Pipeline 插入纠错步骤 + 写 .mp3.md + 失败降级

**依赖：** 任务 1（TaskManager）、任务 2（correct()）、任务 4（build_transcript_markdown）

**文件：**
- 修改：`core/pipeline.py`、`tests/test_pipeline_resume.py`

> **重要：** 新纠错步骤会让现有 4 个 pipeline 测试的 mock 链失效（mock summarizer 现在必须有 `.correct`），且 `test_intermediate_file_states` 的 State 2 断言会翻转。本任务一并更新。

- [ ] **步骤 1：编写/更新失败测试**

**1a. 更新 `tests/test_pipeline_resume.py` 中所有 mock summarizer，补 `.correct` 配置：**

在 `test_resume_from_transcription` 的 Phase 2（`mock_summarizer = mock.MagicMock()` 之后）补：

```python
        mock_summarizer.correct.return_value = "Corrected transcript."
```

在 `test_resume_from_summarization` 的 Phase 1（`mock_summarizer_fail = mock.MagicMock()` 之后）补：

```python
        mock_summarizer_fail.correct.return_value = "Corrected transcript."
```

在 `test_multi_step_resume` 的 Phase 2（`mock_s2 = mock.MagicMock()` 之后）补：

```python
        mock_s2.correct.return_value = "Corrected transcript."
```

**1b. 更新 `test_intermediate_file_states` 的 State 2**（现有断言 `assert mgr.can_resume_from_summarization(task)` 在仅有 transcript 时已不再成立）。把 State 2 块改为：

```python
        # State 2: After transcription (simulate transcript file)
        task_manager.save_transcript(task, "Transcribed content.")
        task_manager.update_state(task, TaskState.FAILED, error="summarization crash")

        assert task_manager.has_audio(task)
        assert task_manager.has_transcript(task)
        assert task_manager.can_resume_from_transcription(task)
        assert not task_manager.can_resume_from_summarization(task), "仅有 transcript 不应能续传至总结"
        print("  State 2: audio + transcript -> still needs correction ✓")

        # State 2b: After correction (simulate corrected file)
        task_manager.save_corrected(task, "Corrected content.")
        assert task_manager.can_resume_from_summarization(task)
        print("  State 2b: + corrected -> can_resume_from_summarization ✓")
```

**1c. 新增纠错失败降级测试**（在 `test_intermediate_file_states` 之后）：

```python
def test_correction_failure_falls_back_to_raw():
    """纠错抛错时，流水线降级用原始转写总结、不写 .mp3.md、标记 correction_failed。"""
    tmp = tempfile.mkdtemp(prefix="test_corr_fail_")
    _setup(tmp)
    try:
        from config import DataConfig
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcript.return_value = "RAW TRANSCRIPT TEXT"
        pipeline._transcriber = mock_transcriber

        mock_summarizer = mock.MagicMock()
        mock_summarizer.correct.side_effect = RuntimeError("all models failed")
        mock_summarizer.summarize.return_value = "# Summary"
        pipeline._summarizer = mock_summarizer

        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is not None, "纠错失败不应阻断流水线"
        assert task.state == TaskState.COMPLETED
        assert task.correction_failed is True
        # 总结吃的是原始转写（未纠错）
        mock_summarizer.summarize.assert_called_once()
        summarize_args = mock_summarizer.summarize.call_args
        text_arg = summarize_args.args[1] if len(summarize_args.args) > 1 else summarize_args.args[0]
        assert text_arg == "RAW TRANSCRIPT TEXT"
        # 未产出 .mp3.md
        assert not (DataConfig.NOTES_DIR / f"{task.task_id}.mp3.md").exists()
        print("test_correction_failure_falls_back_to_raw PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_correction_writes_mp3md_and_feeds_summary():
    """正常纠错：写出 notes/{id}.mp3.md，总结吃纠错稿而非原始转写。"""
    tmp = tempfile.mkdtemp(prefix="test_corr_ok_")
    _setup(tmp)
    try:
        from config import DataConfig
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcript.return_value = "RAW TEXT"
        pipeline._transcriber = mock_transcriber

        mock_summarizer = mock.MagicMock()
        mock_summarizer.correct.return_value = "CORRECTED TEXT"
        mock_summarizer.summarize.return_value = "# Summary"
        pipeline._summarizer = mock_summarizer

        with mock.patch("core.pipeline.get_downloader", return_value=_mock_downloader()):
            result = pipeline.process(url, task)

        assert result is not None
        assert task.state == TaskState.COMPLETED
        # .mp3.md 已写入并含纠错正文
        mp3 = DataConfig.NOTES_DIR / f"{task.task_id}.mp3.md"
        assert mp3.exists(), "纠错稿 .mp3.md 应已写入"
        assert "CORRECTED TEXT" in mp3.read_text(encoding="utf-8")
        # 总结吃的是纠错稿
        text_arg = mock_summarizer.summarize.call_args.args[1]
        assert text_arg == "CORRECTED TEXT"
        print("test_correction_writes_mp3md_and_feeds_summary PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_subtitle_path_skips_correction():
    """字幕路径不纠错、不写 .mp3.md，但仍走总结。"""
    tmp = tempfile.mkdtemp(prefix="test_sub_")
    _setup(tmp)
    try:
        from config import DataConfig
        url = BILIBILI_URL
        task_manager = TaskManager()
        pipeline = Pipeline()
        task, _ = task_manager.get_or_create(url, "bilibili")

        # downloader 返回字幕、不下载音频
        dl = mock.MagicMock()
        dl.extract_info.return_value = {"title": "Subtitled Video"}
        dl.extract_subtitles.return_value = "SUBTITLE TEXT"

        mock_summarizer = mock.MagicMock()
        mock_summarizer.summarize.return_value = "# Summary"
        pipeline._summarizer = mock_summarizer

        with mock.patch("core.pipeline.get_downloader", return_value=dl):
            result = pipeline.process(url, task)

        assert result is not None
        mock_summarizer.correct.assert_not_called()
        mock_summarizer.summarize.assert_called_once()
        assert mock_summarizer.summarize.call_args.args[1] == "SUBTITLE TEXT"
        assert not (DataConfig.NOTES_DIR / f"{task.task_id}.mp3.md").exists()
        print("test_subtitle_path_skips_correction PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

把三个新函数名追加到 `if __name__ == "__main__":` 调用列表。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_pipeline_resume.py -v`
预期：FAIL（pipeline 还没插纠错；`correct` 未被调用、`.mp3.md` 不存在、降级断言不成立）。

- [ ] **步骤 3：实现 pipeline 纠错步骤**

在 `core/pipeline.py::_do_process` 中，定位到 Step 4 转录块（`if not subtitle_text:` 内、`audio_text = self.task_manager.load_transcript(task)` 的 `else` 分支之后、`# Step 6: Summarize` 之前）。在转录块结束之后、Summarize 之前，插入纠错块。

先把函数体内靠前的 `audio_text = None` 那一行下方新增 `corrected_text = None`（确保字幕路径也有该变量）：

```python
        # Step 4: Download audio if no subtitles
        audio_text = None
        corrected_text = None
```

然后在转录块（`if not subtitle_text:` 整块）之后、`# Step 6: Summarize` 之前插入：

```python
        # Step 5.5: Correct transcript (only when audio was transcribed, not subtitles)
        if not subtitle_text and audio_text:
            if not self.task_manager.has_corrected(task):
                gc.collect()
                mem_before_corr = _mem_mb()
                logger.info(f"[MEM] Before correction [mem: {mem_before_corr:.0f}MB]")
                try:
                    corrected_text = self.summarizer.correct(audio_text)
                    self.task_manager.save_corrected(task, corrected_text)
                    from bot.formatter import build_transcript_markdown
                    mp3_md = build_transcript_markdown(task.title or "video", url, corrected_text)
                    mp3_md_path = self.data_config.NOTES_DIR / f"{task.task_id}.mp3.md"
                    mp3_md_path.write_text(mp3_md, encoding="utf-8")
                    logger.info(f"Corrected transcript ({len(corrected_text)} chars) -> {mp3_md_path}")
                    del corrected_text
                    gc.collect()
                    _malloc_trim()
                    logger.info(f"[MEM] After correction+trim [mem: {_mem_mb():.0f}MB]")
                except Exception as e:
                    logger.warning(
                        f"Transcript correction failed, falling back to raw transcript: {e}",
                        exc_info=True,
                    )
                    self.task_manager.update_state(task, task.state, correction_failed=True)
                    corrected_text = None
            else:
                corrected_text = self.task_manager.load_corrected(task)
```

然后把 `text_content` 的来源改为优先纠错稿。找到：

```python
        text_content = subtitle_text or audio_text or ""
```

替换为：

```python
        text_content = corrected_text or subtitle_text or audio_text or ""
```

> 优先级：纠错稿 > 字幕 > 原始转写。字幕路径 `corrected_text=None` → 走字幕；纠错失败 `corrected_text=None` → 走原始转写；正常纠错 → 走纠错稿。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_pipeline_resume.py -v`
预期：PASS（4 个更新后的原用例 + 3 个新用例）。

- [ ] **步骤 5：跑全量回归**

运行：`python -m pytest tests/ -v --ignore=tests/test_integration.py`
预期：PASS（任务 1/2/3/4 的用例 + 本任务的 pipeline 用例全部通过）。

- [ ] **步骤 6：Commit**

```bash
git add core/pipeline.py tests/test_pipeline_resume.py
git commit -m "feat: pipeline 插入纠错步骤，产出 .mp3.md，失败降级原始转写"
```

---

## 任务 6：Handler 一并推送总结 + 纠错稿

**依赖：** 任务 4（collect_deliverables/save_temp_markdown suffix）、任务 5（pipeline 写出 `notes/{id}.mp3.md`）

**文件：**
- 修改：`bot/handler.py`
- 测试：`tests/test_formatter.py`（`collect_deliverables` 已在任务 4 覆盖；本任务无新单测，靠集成验证）

> handler 涉及 aiogram 难以单测，故把可测逻辑 `collect_deliverables` 抽到 formatter（任务 4 已测）。本任务只改薄薄的推送循环。

- [ ] **步骤 1：改造 `_send_note` 支持多文件**

在 `bot/handler.py` 顶部 import 区，把：

```python
from bot.formatter import build_markdown, save_temp_markdown
```

改为：

```python
from bot.formatter import save_temp_markdown, collect_deliverables
```

把整个 `_send_note` 函数替换为：

```python
async def _send_note(message: types.Message, title: str, platform: str, url: str, content: str, model_name: Optional[str] = None, task_id: Optional[str] = None):
    """推送总结笔记；若 notes/{task_id}.mp3.md 存在则一并推送纠错稿。"""
    deliverables = collect_deliverables(
        title, platform, url, content, model_name, task_id, pipeline.data_config.NOTES_DIR
    )

    # Telegram 发送器：用实际文件名，对总结/纠错稿通用
    tg = uploader_manager.get_telegram_uploader()
    if tg:
        async def _send(title_, fp):
            doc = types.FSInputFile(fp, filename=os.path.basename(fp))
            await message.answer_document(doc, caption=f"{title_}", parse_mode=None)
        tg.set_sender(_send)

    success_names, failed_items, skipped_names = [], [], []
    for suffix, deliverable_content in deliverables:
        file_path = save_temp_markdown(title, deliverable_content, suffix=suffix)
        try:
            results = await uploader_manager.upload(file_path, title)
            for r in results:
                if r.success:
                    success_names.append(f"{r.uploader}({suffix})")
                elif r.message == "not enabled":
                    skipped_names.append(r.uploader)
                else:
                    failed_items.append(f"  - {r.uploader}({suffix}): {r.message}")
        finally:
            try:
                os.remove(file_path)
                os.rmdir(os.path.dirname(file_path))
            except OSError:
                pass

    parts = []
    if success_names:
        parts.append(f"Uploaded via: {', '.join(success_names)}")
    if skipped_names:
        parts.append(f"Skipped (disabled): {', '.join(skipped_names)}")
    if failed_items:
        parts.append("Failed:\n" + "\n".join(failed_items))

    if failed_items:
        await message.answer("\n".join(parts))
    elif skipped_names and not success_names:
        await message.answer("No uploaders available. " + "\n".join(parts))
```

- [ ] **步骤 2：更新两处 `_send_note` 调用，传入 `task_id`**

在 `_process_single_url` 的**缓存命中分支**：

```python
            await _send_note(message, task.title or "video", task.platform, url, cached, model_name, task_id=task.task_id)
```

在 `_process_single_url` 的**新处理成功分支**：

```python
                await _send_note(message, result.title, platform, url, result.content, result.model_name, task_id=task.task_id)
```

- [ ] **步骤 3：静态校验 + 回归**

运行：`python -c "import bot.handler"` （确认无语法/导入错误）
运行：`python -m pytest tests/ -v --ignore=tests/test_integration.py`
预期：PASS（collect_deliverables 已在任务 4 覆盖；全量回归不回归）。

- [ ] **步骤 4：Commit**

```bash
git add bot/handler.py
git commit -m "feat: handler 一并推送总结笔记与纠错转写稿"
```

- [ ] **步骤 5：端到端冒烟（可选，需网络/模型）**

用一个真实的 Bilibili/YouTube 链接跑 `python main.py`，确认 Telegram 收到**两份**文档：`{title}_summary.md`（技术知识点结构）与 `{title}.mp3.md`（纠错转写稿）。若所配模型对长纠错块频繁截断（日志可见 `finish_reason=length`），可把 `correction_chunk_char_limit` 调小（如复用 `refine_chunk_char_limit`），无需引入输出上限常量。

---

## 自检

**1. 规格覆盖度：**
- §2 数据流（转录→纠错→总结）→ 任务 5 ✓
- §2.2 字幕路径不纠错 → 任务 5 `test_subtitle_path_skips_correction` ✓
- §2.3 断点续传三级跳 → 任务 1（`can_resume_from_summarization`）+ 任务 5（pipeline `has_corrected` 分支）✓
- §3.1 分块（复用上下文窗口、overlap=0、不传 max_tokens）→ 任务 2 ✓
- §3.2 纠错提示词 → 任务 2 ✓
- §3.3 失败降级 + 截断告警 → 任务 5（降级测试）+ 任务 2（截断 warning）✓
- §4.1 存储与命名（`corrected.json` / `notes/{id}.mp3.md` / `{title}.mp3.md`）→ 任务 1、4、5、6 ✓
- §4.2 极简头 → 任务 4 `build_transcript_markdown` ✓
- §4.3 多文件推送 → 任务 6 ✓
- §4.4 缓存命中读 `.mp3.md` → 任务 4 `collect_deliverables` + 任务 6 ✓
- §5 技术知识点提示词（四字段 + 软兜底）→ 任务 3 ✓
- §6 TaskManager 改动 → 任务 1 ✓
- §7 Pipeline 改动 → 任务 5 ✓
- §8 测试计划 → 各任务内分布 ✓

无遗漏。

**2. 占位符扫描：** 全部步骤含实际代码；无 TODO/待定/"类似任务 N"。

**3. 类型/命名一致性：**
- `Task.corrected_file` / `has_corrected` / `save_corrected` / `load_corrected`：任务 1 定义，任务 5 使用——一致 ✓
- `correction_failed` 字段：任务 1 定义（含 `update_state` kwarg + `_load` setdefault），任务 5 写入，降级测试断言——一致 ✓
- `LLMSummarizer.correct()` / `_correct_single()` / `correction_chunk_char_limit` / `CORRECTION_SYSTEM_PROMPT`：任务 2 定义，任务 5 调用 `self.summarizer.correct(...)`——一致 ✓
- `build_transcript_markdown(title, url, text)`：任务 4 定义，任务 5 导入调用 `build_transcript_markdown(task.title or "video", url, corrected_text)`——签名一致 ✓
- `save_temp_markdown(title, content, suffix=...)`：任务 4 定义（默认 `_summary` 保持向后兼容），任务 6 调用 `suffix=suffix`——一致 ✓
- `collect_deliverables(title, platform, url, summary, model_name, task_id, notes_dir)`：任务 4 定义，任务 6 调用参数顺序一致 ✓

无命名漂移。

---

## 执行交接

计划已完成并保存到 `docs/superpowers/plans/2026-08-08-corrected-transcript-and-tech-notes.md`。两种执行方式：

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代
**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

选哪种方式？
