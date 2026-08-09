# 纠错部分失败容忍 + LLM 流式调用 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 让纠错步骤在单块失败时保留已成功块并用原文兜底（A1）；让所有 LLM 调用走流式，消除长输出的总时间 read timeout（B1）。

**架构：** 仅改 `summarizer/llm.py`。新增实例方法 `_complete_stream` 统一处理流式拉取 + 内容拼接 + 空内容/length 错误判定，三处 `_fn`（correct / summarize-single / refine）改为调用它；删除模块级非流式函数 `_extract_content`。`correct()` 循环对 `_correct_single` 加 try/except，失败块用原文兜底。`_call_with_fallback` 编排逻辑不变。

**技术栈：** Python 3.11、openai SDK（`chat.completions.create(stream=True, stream_options={"include_usage": True})`）、pytest、unittest.mock。

**规格：** `docs/superpowers/specs/2026-08-09-correct-failure-tolerance-and-streaming-design.md`

---

## 文件结构

- **修改** `summarizer/llm.py`：
  - 删除模块级函数 `_extract_content`（第 27 行起，B1 后无调用点）。
  - 新增实例方法 `_complete_stream`（统一流式调用，承担原 `_extract_content` 的错误判定职责）。
  - 改造三处 `_fn`：`_correct_single`（第 289 行）、`_summarize_single`（第 320 行）、`_call_llm`（第 340 行）。
  - `correct()`（第 271 行）循环加 try/except。
- **测试** `tests/test_corrector.py`：新增 `_stream` helper；新增 `_complete_stream` 与 A1 单测；现有 `_resp` 调用迁移为 `_stream`。
- **测试** `tests/test_summarizer.py`：新增 `_stream` helper；现有内联非流式 mock 迁移为 `_stream`。

---

## 任务 1：LLM 调用流式化（B1）

**文件：**
- 修改：`summarizer/llm.py`
- 测试：`tests/test_corrector.py`、`tests/test_summarizer.py`

- [ ] **步骤 1：在 `tests/test_corrector.py` 顶部新增流式 mock helper `_stream`（暂不删 `_resp`）**

在 `tests/test_corrector.py` 现有 `_resp` 函数之后追加：

```python
def _stream(content, finish_reason="stop", comp_tokens=None):
    """构造流式响应 chunk 列表：一个携带 content 的 delta chunk + 一个携带 finish_reason/usage 的尾部 chunk。
    content="" 时头部 delta.content 为空字符串（falsy），_complete_stream 不会 append，用于测试空内容。"""
    chunks = []
    head = MagicMock()
    head_choice = MagicMock()
    head_choice.delta.content = content
    head_choice.finish_reason = None
    head.choices = [head_choice]
    head.usage = None
    chunks.append(head)
    tail = MagicMock()
    tail_choice = MagicMock()
    tail_choice.delta.content = None
    tail_choice.finish_reason = finish_reason
    tail.choices = [tail_choice]
    tail.usage = MagicMock(completion_tokens=comp_tokens) if comp_tokens is not None else None
    chunks.append(tail)
    return chunks
```

同时在文件顶部 import 区追加（若尚无）：

```python
import pytest
```

- [ ] **步骤 2：编写 `_complete_stream` 的失败测试**

在 `tests/test_corrector.py` 追加三个测试（直接调用实例方法 `_complete_stream`）：

```python
def test_complete_stream_concatenates_content():
    mc = MagicMock()
    mc.chat.completions.create.return_value = _stream("hello world", comp_tokens=10)
    s = _make_summarizer(mc)
    result = s._complete_stream(
        mc, "test-model", [{"role": "user", "content": "hi"}],
        temperature=0.3, reasoning=False, label="t",
    )
    assert result == "hello world"
    call = mc.chat.completions.create.call_args
    assert call.kwargs["stream"] is True
    assert call.kwargs["stream_options"] == {"include_usage": True}


def test_complete_stream_empty_content_raises():
    from summarizer.llm import EmptyLLMResponseError
    mc = MagicMock()
    mc.chat.completions.create.return_value = _stream("", finish_reason="stop")
    s = _make_summarizer(mc)
    with pytest.raises(EmptyLLMResponseError):
        s._complete_stream(
            mc, "test-model", [{"role": "user", "content": "hi"}],
            temperature=0.3, reasoning=False, label="t",
        )


def test_complete_stream_finish_reason_length_raises():
    from summarizer.llm import EmptyLLMResponseError
    mc = MagicMock()
    mc.chat.completions.create.return_value = _stream("部分内容", finish_reason="length", comp_tokens=100)
    s = _make_summarizer(mc)
    with pytest.raises(EmptyLLMResponseError):
        s._complete_stream(
            mc, "test-model", [{"role": "user", "content": "hi"}],
            temperature=0.3, reasoning=False, label="t",
        )
```

- [ ] **步骤 3：运行测试验证失败**

运行：`python -m pytest tests/test_corrector.py::test_complete_stream_concatenates_content tests/test_corrector.py::test_complete_stream_empty_content_raises tests/test_corrector.py::test_complete_stream_finish_reason_length_raises -v`

预期：FAIL，报错 `AttributeError: 'LLMSummarizer' object has no attribute '_complete_stream'`（方法尚未实现）。

- [ ] **步骤 4：实现 `_complete_stream` 实例方法**

在 `summarizer/llm.py` 的 `LLMSummarizer` 类中、`_call_with_fallback` 方法之后新增（先不要删 `_extract_content`）：

```python
def _complete_stream(self, client, model, messages, *, temperature, reasoning, label) -> str:
    """流式调用 LLM，拼接 content；空内容 / finish_reason=length 视为失败抛 EmptyLLMResponseError，
    交由 _call_with_fallback 切换下一个模型。流式下 read timeout 为空闲超时，避免长输出的总时间超时。"""
    kwargs = dict(temperature=temperature, stream=True,
                  stream_options={"include_usage": True})
    if reasoning:
        kwargs["reasoning_effort"] = "low"
    stream = client.chat.completions.create(model=model, messages=messages, **kwargs)

    parts: list[str] = []
    finish_reason = None
    usage = None
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                parts.append(delta.content)
            fr = chunk.choices[0].finish_reason
            if fr:
                finish_reason = fr
    content = "".join(parts).strip()
    comp_tok = getattr(usage, "completion_tokens", "?") if usage else "?"

    if not content:
        raise EmptyLLMResponseError(
            f"{label} returned empty content (finish_reason={finish_reason}, "
            f"completion_tokens={comp_tok}) — likely reasoning exhausted the output budget"
        )
    if finish_reason == "length":
        raise EmptyLLMResponseError(
            f"{label} hit output limit (finish_reason=length, completion_tokens={comp_tok})"
        )
    logger.info(
        f"{label} ok: content={len(content)} chars, finish_reason={finish_reason}, "
        f"completion_tokens={comp_tok}"
    )
    return content
```

- [ ] **步骤 5：运行新测试验证通过**

运行：`python -m pytest tests/test_corrector.py::test_complete_stream_concatenates_content tests/test_corrector.py::test_complete_stream_empty_content_raises tests/test_corrector.py::test_complete_stream_finish_reason_length_raises -v`

预期：3 个 PASS。

- [ ] **步骤 6：改造三处 `_fn` 调用 `_complete_stream`，并删除模块级 `_extract_content`**

**6a.** 替换 `_correct_single`（`summarizer/llm.py` 第 289 行起）整个方法为：

```python
def _correct_single(self, chunk: str) -> str:
    def _fn(client, model):
        return self._complete_stream(
            client, model,
            messages=[
                {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"原始转写文本：\n{chunk}"},
            ],
            temperature=0.1, reasoning=_is_reasoning_model(model), label="correct",
        )
    return self._call_with_fallback(_fn, "correct")
```

**6b.** 替换 `_summarize_single`（第 320 行起）的 `_fn` 内部为（保留 `logger.info` 行与 `return self._call_with_fallback(...)`）：

```python
def _summarize_single(self, title: str, transcript: str) -> str:
    def _fn(client, model):
        inner = f"视频标题：{title}\n\n转录内容：\n{transcript}"
        user_content = self._wrap_user_content(inner)
        logger.info(f"Calling LLM (model={model}, text_len={len(transcript)})")
        return self._complete_stream(
            client, model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3, reasoning=_is_reasoning_model(model), label="summarize",
        )
    return self._call_with_fallback(_fn, "summarize")
```

**6c.** 替换 `_call_llm`（第 340 行起）的 `_fn` 内部为：

```python
def _call_llm(self, system_prompt: str, user_content: str) -> str:
    def _fn(client, model):
        return self._complete_stream(
            client, model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3, reasoning=_is_reasoning_model(model), label="refine",
        )
    return self._call_with_fallback(_fn, "refine")
```

**6d.** 删除模块级函数 `_extract_content`（第 27 行起至其函数体结束，含其上方一行注释 `# ...` 若有）。删除后全文件搜索确认无 `_extract_content` 残留：

```bash
grep -n "_extract_content" summarizer/llm.py
```

预期：无输出（或仅剩 docstring 提及——见步骤 6e 清理）。

**6e.** 更新 `correct()` 的 docstring（第 274 行）：将其中的 `由 _extract_content 抛 EmptyLLMResponseError` 改为 `由 _complete_stream 抛 EmptyLLMResponseError`。

- [ ] **步骤 7：迁移 `tests/test_corrector.py` 现有测试的 mock 为流式**

此步与步骤 6 是一次原子迁移（代码已切流式，测试 mock 必须同步切流式）。把 `tests/test_corrector.py` 中所有 `_resp(...)` 调用替换为 `_stream(...)`，具体：

- `test_correct_single_chunk_returns_content`：`_resp("纠错后的文本")` → `_stream("纠错后的文本")`
- `test_correct_uses_correction_system_prompt`：`_resp("x")` → `_stream("x")`
- `test_correct_does_not_set_max_tokens`：`_resp("x")` → `_stream("x")`
- `test_correct_multi_chunk_concatenates_in_order`：`side_effect = [_resp("A"), _resp("B")]` → `[_stream("A"), _stream("B")]`
- `test_correct_no_overlap_reconstructs_original`：echo 内 `return _resp(body)` → `return _stream(body)`

（`test_correct_empty_returns_empty_without_call` 不涉及 `_resp`，不改。）

- [ ] **步骤 8：迁移 `tests/test_summarizer.py` 现有测试的 mock 为流式**

**8a.** 在 `tests/test_summarizer.py` 顶部 import 区之后新增与 `test_corrector.py` 相同的 `_stream` helper（复制步骤 1 的 `_stream` 函数定义，保持两个测试文件各自自包含，符合现有模式）：

```python
def _stream(content, finish_reason="stop", comp_tokens=None):
    chunks = []
    head = MagicMock()
    head_choice = MagicMock()
    head_choice.delta.content = content
    head_choice.finish_reason = None
    head.choices = [head_choice]
    head.usage = None
    chunks.append(head)
    tail = MagicMock()
    tail_choice = MagicMock()
    tail_choice.delta.content = None
    tail_choice.finish_reason = finish_reason
    tail.choices = [tail_choice]
    tail.usage = MagicMock(completion_tokens=comp_tokens) if comp_tokens is not None else None
    chunks.append(tail)
    return chunks
```

**8b.** `test_summarize` 中替换内联 mock（`mock_choice` / `mock_response` 那几行）为：

```python
mock_client.chat.completions.create.return_value = _stream("# Test Summary\n\nThis is a summary.")
```

删除原来的 `mock_choice = MagicMock()` / `mock_choice.message.content = ...` / `mock_response = MagicMock()` / `mock_response.choices = [mock_choice]` 四行。

**8c.** `test_summarize_uses_config` 中同样把内联 mock 替换为：

```python
mock_client.chat.completions.create.return_value = _stream("Summary")
```

删除原 `mock_choice` / `mock_response` 构造行。

（`test_check_connectivity_*` 三个测试 mock 的是 `models.list()`，不涉及 chat completions，不改。）

- [ ] **步骤 9：运行两个测试文件全部测试验证通过**

运行：`python -m pytest tests/test_corrector.py tests/test_summarizer.py -v`

预期：全部 PASS（含 3 个新 `_complete_stream` 测试 + 所有迁移后的现有测试）。

- [ ] **步骤 10：删除 `tests/test_corrector.py` 中废弃的 `_resp` helper**

`_resp` 已无任何引用，删除其整个函数定义。确认无残留：

```bash
grep -n "_resp" tests/test_corrector.py
```

预期：无输出。

- [ ] **步骤 11：运行全部相关测试确认通过**

运行：`python -m pytest tests/test_corrector.py tests/test_summarizer.py -v`

预期：全部 PASS。

- [ ] **步骤 12：Commit**

```bash
git add summarizer/llm.py tests/test_corrector.py tests/test_summarizer.py
git commit -m "refactor: LLM 调用改为流式，避免长输出 read timeout"
```

---

## 任务 2：纠错部分失败容忍（A1）

**文件：**
- 修改：`summarizer/llm.py` 的 `correct()`
- 测试：`tests/test_corrector.py`

- [ ] **步骤 1：编写失败测试 `test_correct_chunk_failure_falls_back_to_raw`**

在 `tests/test_corrector.py` 追加：

```python
def test_correct_chunk_failure_falls_back_to_raw():
    """多块纠错时，中间块失败应使用该块原文，其余块用纠正结果，整体不抛错。"""
    mc = MagicMock()
    # correction_chunk_char_limit=5 => "0123456789" 切成 ["01234", "56789"]
    # 第1块纠正成功返回 "AAAAA"，第2块抛 RuntimeError（模拟所有模型失败）
    mc.chat.completions.create.side_effect = [_stream("AAAAA"), RuntimeError("All 2 model(s) failed for correct")]
    s = _make_summarizer(mc)
    s.correction_chunk_char_limit = 5
    result = s.correct("0123456789")
    assert result == "AAAAA" + "56789"   # 第1块纠正 + 第2块原文兜底
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_corrector.py::test_correct_chunk_failure_falls_back_to_raw -v`

预期：FAIL——当前 `correct()` 不 catch 异常，`RuntimeError` 向上冒泡（报错 `RuntimeError: All 2 model(s) failed for correct`）。

- [ ] **步骤 3：实现 `correct()` 部分失败容忍**

替换 `summarizer/llm.py` 中 `correct()`（第 271 行起）整个方法为：

```python
def correct(self, text: str) -> str:
    """对 ASR 原始转写做纠错。无损：保留全部内容，仅修正错误。
    复用上下文窗口分块、overlap=0，逐块纠错后按顺序拼接。
    单块失败时用该块原文兜底，不中断后续块（A1）；全块失败则返回原文。
    截断/空响应由 _complete_stream 抛 EmptyLLMResponseError，经 _call_with_fallback 自动切模型，
    切尽仍失败则在本方法内被 catch、用原文兜底。"""
    if not text:
        return text

    chunks = _split_chunks(text, chunk_size=self.correction_chunk_char_limit, overlap=0)
    logger.info(f"Correcting transcript ({len(text)} chars, {len(chunks)} chunk(s))")

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            corrected = self._correct_single(chunk)
            parts.append(corrected)
            logger.info(f"Corrected chunk {i}/{len(chunks)} ({len(corrected)} chars)")
        except Exception as e:
            logger.warning(
                f"Correct chunk {i}/{len(chunks)} failed, using raw text "
                f"({len(chunk)} chars): {e}"
            )
            parts.append(chunk)
    return "".join(parts)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_corrector.py::test_correct_chunk_failure_falls_back_to_raw -v`

预期：PASS。

- [ ] **步骤 5：编写测试 `test_correct_all_chunks_fail_returns_raw`**

在 `tests/test_corrector.py` 追加：

```python
def test_correct_all_chunks_fail_returns_raw():
    """所有块都失败时，返回值等于原文，且不抛错。"""
    mc = MagicMock()
    mc.chat.completions.create.side_effect = [
        RuntimeError("All 2 model(s) failed for correct"),
        RuntimeError("All 2 model(s) failed for correct"),
    ]
    s = _make_summarizer(mc)
    s.correction_chunk_char_limit = 5
    result = s.correct("0123456789")
    assert result == "0123456789"
```

- [ ] **步骤 6：运行测试验证通过**

运行：`python -m pytest tests/test_corrector.py::test_correct_all_chunks_fail_returns_raw -v`

预期：PASS。

- [ ] **步骤 7：运行两个测试文件全部测试确认无回归**

运行：`python -m pytest tests/test_corrector.py tests/test_summarizer.py -v`

预期：全部 PASS。

- [ ] **步骤 8：Commit**

```bash
git add summarizer/llm.py tests/test_corrector.py
git commit -m "fix: 纠错单块失败时用原文兜底，不再整体回退"
```

---

## 自检

**1. 规格覆盖度：**
- 规格 §3（A1 部分失败容忍）→ 任务 2 全部步骤覆盖（try/except、单块兜底、全块失败返回原文、pipeline 兜底保留不改）。✓
- 规格 §4（B1 流式）→ 任务 1 覆盖（`_complete_stream`、三处 `_fn` 改造、删 `_extract_content`、`include_usage`、`READ_TIMEOUT` 不变）。✓
- 规格 §5.2（测试 mock 形状迁移）→ 任务 1 步骤 1/7/8 覆盖两个测试文件的 helper 迁移。✓
- 规格 §6（测试计划 8 项）→ `_complete_stream` concat/empty/length（任务1步骤2）、A1 单块失败/全块失败（任务2步骤1/5）、现有测试迁移回归（任务1步骤9、任务2步骤7）。✓
- 规格 §8（验收标准）→ 任务 1（流式 + token 日志）、任务 2（部分失败）对应。✓
- 遗漏：无。

**2. 占位符扫描：** 无 TODO/待定/“类似任务 N”/“添加错误处理”等。所有代码步骤含完整代码块。✓

**3. 类型一致性：**
- `_complete_stream` 签名 `(self, client, model, messages, *, temperature, reasoning, label)` 在任务 1 步骤 4 定义，步骤 6a/6b/6c 三处调用参数（`client, model, messages=[...], temperature=..., reasoning=_is_reasoning_model(model), label=...`）一致。✓
- `_stream(content, finish_reason="stop", comp_tokens=None)` 在任务 1 步骤 1 与步骤 8a 两处定义完全一致。✓
- A1 测试用 `_stream`（任务 1 已引入），无需额外 helper。✓
- `_make_summarizer`（test_corrector.py 既有）在任务 1/2 测试中复用，未改名。✓
- `EmptyLLMResponseError` 为 `summarizer/llm.py` 既有异常类，任务 1 步骤 4 引用一致。✓

无需修复。
