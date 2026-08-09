# 设计规格：纠错步骤部分失败容忍 + LLM 流式调用

- **日期**：2026-08-09
- **状态**：已确认，待编写实现计划
- **范围**：单次实现可覆盖（仅改 `summarizer/llm.py` + 对应单测适配）

---

## 1. 目标与背景

处理一条 68069 字符的长转录文本时，日志暴露两个失败：

```
07:21:56  correct ok: chunk 1/2 (37095 chars)         ← 第1块成功
07:21:56  Calling correct chunk 2 (step-3.7-flash)
07:24:57  超时 (3分钟)
07:24:57  Calling correct chunk 2 (deepseek)
07:27:57  超时 (3分钟)
07:27:57  Transcript correction failed → 回退原始转录   ← 第1块成果被丢弃
07:27:57  Calling summarize chunk 1 (step-3.7-flash)
07:30:58  超时 (3分钟)
07:30:58  Calling summarize chunk 1 (deepseek)
07:32:18  成功 (~80s)
```

本次改动解决两个根因：

1. **A1 — 纠错"全有或全无"**：任一分块失败，整个 `correct()` 抛异常，已成功块的成果与耗时全部丢弃，回退原始转录。
2. **B1 — 长输出触发 read timeout**：非流式调用下 `READ_TIMEOUT=180s` 是"总时间"超时；推理模型（`step-3.7-flash`，`_is_reasoning_model` 命中 `step-3` 前缀）处理 37000 字符的大输入时，reasoning + 长输出很容易超过 180s。

最终目标：纠错步骤不再因单块失败而全盘白费；LLM 调用改为流式，消除"输出阶段"超时。

---

## 2. 问题分析

### 2.1 纠错全有或全无（A1 对应）

`summarizer/llm.py` 的 `correct()`：

```python
for i, chunk in enumerate(chunks, start=1):
    corrected = self._correct_single(chunk)   # 任一块失败 → 整个方法抛 RuntimeError
    parts.append(corrected)
```

`corrected.json` 由 `pipeline.py` 在 `correct()` **整体返回后**才一次性落盘。因此只要第 N 块失败，前 N−1 块已成功的纠正结果（耗时 + token）全部丢失。日志中第 1 块成功、第 2 块失败，导致整步回退原始转录。

### 2.2 长输出 read timeout（B1 对应）

- `LLMSummarizer.READ_TIMEOUT = 180.0`，OpenAI 客户端 `max_retries=0`。
- 非流式调用下，read timeout 是"整个响应的总时长"。纠错/总结单块输入约 37000 字符、输出与之相当，推理模型先做 reasoning 再输出，180s 频繁不够。
- `deepseek` 做 summarize 仅用 ~80s，证明任务本身可行，瓶颈在"推理模型 + 非流式总时间超时"的组合。

---

## 3. 设计：A1 — 纠错部分失败容忍

### 3.1 改动

`summarizer/llm.py` 的 `correct()` 循环，对 `_correct_single` 加 try/except：

```python
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
        parts.append(chunk)   # 该块用原文兜底
return "".join(parts)
```

### 3.2 行为约定

- 任一块成功 → 纠正后的内容保留，不白费。
- 任一块失败 → 该块用原文，其余块正常处理，最终按序拼接。
- 极端情况（所有块都失败）→ 返回值等于原文，语义上等同"未纠正"，`pipeline.py` 照常拿它做 summarize。
- `pipeline.py` 现有 try/except（`correction_failed=True` 回退原文）**保留作为最后兜底**（如 `_split_chunks` 异常、`correct()` 入口校验异常），但 A1 后正常路径不再触发。
- `_correct_single` → `_call_with_fallback` 的"遍历所有模型、全失败抛 RuntimeError"机制**不变**；A1 只是把它从"致命"降级为"可兜底"。

### 3.3 不做的事（YAGNI）

- 不为单块做断点续传落盘（不引入 `corrected.partN.json`）。A1 的目标只是"不丢弃已成功块"，逐块持久化属于过度设计。
- 不改 `correction_chunk_char_limit`（仍复用总结的上下文窗口切分，overlap=0）。
- 不改 `pipeline.py` 逻辑。

---

## 4. 设计：B1 — LLM 流式调用

### 4.1 思路

所有 `chat.completions.create` 调用改为 `stream=True`。流式下 httpx 的 read timeout 语义变为"**两次数据之间的空闲超时**"：只要模型持续吐 token，就不会超时。这正好解决日志里推理模型在长输出阶段的总时间超时。

### 4.2 新增统一方法 `_complete_stream`

替换现有 `_extract_content`（非流式），新增一个统一处理"流式拉取 + 内容拼接 + 错误判定"的方法：

```python
def _complete_stream(self, client, model, messages, *, temperature, reasoning, label) -> str:
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

### 4.3 调用点改造

correct / summarize-single / refine 三处 `_fn` 统一改调 `_complete_stream`，删除 `_extract_content`：

```python
# correct 的 _fn
def _fn(client, model):
    user_content = f"原始转写文本：\n{chunk}"
    return self._complete_stream(
        client, model,
        messages=[
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1, reasoning=_is_reasoning_model(model), label="correct",
    )
```

`summarize-single` 与 refine 的 `_fn` 同理，`temperature=0.3`，各自传 system prompt 与 user_content。

### 4.4 关键约定

- **usage 统计**：必须传 `stream_options={"include_usage": True}`。否则流式拿不到 `completion_tokens`，日志的 token 用量会丢失（日志里 `completion_tokens` 是排查"推理耗尽输出预算"的关键信号，不能丢）。
- **错误判定不变**：空 content / `finish_reason=length` 仍抛 `EmptyLLMResponseError` → `_call_with_fallback` 自动切下一个模型。
- **fallback 编排不变**：`_call_with_fallback`（遍历模型、catch `APITimeoutError`/`EmptyLLMResponseError` 等、全失败抛 `RuntimeError`）**完全不改**。B1 的改动局限在 `_fn` 与新增方法层面。
- **推理模型**：`step` 系列流式通常把 reasoning 放在 `delta.reasoning_content`，不计入 `content`；本方法只拼接 `content`，符合预期。`reasoning_effort="low"` 仍按 `_is_reasoning_model(model)` 条件传入。
- **`READ_TIMEOUT` 保持 180s 不变**：流式下它变成"空闲超时"。若某 provider 在 reasoning 阶段长时间完全不发任何字节，仍可能触发，但已远比非流式"总时间超时"宽松。先观察实际效果，不提前调参。
- **移除 `_extract_content`**：B1 后无任何非流式调用点，按 YAGNI 删除该函数，避免遗留死代码。

---

## 5. 影响面

### 5.1 代码

- 仅改 `summarizer/llm.py`：
  - `correct()` 循环加 try/except（A1）。
  - 新增 `_complete_stream`，三个 `_fn` 改调用它，删除 `_extract_content`（B1）。
- `core/pipeline.py` **不改**（现有 correct 失败兜底保留，正常路径不再触发）。

### 5.2 测试（重点：mock 形状破坏）

现有 `tests/test_corrector.py` 与 `tests/test_summarizer.py` 的 mock helper 构造的是**非流式响应**结构：

```python
def _resp(content):
    m = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    m.choices = [choice]
    m.usage = None
    return m
```

B1 后 `create(..., stream=True)` 返回的是**可迭代的流**，逐 chunk 带 `choices[0].delta.content`、末块 `choices[0].finish_reason`、末块 `usage`。因此：

- `_resp` 必须改为生成"流式 chunk 列表"的 helper，例如：
  ```python
  def _stream(content, finish_reason="stop", comp_tokens=None):
      chunks = []
      for piece in [content]:          # 简化：单 chunk 携带全部 content
          c = MagicMock()
          ch = MagicMock(); ch.delta.content = piece; ch.finish_reason = None
          c.choices = [ch]; c.usage = None
          chunks.append(c)
      tail = MagicMock()
      ch = MagicMock(); ch.delta.content = None; ch.finish_reason = finish_reason
      tail.choices = [ch]
      tail.usage = MagicMock(completion_tokens=comp_tokens) if comp_tokens else None
      chunks.append(tail)
      return chunks
  ```
- `test_corrector.py` / `test_summarizer.py` 中所有 `mc.chat.completions.create.return_value = _resp(...)` 与 `side_effect = [_resp(...), ...]` 都要迁移到 `_stream(...)`。
- 这是 B1 必然的、范围可控的连带改动，不视为额外功能。

---

## 6. 测试计划

新增/更新单测（均在 `summarizer/llm.py` 对应测试文件，mock 流式响应）：

**A1 — 部分失败容忍**
1. `test_correct_chunk_failure_falls_back_to_raw`：多块中某块抛异常，断言该块用原文、其余块用纠正结果、整体不抛错。
2. `test_correct_all_chunks_fail_returns_raw`：所有块都失败，断言返回值等于原文、不抛错。

**B1 — 流式调用**
3. `test_stream_concatenates_content`：mock 流式多个 delta chunk，断言拼接出完整 content。
4. `test_stream_empty_content_raises`：流式无 content delta（finish_reason=stop）→ 抛 `EmptyLLMResponseError`，可被 `_call_with_fallback` 捕获切模型。
5. `test_stream_finish_reason_length_raises`：末块 `finish_reason=length` → 抛 `EmptyLLMResponseError`。
6. `test_stream_carries_usage`：传 `include_usage`，末块 `usage.completion_tokens` 被记录（日志/不抛错即可）。

**回归**
7. 现有 `test_corrector.py`（单块返回、系统提示词、不传 max_tokens、空输入、多块顺序拼接、overlap=0 还原原文）全部迁移到流式 mock 后保持通过。
8. 现有 `test_summarizer.py`（summarize 基本流程、配置传入、提示词字段、连通性探测）迁移后保持通过。

**验证命令**：
```bash
python -m pytest tests/test_corrector.py tests/test_summarizer.py -v
```

---

## 7. 非目标

- 不改 `READ_TIMEOUT`（180s）、不改 `max_retries=0`、不改 chunk 切分大小。
- 不改 `pipeline.py`、`task_manager.py`、断点续传逻辑。
- 不引入 correct 单块断点续传落盘。
- 不把 correct 设为可选开关（本次保留为默认开启；是否降级为可选留待后续）。
- 不改总结/纠错/改写的提示词内容。

---

## 8. 验收标准

- `correct()` 任一分块失败时不再整体抛异常，已成功块保留、失败块用原文。
- 所有 LLM 调用走流式，日志仍能打印 `completion_tokens`。
- 推理模型（step 系列）在大输入下不再因"总时间 read timeout"失败（流式下只要持续有 token 即不超时）。
- `tests/test_corrector.py` 与 `tests/test_summarizer.py` 全部通过。
