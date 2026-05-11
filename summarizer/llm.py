import logging
from dataclasses import dataclass

import httpx
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError

from config import AIConfig, ModelConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个专业的笔记助手，擅长将视频转录内容整理成清晰、有条理且信息丰富的笔记。

语言要求：
- 笔记必须使用 **简体中文** 撰写，即使转录原文为繁体中文或其他语言，输出一律转换为简体中文。
- 专有名词、技术术语、品牌名称和人名应适当保留 **英文**。

输出说明：
- 仅返回最终的 **Markdown 内容**。
- **不要**将输出包裹在代码块中。

请根据提供的转录内容生成结构化笔记，遵循以下原则：

1. **完整信息**：记录尽可能多的相关细节，确保内容全面。
2. **纠错**：语音转文字可能产生错别字或同音错误（如"人工只能"应为"人工智能"、"濃縮"应为"浓缩"），需根据上下文自动修正，输出正确用字。
3. **去除无关内容**：省略广告、填充词、问候语和不相关的言论。
4. **保留关键细节**：保留重要事实、示例、结论和建议。
5. **可读布局**：必要时使用项目符号，保持段落简短，增强可读性。
6. 视频中提及的数学公式必须保留，并以 LaTeX 语法形式呈现。
7. **Markdown 表格**：对核心观点、关键要点、对比分析、分类信息等结构化内容，应使用 Markdown 表格进行梳理输出，提升信息密度和可读性。
8. 在笔记末尾添加一段专业的 **AI 总结**，简要概括整个视频的核心内容。
"""

REFINE_SYSTEM_PROMPT = """你是一个专业的笔记助手，正在分阶段处理一段长视频转录内容。

你已有前一部分内容的笔记摘要，现在需要结合新提供的转录片段，更新和完善笔记。

语言要求：
- 笔记必须使用 **简体中文** 撰写，即使转录原文为繁体中文或其他语言，输出一律转换为简体中文。
- 专有名词、技术术语、品牌名称和人名应适当保留 **英文**。
- **不要**将输出包裹在代码块中。

规则：
1. 保留已有笔记中的所有重要信息，不要丢弃。最好原始保留已有内容，除非新片段提供了更准确或更详细的信息。
2. 将新片段中的重要信息整合进笔记的对应位置。
3. 如果新片段与已有内容重复，合并而非简单堆叠。
4. **纠错**：语音转文字可能产生错别字或同音错误，需根据上下文自动修正。
5. 去除广告、填充词、问候语等无关内容。
6. 视频中提及的数学公式必须保留，并以 LaTeX 语法形式呈现。
7. **Markdown 表格**：对核心观点、关键要点、对比分析、分类信息等结构化内容，应使用 Markdown 表格进行梳理输出，提升信息密度和可读性。
8. 在笔记末尾保留/更新一段专业的 **AI 总结**。
"""

# 中文约 1.5-2 char/token，取保守 1.5
CHARS_PER_TOKEN = 1.5
# 为 system prompt + 输出预留的 token 数
RESERVED_TOKENS = 8000
# refine 阶段额外需要携带已有摘要，多预留一些
REFINE_EXTRA_RESERVED = 4000


def _calc_chunk_char_limit(max_context_tokens: int) -> int:
    available = max_context_tokens - RESERVED_TOKENS
    return int(available / CHARS_PER_TOKEN)


CHUNK_OVERLAP = 500


def _split_chunks(text: str, chunk_size: int = 0, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        chunk_size = _calc_chunk_char_limit(AIConfig.MAX_CONTEXT_TOKENS)
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break

        search_start = max(start + int(chunk_size * 0.9), start)
        split_pos = text.rfind("\n", search_start, end + 1)
        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunks.append(text[start:split_pos])
        start = split_pos - overlap if split_pos - overlap > start else split_pos

    return chunks


@dataclass
class _ModelClient:
    config: ModelConfig
    client: OpenAI


class LLMSummarizer:
    CONNECT_TIMEOUT = 10.0
    READ_TIMEOUT = 180.0

    def __init__(self):
        self._last_model_name: str | None = None
        self._models: list[_ModelClient] = []
        for mc in AIConfig.load_models():
            client = OpenAI(
                api_key=mc.key,
                base_url=mc.url,
                timeout=httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT),
                max_retries=0,
            )
            self._models.append(_ModelClient(config=mc, client=client))

        # Chunk limits based on first model's context (all models should handle similar sizes)
        first_ctx = self._models[0].config.max_context_tokens
        self.chunk_char_limit = _calc_chunk_char_limit(first_ctx)
        self.refine_chunk_char_limit = self.chunk_char_limit - int(REFINE_EXTRA_RESERVED / CHARS_PER_TOKEN)
        logger.info(f"Chunk size: {self.chunk_char_limit} chars, refine chunk: {self.refine_chunk_char_limit} chars")

        self._check_connectivity()

    def _check_connectivity(self):
        for mc in self._models:
            try:
                mc.client.models.list()
                logger.info(f"API OK: {mc.config.name} @ {mc.config.url}")
            except Exception as e:
                logger.warning(f"API unreachable: {mc.config.name} @ {mc.config.url} — {e}")

    def _call_with_fallback(self, fn, label: str = "LLM call"):
        """Try fn(client, model_name) on each model in order, return first success."""
        last_err = None
        for mc in self._models:
            try:
                logger.info(f"Calling {label} (model={mc.config.name})")
                result = fn(mc.client, mc.config.name)
                self._last_model_name = mc.config.name
                return result
            except (APITimeoutError, APIConnectionError, RateLimitError, httpx.TimeoutException) as e:
                last_err = e
                logger.warning(f"{label} failed on {mc.config.name}: {type(e).__name__}: {e}")
            except APIStatusError as e:
                last_err = e
                logger.warning(f"{label} failed on {mc.config.name}: HTTP {e.status_code} — {e.message}")

        raise RuntimeError(f"All {len(self._models)} model(s) failed for {label}") from last_err

    def summarize(self, title: str, transcript: str) -> str:
        text_len = len(transcript)

        if text_len <= self.chunk_char_limit:
            return self._summarize_single(title, transcript)

        chunks = _split_chunks(transcript, chunk_size=self.chunk_char_limit)
        logger.info(
            f"Long transcript ({text_len} chars), using refine strategy: "
            f"{len(chunks)} chunks of ~{self.chunk_char_limit} chars"
        )
        return self._summarize_refine(title, chunks)

    def _summarize_single(self, title: str, transcript: str) -> str:
        def _fn(client, model):
            user_content = f"视频标题：{title}\n\n转录内容：\n{transcript}"
            logger.info(f"Calling LLM (model={model}, text_len={len(transcript)})")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=4000,
                temperature=0.7,
            )
            result = response.choices[0].message.content
            logger.info(f"LLM summary generated ({len(result)} chars)")
            return result

        return self._call_with_fallback(_fn, "summarize")

    def _call_llm(self, system_prompt: str, user_content: str):
        def _fn(client, model):
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=4000,
                temperature=0.7,
            )

        return self._call_with_fallback(_fn, "refine")

    def _summarize_refine(self, title: str, chunks: list[str]) -> str:
        running_summary = self._summarize_single(title, chunks[0])

        remaining_text = "\n".join(chunks[1:])
        refine_chunks = _split_chunks(remaining_text, chunk_size=self.refine_chunk_char_limit)

        for i, chunk in enumerate(refine_chunks, start=1):
            logger.info(f"Refining with chunk {i}/{len(refine_chunks)}")

            user_content = (
                f"视频标题：{title}\n\n"
                f"以下是已有的笔记摘要：\n\n{running_summary}\n\n"
                f"---\n\n"
                f"以下是新的转录片段：\n\n{chunk}"
            )

            response = self._call_llm(REFINE_SYSTEM_PROMPT, user_content)
            running_summary = response.choices[0].message.content
            logger.info(f"Refined with chunk {i}/{len(refine_chunks)}, summary now {len(running_summary)} chars")

        logger.info(f"Final summary generated ({len(running_summary)} chars)")
        return running_summary
