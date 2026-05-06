import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APITimeoutError, APIConnectionError, RateLimitError

from config import AIConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个专业的笔记助手，擅长将视频转录内容整理成清晰、有条理且信息丰富的笔记。

语言要求：
- 笔记必须使用 **中文** 撰写。
- 专有名词、技术术语、品牌名称和人名应适当保留 **英文**。

输出说明：
- 仅返回最终的 **Markdown 内容**。
- **不要**将输出包裹在代码块中。

请根据提供的转录内容生成结构化笔记，遵循以下原则：

1. **完整信息**：记录尽可能多的相关细节，确保内容全面。
2. **纠错**：语音转文字可能产生错别字或同音错误（如"人工只能"应为"人工智能"），需根据上下文自动修正，输出正确用字。
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
- 笔记必须使用 **中文** 撰写。
- 专有名词、技术术语、品牌名称和人名应适当保留 **英文**。

输出说明：
- 仅返回最终的 **Markdown 内容**。
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

# Token 预算：qwen-plus <120K token 免费额度
# 中文约 1.5-2 char/token，取保守 1.5，留 20K 给 system prompt + 输出
# 100K token ÷ 1.5 char/token ≈ 66K 字符
CHUNK_CHAR_LIMIT = 90_000
CHUNK_OVERLAP = 500


def _split_chunks(text: str, chunk_size: int = CHUNK_CHAR_LIMIT, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks at paragraph boundaries with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to split at paragraph boundary within last 10% of chunk
        search_start = max(start + int(chunk_size * 0.9), start)
        split_pos = text.rfind("\n", search_start, end + 1)
        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunks.append(text[start:split_pos])
        start = split_pos - overlap if split_pos - overlap > start else split_pos

    return chunks


class LLMSummarizer:
    def __init__(self):
        self.client = OpenAI(
            api_key=AIConfig.API_KEY,
            base_url=AIConfig.API_URL,
            timeout=120.0,
        )
        self.model = AIConfig.MODEL_NAME

    def summarize(self, title: str, transcript: str) -> str:
        text_len = len(transcript)

        if text_len <= CHUNK_CHAR_LIMIT:
            return self._summarize_single(title, transcript)

        chunks = _split_chunks(transcript)
        logger.info(
            f"Long transcript ({text_len} chars), using refine strategy: "
            f"{len(chunks)} chunks of ~{CHUNK_CHAR_LIMIT} chars"
        )
        return self._summarize_refine(title, chunks)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=4, max=60),
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
        before_sleep=lambda rs: logger.warning(
            f"LLM call failed ({rs.outcome.exception()}), retrying in {rs.next_action.sleep:.1f}s "
            f"(attempt {rs.attempt_number}/3)"
        ),
    )
    def _summarize_single(self, title: str, transcript: str) -> str:
        user_content = f"视频标题：{title}\n\n转录内容：\n{transcript}"

        logger.info(f"Calling LLM (model={self.model}, text_len={len(transcript)})")

        response = self.client.chat.completions.create(
            model=self.model,
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=4, max=60),
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
        before_sleep=lambda rs: logger.warning(
            f"LLM call failed ({rs.outcome.exception()}), retrying in {rs.next_action.sleep:.1f}s "
            f"(attempt {rs.attempt_number}/3)"
        ),
    )
    def _call_llm(self, system_prompt: str, user_content: str):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=4000,
            temperature=0.7,
        )

    def _summarize_refine(self, title: str, chunks: list[str]) -> str:
        """Iteratively refine summary by processing chunks sequentially."""
        # First chunk: generate initial summary
        running_summary = self._summarize_single(title, chunks[0])

        # Subsequent chunks: refine existing summary with new content
        for i in range(1, len(chunks)):
            logger.info(f"Refining with chunk {i + 1}/{len(chunks)}")

            user_content = (
                f"视频标题：{title}\n\n"
                f"以下是已有的笔记摘要：\n\n{running_summary}\n\n"
                f"---\n\n"
                f"以下是新的转录片段：\n\n{chunks[i]}"
            )

            response = self._call_llm(REFINE_SYSTEM_PROMPT, user_content)
            running_summary = response.choices[0].message.content
            logger.info(f"Refined with chunk {i + 1}, summary now {len(running_summary)} chars")

        logger.info(f"Final summary generated ({len(running_summary)} chars)")
        return running_summary
