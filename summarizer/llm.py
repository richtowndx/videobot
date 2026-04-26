import logging
from typing import Optional

from openai import OpenAI

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
2. **去除无关内容**：省略广告、填充词、问候语和不相关的言论。
3. **保留关键细节**：保留重要事实、示例、结论和建议。
4. **可读布局**：必要时使用项目符号，保持段落简短，增强可读性。
5. 视频中提及的数学公式必须保留，并以 LaTeX 语法形式呈现。
6. 在笔记末尾添加一段专业的 **AI 总结**，简要概括整个视频的核心内容。
"""


class LLMSummarizer:
    def __init__(self):
        self.client = OpenAI(
            api_key=AIConfig.API_KEY,
            base_url=AIConfig.API_URL,
        )
        self.model = AIConfig.MODEL_NAME

    def summarize(self, title: str, transcript: str) -> str:
        user_content = f"视频标题：{title}\n\n转录内容：\n{transcript}"

        logger.info(f"Calling LLM for summary (model={self.model}, text_len={len(transcript)})")

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
