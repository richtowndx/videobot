"""LLM summarisation via an OpenAI-compatible chat completions endpoint.

The model is asked to produce a structured Chinese Markdown note with three
sections: core points, detailed notes, and a summary paragraph.
"""

from __future__ import annotations

import logging

import openai

import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
你是一个专业的视频内容总结助手。请将提供的字幕或转录文本整理为结构清晰的中文笔记。

输出格式（严格遵循 Markdown）：

# [视频标题/主题]

## 核心要点
- 要点一
- 要点二
- 要点三（3-5 条，每条一句话）

## 详细笔记
（按逻辑分段，保留关键信息、数据、观点；每段给出小标题）

## 总结
（1-2 段，概括核心价值与最重要收获）
"""


def summarize(transcript: str, title: str = "") -> str:
    """Call the LLM and return structured Markdown notes.

    Parameters
    ----------
    transcript:
        The plain-text subtitle / Whisper transcript.
    title:
        Optional video title used as extra context for the model.
    """
    client = openai.OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_API_BASE,
    )

    user_parts: list[str] = []
    if title:
        user_parts.append(f"视频标题：{title}")
    user_parts.append(f"字幕/转录文本：\n{transcript}")
    user_content = "\n\n".join(user_parts)

    logger.info("summarising transcript (%d chars) with model %s", len(transcript), config.OPENAI_MODEL)

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=2048,
    )

    note: str = response.choices[0].message.content or ""
    logger.info("summary generated (%d chars)", len(note))
    return note
