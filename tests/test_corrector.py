import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
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


def test_correct_single_chunk_returns_content():
    mc = MagicMock()
    mc.chat.completions.create.return_value = _stream("纠错后的文本")
    s = _make_summarizer(mc)
    assert s.correct("原始文本") == "纠错后的文本"
    mc.chat.completions.create.assert_called_once()


def test_correct_uses_correction_system_prompt():
    mc = MagicMock()
    mc.chat.completions.create.return_value = _stream("x")
    s = _make_summarizer(mc)
    s.correct("text")
    from summarizer.llm import CORRECTION_SYSTEM_PROMPT
    call = mc.chat.completions.create.call_args
    assert call.kwargs["messages"][0]["content"] == CORRECTION_SYSTEM_PROMPT


def test_correct_does_not_set_max_tokens():
    """请求不限定上下文长度：纠错调用不传 max_tokens。"""
    mc = MagicMock()
    mc.chat.completions.create.return_value = _stream("x")
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
    mc.chat.completions.create.side_effect = [_stream("A"), _stream("B")]
    s = _make_summarizer(mc)
    s.correction_chunk_char_limit = 5  # 强制分块
    result = s.correct("0123456789")   # 10 chars -> 2 chunks
    assert result == "AB"


def test_correct_no_overlap_reconstructs_original():
    """overlap=0 + 逐字回显 => 拼接结果应能还原原文，无重复。"""
    def echo(*a, **kw):
        body = kw["messages"][1]["content"].split("\n", 1)[1]
        return _stream(body)
    mc = MagicMock()
    mc.chat.completions.create.side_effect = echo
    s = _make_summarizer(mc)
    s.correction_chunk_char_limit = 5
    text = "abcdefghij"
    assert s.correct(text) == text


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


if __name__ == "__main__":
    test_correct_single_chunk_returns_content()
    test_correct_uses_correction_system_prompt()
    test_correct_does_not_set_max_tokens()
    test_correct_empty_returns_empty_without_call()
    test_correct_multi_chunk_concatenates_in_order()
    test_correct_no_overlap_reconstructs_original()
    print("All corrector tests passed!")
