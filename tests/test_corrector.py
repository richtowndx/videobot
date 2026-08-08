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
