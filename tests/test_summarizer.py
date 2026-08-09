import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock


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


@patch("summarizer.llm.OpenAI")
@patch("summarizer.llm.AIConfig.load_models")
def test_summarize(mock_load_models, mock_openai_cls):
    from config import ModelConfig
    mock_load_models.return_value = [ModelConfig(name="test-model", url="http://test", key="test-key")]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _stream("# Test Summary\n\nThis is a summary.")
    mock_client.models.list.return_value = []
    mock_openai_cls.return_value = mock_client

    from summarizer.llm import LLMSummarizer
    s = LLMSummarizer()
    result = s.summarize("Test Video", "Some transcript text")

    assert "# Test Summary" in result
    mock_client.chat.completions.create.assert_called_once()

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert any("Test Video" in m["content"] for m in messages)
    assert any("transcript text" in m["content"] for m in messages)


@patch("summarizer.llm.OpenAI")
@patch("summarizer.llm.AIConfig.load_models")
def test_summarize_uses_config(mock_load_models, mock_openai_cls):
    from config import ModelConfig
    mock_load_models.return_value = [ModelConfig(name="test-model", url="http://test", key="test-key")]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _stream("Summary")
    mock_client.models.list.return_value = []
    mock_openai_cls.return_value = mock_client

    from summarizer.llm import LLMSummarizer
    s = LLMSummarizer()
    s.summarize("Title", "Text")

    mock_openai_cls.assert_called_once()
    call_kwargs = mock_openai_cls.call_args.kwargs
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["base_url"] == "http://test"


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


@patch("summarizer.llm.OpenAI")
@patch("summarizer.llm.AIConfig.load_models")
def test_check_connectivity_stops_at_first_reachable(mock_load_models, mock_openai_cls):
    """主 provider 可达时，只探主、不探兑底。"""
    from config import ModelConfig
    mock_load_models.return_value = [
        ModelConfig(name="primary", url="http://p", key="k"),
        ModelConfig(name="backup", url="http://b", key="k"),
    ]
    primary_client = MagicMock(); primary_client.models.list.return_value = []
    backup_client = MagicMock(); backup_client.models.list.return_value = []
    mock_openai_cls.side_effect = [primary_client, backup_client]

    from summarizer.llm import LLMSummarizer
    LLMSummarizer()  # __init__ -> _check_connectivity

    primary_client.models.list.assert_called_once()   # 探了主
    backup_client.models.list.assert_not_called()     # 兑底未探


@patch("summarizer.llm.OpenAI")
@patch("summarizer.llm.AIConfig.load_models")
def test_check_connectivity_falls_through_to_backup(mock_load_models, mock_openai_cls):
    """主不可达时，回落探兑底并停在其上。"""
    from config import ModelConfig
    mock_load_models.return_value = [
        ModelConfig(name="primary", url="http://p", key="k"),
        ModelConfig(name="backup", url="http://b", key="k"),
    ]
    primary_client = MagicMock(); primary_client.models.list.side_effect = RuntimeError("down")
    backup_client = MagicMock(); backup_client.models.list.return_value = []
    mock_openai_cls.side_effect = [primary_client, backup_client]

    from summarizer.llm import LLMSummarizer
    LLMSummarizer()

    primary_client.models.list.assert_called_once()   # 主不可达，探了
    backup_client.models.list.assert_called_once()    # 回落探 backup 并停


@patch("summarizer.llm.OpenAI")
@patch("summarizer.llm.AIConfig.load_models")
def test_check_connectivity_all_unreachable_no_crash(mock_load_models, mock_openai_cls):
    """全部不可达时不抛错（实际调用会在运行期失败）。"""
    from config import ModelConfig
    mock_load_models.return_value = [
        ModelConfig(name="primary", url="http://p", key="k"),
        ModelConfig(name="backup", url="http://b", key="k"),
    ]
    primary_client = MagicMock(); primary_client.models.list.side_effect = RuntimeError("down")
    backup_client = MagicMock(); backup_client.models.list.side_effect = RuntimeError("down")
    mock_openai_cls.side_effect = [primary_client, backup_client]

    from summarizer.llm import LLMSummarizer
    LLMSummarizer()  # 不抛
    primary_client.models.list.assert_called_once()
    backup_client.models.list.assert_called_once()


if __name__ == "__main__":
    test_summarize()
    test_summarize_uses_config()
    test_system_prompt_has_four_knowledge_fields()
    test_system_prompt_has_soft_fallback()
    test_refine_prompt_has_four_knowledge_fields()
    test_refine_prompt_has_soft_fallback()
    test_check_connectivity_stops_at_first_reachable()
    test_check_connectivity_falls_through_to_backup()
    test_check_connectivity_all_unreachable_no_crash()
    print("All summarizer tests passed!")
