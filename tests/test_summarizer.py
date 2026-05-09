import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock


@patch("summarizer.llm.OpenAI")
@patch("summarizer.llm.AIConfig.load_models")
def test_summarize(mock_load_models, mock_openai_cls):
    from config import ModelConfig
    mock_load_models.return_value = [ModelConfig(name="test-model", url="http://test", key="test-key")]

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "# Test Summary\n\nThis is a summary."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
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
    mock_choice = MagicMock()
    mock_choice.message.content = "Summary"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_client.models.list.return_value = []
    mock_openai_cls.return_value = mock_client

    from summarizer.llm import LLMSummarizer
    s = LLMSummarizer()
    s.summarize("Title", "Text")

    mock_openai_cls.assert_called_once()
    call_kwargs = mock_openai_cls.call_args.kwargs
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["base_url"] == "http://test"


if __name__ == "__main__":
    test_summarize()
    test_summarize_uses_config()
    print("All summarizer tests passed!")
