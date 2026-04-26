import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock


@patch("summarizer.llm.OpenAI")
def test_summarize(mock_openai_cls):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "# Test Summary\n\nThis is a summary."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    from summarizer.llm import LLMSummarizer
    s = LLMSummarizer()
    result = s.summarize("Test Video", "Some transcript text")

    assert "# Test Summary" in result
    mock_client.chat.completions.create.assert_called_once()

    # Verify the call arguments
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert any("Test Video" in m["content"] for m in messages)
    assert any("transcript text" in m["content"] for m in messages)


@patch("summarizer.llm.OpenAI")
def test_summarize_uses_config(mock_openai_cls):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Summary"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    from summarizer.llm import LLMSummarizer
    from config import AIConfig
    s = LLMSummarizer()
    s.summarize("Title", "Text")

    mock_openai_cls.assert_called_once_with(
        api_key=AIConfig.API_KEY,
        base_url=AIConfig.API_URL,
    )


if __name__ == "__main__":
    test_summarize()
    test_summarize_uses_config()
    print("All summarizer tests passed!")
