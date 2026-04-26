import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock


@patch("transcriber.whisper.WhisperModel")
def test_whisper_transcribe_direct(mock_model_cls):
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " Hello world "
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    mock_model_cls.return_value = mock_model

    from transcriber.whisper import WhisperTranscriber
    t = WhisperTranscriber()

    result = t._transcribe_direct("fake.wav")
    assert "Hello world" in result


@patch("transcriber.whisper.WhisperModel")
def test_whisper_transcript_file_not_found(mock_model_cls):
    mock_model_cls.return_value = MagicMock()

    from transcriber.whisper import WhisperTranscriber
    t = WhisperTranscriber()

    try:
        t.transcript("/nonexistent/file.wav")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


@patch("transcriber.whisper.WhisperModel")
def test_whisper_transcribe_filters_empty(mock_model_cls):
    mock_model = MagicMock()
    seg1 = MagicMock()
    seg1.text = " Real text "
    seg2 = MagicMock()
    seg2.text = "  "
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model.transcribe.return_value = ([seg1, seg2], mock_info)
    mock_model_cls.return_value = mock_model

    from transcriber.whisper import WhisperTranscriber
    t = WhisperTranscriber()
    result = t._transcribe_direct("fake.wav")
    assert result == "Real text"
    assert "  " not in result


if __name__ == "__main__":
    test_whisper_transcribe_direct()
    test_whisper_transcript_file_not_found()
    test_whisper_transcribe_filters_empty()
    print("All transcriber tests passed!")
