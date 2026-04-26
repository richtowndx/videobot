import os
import logging
import tempfile

import ffmpeg

logger = logging.getLogger(__name__)


def preprocess_audio(input_file: str, max_size_mb: int = 100) -> str:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Audio file not found: {input_file}")

    file_size_mb = os.path.getsize(input_file) / 1024 / 1024
    if file_size_mb <= max_size_mb:
        return input_file

    logger.info(f"Preprocessing audio: {file_size_mb:.1f}MB → 16kHz mono WAV")
    temp_dir = tempfile.mkdtemp(prefix="audio_pp_")
    output = os.path.join(temp_dir, "processed.wav")

    try:
        (
            ffmpeg.input(input_file)
            .output(output, acodec="pcm_s16le", ar=16000, ac=1, y=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        out_size = os.path.getsize(output) / 1024 / 1024
        logger.info(f"Preprocessed: {file_size_mb:.1f}MB → {out_size:.1f}MB")
        return output
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}")
        return input_file
