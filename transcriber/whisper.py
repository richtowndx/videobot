import os
import gc
import logging
import tempfile
from typing import Optional

from faster_whisper import WhisperModel

from transcriber.base import BaseTranscriber
from config import WhisperConfig, DataConfig

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "tiny": "pengzhendong/faster-whisper-tiny",
    "base": "pengzhendong/faster-whisper-base",
    "small": "pengzhendong/faster-whisper-small",
    "medium": "pengzhendong/faster-whisper-medium",
}

MAX_SEGMENT_DURATION = 1200  # 20 minutes


class WhisperTranscriber(BaseTranscriber):
    def __init__(self):
        model_size = WhisperConfig.MODEL_SIZE
        model_dir = DataConfig.MODELS_DIR
        model_path = model_dir / f"whisper-{model_size}"

        if not model_path.exists():
            logger.info(f"Downloading whisper-{model_size} model...")
            try:
                from modelscope import snapshot_download
                model_path_str = snapshot_download(
                    MODEL_MAP[model_size], local_dir=str(model_path),
                )
                model_path = type(model_path)(model_path_str)
            except ImportError:
                logger.warning("modelscope not available, using faster-whisper auto-download")
                model_path = model_size

        logger.info(f"Loading WhisperModel: {model_path}")
        self.model = WhisperModel(
            model_size_or_path=str(model_path),
            device=WhisperConfig.DEVICE,
            compute_type=WhisperConfig.COMPUTE_TYPE,
            num_workers=1,
        )
        gc.collect()
        logger.info("WhisperModel loaded")

    def transcript(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        file_size = os.path.getsize(file_path) / 1024 / 1024
        logger.info(f"Transcribing: {file_path} ({file_size:.1f}MB)")

        # Preprocess audio
        processed = self._preprocess_audio(file_path)

        try:
            duration = self._get_duration(processed)
            if duration > MAX_SEGMENT_DURATION:
                text = self._transcribe_segments(processed, duration)
            else:
                text = self._transcribe_direct(processed)
            return text
        finally:
            if processed != file_path and os.path.exists(processed):
                os.remove(processed)

    def _transcribe_direct(self, file_path: str) -> str:
        segments, info = self.model.transcribe(file_path)
        logger.info(f"Detected language: {info.language}")
        texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(texts)

    def _transcribe_segments(self, file_path: str, total_duration: float) -> str:
        import ffmpeg

        segment_files = []
        temp_dir = tempfile.mkdtemp(prefix="whisper_seg_")

        try:
            # Split audio into chunks
            num_segments = int(total_duration / MAX_SEGMENT_DURATION) + 1
            for i in range(num_segments):
                start = i * MAX_SEGMENT_DURATION
                seg_path = os.path.join(temp_dir, f"seg_{i:04d}.wav")
                try:
                    (
                        ffmpeg.input(file_path, ss=start, t=MAX_SEGMENT_DURATION)
                        .output(seg_path, acodec="pcm_s16le", ar=16000, ac=1, y=None)
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                    if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
                        segment_files.append(seg_path)
                except Exception as e:
                    logger.warning(f"Segment {i} split failed: {e}")

            # Transcribe each segment
            all_texts = []
            for i, seg_path in enumerate(segment_files):
                logger.info(f"Transcribing segment {i+1}/{len(segment_files)}")
                gc.collect()
                try:
                    segments, info = self.model.transcribe(seg_path)
                    texts = [seg.text.strip() for seg in segments if seg.text.strip()]
                    all_texts.extend(texts)
                except Exception as e:
                    logger.error(f"Segment {i+1} transcription failed: {e}")
                finally:
                    try:
                        os.remove(seg_path)
                    except OSError:
                        pass

            return " ".join(all_texts)

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _preprocess_audio(self, file_path: str) -> str:
        import ffmpeg

        file_size = os.path.getsize(file_path) / 1024 / 1024
        if file_size < 100:
            return file_path

        logger.info(f"Preprocessing large audio ({file_size:.1f}MB)")
        temp_dir = tempfile.mkdtemp(prefix="audio_pp_")
        output = os.path.join(temp_dir, "processed.wav")

        try:
            (
                ffmpeg.input(file_path)
                .output(output, acodec="pcm_s16le", ar=16000, ac=1, y=None)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}, using original")
            return file_path

    def _get_duration(self, file_path: str) -> float:
        import ffmpeg
        try:
            probe = ffmpeg.probe(file_path)
            return float(probe["format"]["duration"])
        except Exception:
            return 0
