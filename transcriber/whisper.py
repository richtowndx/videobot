import os
import gc
import logging
import tempfile
import resource
from typing import Optional

from faster_whisper import WhisperModel

from transcriber.base import BaseTranscriber
from config import WhisperConfig, DataConfig

logger = logging.getLogger(__name__)


def _mem_mb():
    """Get current memory usage in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

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

        # Validate existing directory matches expected model size
        if model_path.exists():
            # Check if it looks like a valid model directory (has bin/model files)
            # If model_size changed (e.g., base->tiny), old dir may be wrong model
            import shutil
            if not self._is_valid_model_dir(model_path, model_size):
                logger.warning(f"Model directory {model_path} exists but may be wrong model, removing...")
                shutil.rmtree(model_path)

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
        mem_before = _mem_mb()
        self.model = WhisperModel(
            model_size_or_path=str(model_path),
            device=WhisperConfig.DEVICE,
            compute_type=WhisperConfig.COMPUTE_TYPE,
            num_workers=1,
        )
        gc.collect()
        mem_after = _mem_mb()
        logger.info(f"WhisperModel loaded [mem: {mem_after:.0f}MB, delta: {mem_after-mem_before:.0f}MB]")

    def _is_valid_model_dir(self, path, expected_size: str) -> bool:
        """Check if model directory matches expected model size."""
        # Simple check: directory name should contain expected size
        return expected_size in path.name

    def transcript(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        file_size = os.path.getsize(file_path) / 1024 / 1024
        mem_start = _mem_mb()
        logger.info(f"Transcribing: {file_path} ({file_size:.1f}MB) [mem: {mem_start:.0f}MB]")

        duration = self._get_duration(file_path)

        # Long audio: segment directly from source, no full preprocess needed
        if duration > MAX_SEGMENT_DURATION:
            return self._transcribe_segments(file_path, duration)

        # Short audio: preprocess if large, then transcribe in one pass
        processed = self._preprocess_audio(file_path)
        try:
            return self._transcribe_direct(processed)
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

        num_segments = int(total_duration / MAX_SEGMENT_DURATION) + 1
        temp_dir = tempfile.mkdtemp(prefix="whisper_seg_")
        all_texts = []

        try:
            for i in range(num_segments):
                seg_path = os.path.join(temp_dir, f"seg_{i:04d}.wav")

                # Split one segment
                start = i * MAX_SEGMENT_DURATION
                try:
                    (
                        ffmpeg.input(file_path, ss=start, t=MAX_SEGMENT_DURATION)
                        .output(seg_path, acodec="pcm_s16le", ar=16000, ac=1, y=None)
                        .overwrite_output()
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                    if not os.path.exists(seg_path) or os.path.getsize(seg_path) == 0:
                        logger.warning(f"Segment {i+1}/{num_segments} is empty, skipping")
                        continue
                except Exception as e:
                    logger.warning(f"Segment {i+1}/{num_segments} split failed: {e}")
                    continue

                # Transcribe immediately, then release memory
                seg_mem_start = _mem_mb()
                logger.info(f"Transcribing segment {i+1}/{num_segments} [mem: {seg_mem_start:.0f}MB]")
                gc.collect()
                try:
                    segments, info = self.model.transcribe(seg_path)
                    texts = [seg.text.strip() for seg in segments if seg.text.strip()]
                    all_texts.extend(texts)
                except Exception as e:
                    logger.error(f"Segment {i+1}/{num_segments} transcription failed: {e}")
                finally:
                    try:
                        os.remove(seg_path)
                    except OSError:
                        pass
                gc.collect()
                seg_mem_end = _mem_mb()
                logger.info(f"Segment {i+1}/{num_segments} done [mem: {seg_mem_end:.0f}MB, delta: {seg_mem_end-seg_mem_start:.0f}MB]")

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
