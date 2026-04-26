from abc import ABC, abstractmethod


class BaseTranscriber(ABC):
    @abstractmethod
    def transcript(self, file_path: str) -> str:
        """Transcribe audio file and return full text."""
        pass
