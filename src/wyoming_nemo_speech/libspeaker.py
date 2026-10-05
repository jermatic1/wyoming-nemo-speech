"""TitaNet speaker embeddings through sherpa-onnx."""

from __future__ import annotations

import array
import os
import threading
from pathlib import Path

import sherpa_onnx


class TitaNet:
    def __init__(self, model_path: Path, num_threads: int | None = None) -> None:
        threads = num_threads or min(4, os.cpu_count() or 1)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path), num_threads=threads, provider="cpu"
        )
        if not config.validate():
            raise ValueError(f"invalid speaker model: {model_path}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._lock = threading.Lock()
        self.dim: int = self._extractor.dim

    def embed(self, samples: array.array, sample_rate: int) -> list[float]:
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()
            if not self._extractor.is_ready(stream):
                raise ValueError("not enough audio for a speaker embedding")
            return list(self._extractor.compute(stream))
