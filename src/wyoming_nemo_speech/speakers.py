"""Who is speaking: enrollment from WAV files and cosine matching."""

from __future__ import annotations

import array
import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from wyoming_nemo_speech.libasr import pcm16_to_f32

_LOGGER = logging.getLogger(__name__)

Embedding = list[float]
Profiles = dict[str, Embedding]


class SpeakerEncoder(Protocol):
    dim: int

    def embed(self, samples: array.array, sample_rate: int) -> Embedding: ...


@dataclass(frozen=True)
class Match:
    name: str
    score: float


class Identifier(Protocol):
    def identify_audio(
        self, samples: array.array, sample_rate: int
    ) -> Match | None: ...


class SpeakerIdentifier:
    """Names the enrolled speaker closest to an utterance, if close enough."""

    def __init__(
        self,
        encoder: SpeakerEncoder,
        profiles: Profiles,
        threshold: float,
        min_seconds: float,
    ) -> None:
        self._encoder = encoder
        self.profiles = profiles
        self.threshold = threshold
        self.min_seconds = min_seconds

    def identify_audio(self, samples: array.array, sample_rate: int) -> Match | None:
        if len(samples) < self.min_seconds * sample_rate:
            return None
        embedding = self._encoder.embed(samples, sample_rate)
        return identify(embedding, self.profiles, self.threshold)


def read_wav(path: Path) -> tuple[array.array, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        frames = wav.readframes(wav.getnframes())
        return pcm16_to_f32(frames, wav.getnchannels()), wav.getframerate()


def embed_files(
    directory: Path, encoder: SpeakerEncoder
) -> dict[str, list[tuple[Path, Embedding]]]:
    """One embedding per readable WAV under <directory>/<Name>/."""
    files: dict[str, list[tuple[Path, Embedding]]] = {}
    for person in sorted(path for path in directory.iterdir() if path.is_dir()):
        embeddings: list[tuple[Path, Embedding]] = []
        for wav in sorted(person.glob("*.wav")):
            try:
                samples, rate = read_wav(wav)
                embeddings.append((wav, encoder.embed(samples, rate)))
            except Exception:
                _LOGGER.exception("Skipping %s", wav)
        if embeddings:
            files[person.name] = embeddings
        else:
            _LOGGER.warning("No usable WAV files in %s", person)
    return files


def enroll(directory: Path, encoder: SpeakerEncoder) -> Profiles:
    return {
        name: centroid([embedding for _, embedding in items])
        for name, items in embed_files(directory, encoder).items()
    }


def centroid(embeddings: list[Embedding]) -> Embedding:
    count = len(embeddings)
    return normalize([sum(column) / count for column in zip(*embeddings)])


def normalize(vector: Embedding) -> Embedding:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


def cosine(a: Embedding, b: Embedding) -> float:
    return sum(x * y for x, y in zip(normalize(a), normalize(b)))


def identify(
    embedding: Embedding, profiles: Profiles, threshold: float
) -> Match | None:
    best: Match | None = None
    for name, profile in profiles.items():
        score = cosine(embedding, profile)
        if best is None or score > best.score:
            best = Match(name, score)
    if best is None or best.score < threshold:
        return None
    return best


def report(directory: Path, encoder: SpeakerEncoder) -> list[str]:
    """Lines showing how well each file and each person separate."""
    files = embed_files(directory, encoder)
    profiles = {
        name: centroid([embedding for _, embedding in items])
        for name, items in files.items()
    }
    lines = [
        f"{name:<12} {wav.name:<32} {cosine(embedding, profiles[name]):.2f}"
        for name, items in files.items()
        for wav, embedding in items
    ]
    names = list(profiles)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            lines.append(f"{a:<12} vs {b:<29} {cosine(profiles[a], profiles[b]):.2f}")
    return lines


def prefix(template: str, name: str | None, text: str) -> str:
    """Put the speaker's name into the transcript, if there is one."""
    if not template or not name or not text:
        return text
    return template.format(name=name, text=text)
