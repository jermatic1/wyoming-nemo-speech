import array
import math
import wave
from pathlib import Path

import pytest

from wyoming_nemo_speech.models import speaker_model
from wyoming_nemo_speech.speakers import (
    SpeakerIdentifier,
    centroid,
    cosine,
    enroll,
    identify,
    prefix,
    report,
)


class FakeEncoder:
    """Embeds the mean and spread of the audio, so tones are tell-tale."""

    dim = 2

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed(self, samples: array.array, sample_rate: int) -> list[float]:
        self.calls.append(len(samples))
        mean = sum(samples) / len(samples)
        spread = math.sqrt(sum((s - mean) ** 2 for s in samples) / len(samples))
        return [mean, spread]


def write_wav(path: Path, samples: list[int], rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(array.array("h", samples).tobytes())


@pytest.fixture
def speakers(tmp_path: Path) -> Path:
    write_wav(tmp_path / "Jeremy" / "a.wav", [1000, 1000, 1200, 800])
    write_wav(tmp_path / "Jeremy" / "b.wav", [900, 1100, 1000, 1000])
    write_wav(tmp_path / "Jade" / "a.wav", [-1000, 1000, -1000, 1000])
    (tmp_path / "Jade" / "notes.txt").write_text("ignored")
    (tmp_path / "Empty").mkdir()
    return tmp_path


def test_enroll_builds_unit_centroids(speakers: Path) -> None:
    profiles = enroll(speakers, FakeEncoder())
    assert set(profiles) == {"Jeremy", "Jade"}
    for profile in profiles.values():
        assert math.sqrt(sum(v * v for v in profile)) == pytest.approx(1.0)


def test_enroll_skips_unreadable_files(speakers: Path, caplog) -> None:
    (speakers / "Jeremy" / "bad.wav").write_bytes(b"not a wav")
    profiles = enroll(speakers, FakeEncoder())
    assert "Jeremy" in profiles
    assert "Skipping" in caplog.text


def test_enroll_empty_directory(tmp_path: Path) -> None:
    assert enroll(tmp_path, FakeEncoder()) == {}


def test_identify_names_the_closest_above_threshold() -> None:
    profiles = {"Jeremy": centroid([[1.0, 0.0]]), "Jade": centroid([[0.0, 1.0]])}
    match = identify([0.9, 0.1], profiles, 0.6)
    assert match is not None
    assert match.name == "Jeremy"
    assert match.score == pytest.approx(cosine([0.9, 0.1], [1.0, 0.0]))
    assert identify([0.5, 0.5], profiles, 0.8) is None
    assert identify([1.0, 0.0], {}, 0.0) is None


def test_identifier_skips_short_audio() -> None:
    encoder = FakeEncoder()
    identifier = SpeakerIdentifier(encoder, {"Jeremy": [1.0, 0.0]}, 0.5, 1.0)
    assert identifier.identify_audio(array.array("f", [0.5] * 8000), 16000) is None
    assert encoder.calls == []
    match = identifier.identify_audio(array.array("f", [0.5] * 16000), 16000)
    assert match is not None
    assert match.name == "Jeremy"
    assert match.score == pytest.approx(1.0)


def test_report_lists_files_and_pairs(speakers: Path) -> None:
    lines = report(speakers, FakeEncoder())
    assert len(lines) == 4
    assert lines[0].startswith("Jade")
    assert lines[0].split()[1] == "a.wav"
    assert lines[-1].split()[:3] == ["Jade", "vs", "Jeremy"]
    assert 0.0 <= float(lines[-1].split()[-1]) <= 1.0


def test_prefix() -> None:
    assert (
        prefix("I'm {name}. {text}", "Jeremy", "lights on") == "I'm Jeremy. lights on"
    )
    assert prefix("I'm {name}. {text}", None, "lights on") == "lights on"
    assert prefix("", "Jeremy", "lights on") == "lights on"
    assert prefix("I'm {name}. {text}", "Jeremy", "") == ""


MODEL = (
    Path.home()
    / ".cache/nemo-speech/models/k2-fsa/sherpa-onnx/speaker-recongition-models"
    / "nemo_en_titanet_large.onnx"
)


@pytest.mark.model
@pytest.mark.skipif(not MODEL.is_file(), reason="TitaNet model not downloaded")
def test_titanet_embeds_speech_like_audio() -> None:
    from wyoming_nemo_speech.libspeaker import TitaNet

    encoder = TitaNet(speaker_model(str(MODEL)))
    assert encoder.dim == 192
    rate = 16000
    tone = array.array(
        "f", (0.3 * math.sin(2 * math.pi * 220 * i / rate) for i in range(3 * rate))
    )
    assert cosine(encoder.embed(tone, rate), encoder.embed(tone, rate)) > 0.99
