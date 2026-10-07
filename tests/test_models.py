import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from wyoming_nemo_speech import models
from wyoming_nemo_speech.models import asr_model, speaker_model, tts_models

ASR_REPO = "nvidia/nemotron-speech-streaming-en-0.6b"
INDEX = {
    "models": [
        {
            "repo": ASR_REPO,
            "aliases": ["nemotron-en"],
            "revision": "r1",
            "artifacts": [{"role": "asr", "type": "file", "filename": "asr.gguf"}],
        },
        {
            "repo": "nvidia/magpie_tts_multilingual_357m",
            "aliases": ["magpie"],
            "revision": "r2",
            "companions": ["nvidia/nemo-nano-codec"],
            "artifacts": [
                {"role": "tts", "type": "file", "filename": "magpie.gguf"},
                {"role": "tokenizer", "type": "tar-prefix", "directory": "tokenizer"},
            ],
        },
        {
            "repo": "nvidia/nemo-nano-codec",
            "aliases": [],
            "revision": "r3",
            "artifacts": [{"role": "codec", "type": "file", "filename": "codec.gguf"}],
        },
    ]
}


@pytest.fixture
def lib_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    prefix = tmp_path / "prefix"
    index = prefix / "share" / "nemo-speech" / "model-index.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps(INDEX))
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(tmp_path / "cache"))
    return prefix


def _asr_path(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "nvidia" / ASR_REPO.split("/")[1] / "r1" / "asr.gguf"


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"gguf")


def test_asr_model_from_cache(lib_dir: Path, tmp_path: Path) -> None:
    _touch(_asr_path(tmp_path))
    assert asr_model("nemotron-en", lib_dir) == _asr_path(tmp_path)


def test_tts_models_from_cache(lib_dir: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "nvidia"
    magpie = cache / "magpie_tts_multilingual_357m" / "r2" / "magpie.gguf"
    tokenizer = magpie.parent / "tokenizer"
    codec = cache / "nemo-nano-codec" / "r3" / "codec.gguf"
    _touch(magpie)
    _touch(codec)
    tokenizer.mkdir()
    assert tts_models("magpie", lib_dir) == (magpie, codec, tokenizer)


def test_missing_model_is_pulled(lib_dir: Path, tmp_path: Path) -> None:
    script = lib_dir / "bin" / "nemo-speech"
    script.parent.mkdir()
    target = _asr_path(tmp_path)
    script.write_text(
        f'#!/bin/sh\necho "$@" > "{tmp_path}/pulled"\n'
        f'mkdir -p "{target.parent}" && echo gguf > "{target}"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    assert asr_model("nemotron-en", lib_dir) == target
    assert (tmp_path / "pulled").read_text().split() == ["pull", "nemotron-en"]


def test_unknown_model_is_not_pulled(lib_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="unknown model"):
        asr_model("whisper", lib_dir)
    assert not os.path.exists(tmp_path / "pulled")


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def read(self, size: int) -> bytes:
        body, self._body = self._body[:size], self._body[size:]
        return body


def test_speaker_model_is_downloaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"onnx"
    sha256 = hashlib.sha256(body).hexdigest()
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(models, "SPEAKER_MODELS", {"tiny": ("tiny.onnx", sha256)})
    urls: list[str] = []

    def urlopen(url: str) -> FakeResponse:
        urls.append(url)
        return FakeResponse(body)

    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    path = speaker_model("tiny")
    assert path.read_bytes() == body
    assert path.parent == (
        tmp_path / "cache" / "k2-fsa" / "sherpa-onnx" / "speaker-recongition-models"
    )
    assert speaker_model("Tiny") == path
    assert len(urls) == 1


def test_speaker_model_rejects_bad_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMO_SPEECH_MODEL_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(models, "SPEAKER_MODELS", {"tiny": ("tiny.onnx", "00")})
    monkeypatch.setattr(
        models.urllib.request, "urlopen", lambda url: FakeResponse(b"onnx")
    )
    with pytest.raises(ValueError, match="checksum"):
        speaker_model("tiny")
    assert list((tmp_path / "cache").rglob("*")) == [
        tmp_path / "cache" / "k2-fsa",
        tmp_path / "cache" / "k2-fsa" / "sherpa-onnx",
        tmp_path / "cache" / "k2-fsa" / "sherpa-onnx" / "speaker-recongition-models",
    ]


def test_speaker_model_accepts_a_path(tmp_path: Path) -> None:
    path = tmp_path / "custom.onnx"
    path.write_bytes(b"onnx")
    assert speaker_model(str(path)) == path
    with pytest.raises(FileNotFoundError, match="unknown speaker model"):
        speaker_model("ecapa")


def test_tts_local_model_uses_index_codec(lib_dir: Path, tmp_path: Path) -> None:
    codec = tmp_path / "cache" / "nvidia" / "nemo-nano-codec" / "r3" / "codec.gguf"
    _touch(codec)
    model = tmp_path / "custom" / "magpie-home.f16.gguf"
    tokenizer = tmp_path / "custom" / "nemo"
    _touch(model)
    tokenizer.mkdir()
    assert tts_models("magpie", lib_dir, model_path=model, tokenizer_dir=tokenizer) == (
        model,
        codec,
        tokenizer,
    )
    own_codec = tmp_path / "custom" / "codec.gguf"
    _touch(own_codec)
    result = tts_models(
        "magpie",
        lib_dir,
        model_path=model,
        tokenizer_dir=tokenizer,
        codec_path=own_codec,
    )
    assert result == (model, own_codec, tokenizer)


def test_tts_local_model_must_exist(lib_dir: Path, tmp_path: Path) -> None:
    tokenizer = tmp_path / "nemo"
    tokenizer.mkdir()
    with pytest.raises(FileNotFoundError, match="tts.model_path"):
        tts_models(
            "magpie",
            lib_dir,
            model_path=tmp_path / "missing.gguf",
            tokenizer_dir=tokenizer,
        )
