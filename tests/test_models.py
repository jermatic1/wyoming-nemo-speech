import json
import os
import stat
from pathlib import Path

import pytest

from wyoming_nemo_speech.models import asr_model, tts_models

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
