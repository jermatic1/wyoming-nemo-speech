from pathlib import Path

import pytest

from wyoming_nemo_speech.config import Config, load


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    config = load(tmp_path / "config.toml")
    assert config == Config()
    assert config.uri == "tcp://0.0.0.0:10400"
    assert config.asr.model == "nemotron-en"
    assert config.home_assistant.token == ""


def test_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'lib_dir = "~/nemo"\ndevice = "vulkan:0"\n\n'
        "[asr]\nspeech_context_boost = 4.0\n\n"
        '[home_assistant]\ntoken = "abc"\n'
    )
    config = load(path)
    assert config.lib_dir == Path.home() / "nemo"
    assert config.device == "vulkan:0"
    assert config.asr.speech_context_boost == 4.0
    assert config.asr.model == "nemotron-en"
    assert config.home_assistant.token == "abc"
    assert config.home_assistant.url == "http://homeassistant.local:8123"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("port = 1\n")
    with pytest.raises(ValueError, match="port"):
        load(path)
    path.write_text("[asr]\nlanguage = 'en'\n")
    with pytest.raises(ValueError, match="asr.language"):
        load(path)


def test_speakers_section(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[speakers]\ndir = "~/voices"\nthreshold = 0.7\nprefix = ""\n')
    config = load(path)
    assert config.speakers.dir == Path.home() / "voices"
    assert config.speakers.threshold == 0.7
    assert config.speakers.prefix == ""
    assert config.speakers.enabled is True
    assert config.speakers.model == "titanet-large"
    path.write_text("[speakers]\nvoices = 'x'\n")
    with pytest.raises(ValueError, match="speakers.voices"):
        load(path)


def test_tts_local_model(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[tts]\nmodel_path = "~/voices/magpie-home.f16.gguf"\n'
        'tokenizer_dir = "~/voices/nemo"\n'
    )
    config = load(path)
    assert config.tts.model == "magpie"
    assert config.tts.model_path == Path.home() / "voices" / "magpie-home.f16.gguf"
    assert config.tts.tokenizer_dir == Path.home() / "voices" / "nemo"
    assert config.tts.codec_path is None
    path.write_text('[tts]\nmodel_path = "magpie.gguf"\n')
    with pytest.raises(ValueError, match="tokenizer_dir"):
        load(path)
