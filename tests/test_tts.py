from pathlib import Path

import pytest

from wyoming_nemo_speech.libtts import ffi, library_path, speaker_map
from wyoming_nemo_speech.tts import resolve_voice, tts_program


def test_tts_program() -> None:
    program = tts_program(["John", "Sofia"], "0.1.0")
    assert program.name == "magpie"
    assert [voice.name for voice in program.voices] == ["John", "Sofia"]
    assert program.voices[0].languages == ["en"]
    assert program.voices[0].version == "0.1.0"


def test_resolve_voice() -> None:
    assert resolve_voice("sofia", ["John", "Sofia"]) == "Sofia"
    assert resolve_voice("nope", ["John", "Sofia"]) is None
    assert resolve_voice(None, ["John"]) is None


def test_installed_library_matches_cdef() -> None:
    prefix = Path.home() / ".local" / "share" / "nemo-speech"
    if library_path(prefix) is None:
        pytest.skip("NeMo-Speech library not installed")
    from wyoming_nemo_speech.libtts import load

    library = load(prefix)
    assert library.version()
    runtime = library.lib.nemo_speech_tts_runtime_config_default()
    options = library.lib.nemo_speech_tts_synthesis_options_default()
    assert runtime.size == ffi.sizeof("nemo_speech_tts_runtime_config")
    assert options.size == ffi.sizeof("nemo_speech_tts_synthesis_options")


def test_speaker_map_reads_the_model_file(tmp_path: Path) -> None:
    (tmp_path / "abc_magpie.nemo.speakers.json").write_text(
        '{"Aria": 0, "Jason": 1, "John": 2, "Leo": 3, "Sofia": 4}'
    )
    assert speaker_map(tmp_path) == {
        "Aria": 0,
        "Jason": 1,
        "John": 2,
        "Leo": 3,
        "Sofia": 4,
    }


def test_speaker_map_is_none_without_the_file(tmp_path: Path) -> None:
    assert speaker_map(tmp_path) is None
