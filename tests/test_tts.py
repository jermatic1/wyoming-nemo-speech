from pathlib import Path

import pytest

from wyoming_nemo_speech.libtts import ffi, library_path
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
