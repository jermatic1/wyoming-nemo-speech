from pathlib import Path

import pytest

from wyoming_nemo_speech.asr import asr_program
from wyoming_nemo_speech.libasr import ffi, gpu_index, library_path, pcm16_to_f32


def pcm16(*samples: int) -> bytes:
    import array

    return array.array("h", samples).tobytes()


def test_asr_program() -> None:
    program = asr_program("nemotron-en", "0.1.0")
    assert program.name == "nemotron"
    assert program.requires_external_vad is True
    assert program.models[0].name == "nemotron-en"
    assert program.models[0].version == "0.1.0"


def test_pcm16_to_f32_scales() -> None:
    assert list(pcm16_to_f32(pcm16(16384, -32768), channels=1)) == [0.5, -1.0]


def test_pcm16_to_f32_downmixes_stereo() -> None:
    assert list(pcm16_to_f32(pcm16(16384, -16384), channels=2)) == [0.0]


def test_gpu_index() -> None:
    assert gpu_index("cpu") == -1
    assert gpu_index("CPU") == -1
    assert gpu_index("vulkan:0") == 0
    assert gpu_index("vulkan") == 0
    assert gpu_index("cuda:1") == 1
    with pytest.raises(ValueError):
        gpu_index("vulkan:-1")
    with pytest.raises(ValueError):
        gpu_index("tpu")


def test_library_path_missing(tmp_path: Path) -> None:
    assert library_path(tmp_path) is None


def test_installed_library_matches_cdef() -> None:
    prefix = Path.home() / ".local" / "share" / "nemo-speech"
    if library_path(prefix) is None:
        pytest.skip("NeMo-Speech library not installed")
    from wyoming_nemo_speech.libasr import load

    library = load(prefix)
    assert library.version()
    options = library.lib.nemo_speech_asr_recognition_options_default()
    assert options.size == ffi.sizeof("nemo_speech_asr_recognition_options")
