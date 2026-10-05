from pathlib import Path

import pytest

from wyoming_nemo_speech import __main__ as main
from wyoming_nemo_speech.config import Speakers

from .test_speakers import FakeEncoder, write_wav


@pytest.fixture
def enrolled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    write_wav(tmp_path / "Jeremy" / "a.wav", [1000, 1000, 1200, 800])
    write_wav(tmp_path / "Jade" / "a.wav", [-1000, 1000, -1000, 1000])
    monkeypatch.setattr(main, "TitaNet", lambda path: FakeEncoder())
    monkeypatch.setattr(main, "speaker_model", lambda name: Path(name))
    return tmp_path


def test_report_speakers_prints_scores(
    enrolled: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(enrolled)
    (enrolled / "config.toml").write_text(f'[speakers]\ndir = "{enrolled}"\n')
    assert main.report_speakers() == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split()[:2] == ["Jade", "a.wav"]
    assert out[-1].split()[:3] == ["Jade", "vs", "Jeremy"]


def test_report_speakers_without_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main.report_speakers() == 1
    assert "No speakers directory" in capsys.readouterr().err


def test_load_speakers(enrolled: Path) -> None:
    identifier = main.load_speakers(Speakers(dir=enrolled, model="fake"))
    assert identifier is not None
    assert set(identifier.profiles) == {"Jeremy", "Jade"}
    assert main.load_speakers(Speakers(dir=enrolled, enabled=False)) is None
    assert main.load_speakers(Speakers(dir=enrolled / "missing")) is None


def test_load_speakers_ignores_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(path: Path) -> FakeEncoder:
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(main, "TitaNet", fail)
    monkeypatch.setattr(main, "speaker_model", lambda name: Path(name))
    assert main.load_speakers(Speakers(dir=tmp_path)) is None
