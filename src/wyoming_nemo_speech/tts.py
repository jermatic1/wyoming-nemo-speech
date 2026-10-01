"""What the TTS side advertises to Wyoming clients, and voice lookup."""

from __future__ import annotations

from typing import Protocol

from wyoming.info import TtsProgram, TtsVoice

from wyoming_nemo_speech import __version__
from wyoming_nemo_speech.asr import ATTRIBUTION
from wyoming_nemo_speech.libtts import PcmCallback


class Synthesizer(Protocol):
    sample_rate: int
    speakers: list[str]

    def synthesize(
        self,
        text: str,
        voice: str | None,
        language: str,
        on_pcm: PcmCallback,
    ) -> None: ...


def resolve_voice(requested: str | None, speakers: list[str]) -> str | None:
    if not requested:
        return None
    for speaker in speakers:
        if speaker.lower() == requested.lower():
            return speaker
    return None


def tts_program(speakers: list[str], version: str | None = None) -> TtsProgram:
    return TtsProgram(
        name="magpie",
        description="Magpie TTS",
        attribution=ATTRIBUTION,
        installed=True,
        version=__version__,
        voices=[
            TtsVoice(
                name=speaker,
                description=speaker,
                attribution=ATTRIBUTION,
                installed=True,
                languages=["en"],
                version=version,
            )
            for speaker in speakers
        ],
        supports_synthesize_streaming=False,
    )
