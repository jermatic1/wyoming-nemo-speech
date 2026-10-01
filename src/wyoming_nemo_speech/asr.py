"""What the ASR side advertises to Wyoming clients."""

from __future__ import annotations

import array
from typing import Protocol

from wyoming.info import AsrModel, AsrProgram, Attribution

from wyoming_nemo_speech import __version__

ATTRIBUTION = Attribution(
    name="NVIDIA", url="https://github.com/NVIDIA/NeMo-Speech.cpp"
)


class Recognizer(Protocol):
    def recognize(
        self,
        samples: array.array,
        sample_rate: int,
        language: str,
        phrases: list[str] | None,
        boost: float,
    ) -> str: ...


def asr_program(model: str, version: str | None = None) -> AsrProgram:
    return AsrProgram(
        name="nemotron",
        description="NeMo Speech recognition",
        attribution=ATTRIBUTION,
        installed=True,
        version=__version__,
        models=[
            AsrModel(
                name=model,
                description=model,
                attribution=ATTRIBUTION,
                installed=True,
                languages=["en"],
                version=version,
            )
        ],
        supports_transcript_streaming=False,
        requires_external_vad=True,
    )
