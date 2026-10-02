"""One Wyoming connection: audio in, transcript out; text in, PCM out."""

from __future__ import annotations

import array
import asyncio
import logging

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

from wyoming_nemo_speech.asr import Recognizer
from wyoming_nemo_speech.libasr import pcm16_to_f32
from wyoming_nemo_speech.names import HassNameCache
from wyoming_nemo_speech.normalize import spoken
from wyoming_nemo_speech.tts import Synthesizer, resolve_voice

_LOGGER = logging.getLogger(__name__)

LANGUAGE = "en-US"


class SpeechEventHandler(AsyncEventHandler):
    def __init__(
        self,
        recognizer: Recognizer,
        synthesizer: Synthesizer,
        info: Info,
        boost: float,
        names: HassNameCache | None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._recognizer = recognizer
        self._synthesizer = synthesizer
        self._info = info.event()
        self._boost = boost
        self._names = names
        self._rate: int | None = None
        self._samples: list[array.array] = []

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info)
        elif AudioStart.is_type(event.type):
            self._rate = None
            self._samples = []
        elif AudioChunk.is_type(event.type):
            self._append(AudioChunk.from_event(event))
        elif AudioStop.is_type(event.type):
            await self._transcribe()
            return False
        elif Synthesize.is_type(event.type):
            await self._synthesize(Synthesize.from_event(event))
        return True

    def _append(self, chunk: AudioChunk) -> None:
        if chunk.width != 2:
            raise ValueError(f"expected 16-bit PCM, got width {chunk.width}")
        if self._rate is None:
            self._rate = chunk.rate
        elif chunk.rate != self._rate:
            raise ValueError(f"sample rate changed from {self._rate} to {chunk.rate}")
        self._samples.append(pcm16_to_f32(chunk.audio, chunk.channels))

    async def _transcribe(self) -> None:
        text = ""
        if self._samples and self._rate:
            phrases = self._names.phrases() if self._names is not None else []
            try:
                text = await asyncio.to_thread(
                    self._recognizer.recognize,
                    _concat(self._samples),
                    self._rate,
                    LANGUAGE,
                    phrases or None,
                    self._boost,
                )
            except Exception:
                _LOGGER.exception("Recognition failed")
        _LOGGER.info("Transcript: %s", text)
        await self.write_event(Transcript(text=text, language=LANGUAGE).event())
        if self._names is not None:
            self._names.maybe_refresh()

    async def _synthesize(self, request: Synthesize) -> None:
        requested = None
        if request.voice is not None:
            requested = request.voice.name or request.voice.speaker
        voice = resolve_voice(requested, self._synthesizer.speakers)
        text = spoken(request.text)
        _LOGGER.debug("Synthesizing: %s", text)
        rate = self._synthesizer.sample_rate
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        def on_pcm(pcm: bytes) -> bool:
            loop.call_soon_threadsafe(queue.put_nowait, pcm)
            return True

        async def run() -> None:
            try:
                await asyncio.to_thread(
                    self._synthesizer.synthesize, text, voice, LANGUAGE, on_pcm
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(run())
        await self.write_event(AudioStart(rate, 2, 1).event())
        while (pcm := await queue.get()) is not None:
            await self.write_event(AudioChunk(rate, 2, 1, pcm).event())
        try:
            await task
        except Exception:
            _LOGGER.exception("Synthesis failed")
        await self.write_event(AudioStop().event())


def _concat(chunks: list[array.array]) -> array.array:
    out = array.array("f")
    for chunk in chunks:
        out.extend(chunk)
    return out
