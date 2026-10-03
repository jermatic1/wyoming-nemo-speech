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
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from wyoming_nemo_speech.asr import Recognizer
from wyoming_nemo_speech.libasr import pcm16_to_f32
from wyoming_nemo_speech.names import HassNameCache
from wyoming_nemo_speech.normalize import sentences, spoken
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
        self._tts_voice: str | None = None
        self._tts_text: str | None = None

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
        # A stream also sends its full text for servers without chunk support.
        # The chunks already covered it, so it is ignored mid-stream.
        elif Synthesize.is_type(event.type) and self._tts_text is None:
            await self._synthesize(Synthesize.from_event(event))
        elif SynthesizeStart.is_type(event.type):
            await self._start_stream(SynthesizeStart.from_event(event))
        elif SynthesizeChunk.is_type(event.type):
            await self._stream_chunk(SynthesizeChunk.from_event(event))
        elif SynthesizeStop.is_type(event.type):
            await self._stop_stream()
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
        voice = self._voice(request.voice)
        complete, rest = sentences(request.text)
        await self.write_event(self._audio_start())
        for sentence in [*complete, rest]:
            await self._speak(sentence, voice)
        await self.write_event(AudioStop().event())

    async def _start_stream(self, request: SynthesizeStart) -> None:
        self._tts_voice = self._voice(request.voice)
        self._tts_text = ""
        await self.write_event(self._audio_start())

    async def _stream_chunk(self, chunk: SynthesizeChunk) -> None:
        if self._tts_text is None:
            _LOGGER.debug("Ignoring text chunk outside a synthesis stream")
            return
        complete, self._tts_text = sentences(self._tts_text + chunk.text)
        for sentence in complete:
            await self._speak(sentence, self._tts_voice)

    async def _stop_stream(self) -> None:
        if self._tts_text is None:
            _LOGGER.debug("Ignoring stop outside a synthesis stream")
            return
        rest, self._tts_text = self._tts_text, None
        await self._speak(rest, self._tts_voice)
        await self.write_event(AudioStop().event())
        await self.write_event(SynthesizeStopped().event())

    def _voice(self, requested: SynthesizeVoice | None) -> str | None:
        name = None
        if requested is not None:
            name = requested.name or requested.speaker
        return resolve_voice(name, self._synthesizer.speakers)

    def _audio_start(self) -> Event:
        return AudioStart(self._synthesizer.sample_rate, 2, 1).event()

    async def _speak(self, text: str, voice: str | None) -> None:
        text = spoken(" ".join(text.split()))
        if not text:
            return
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
        while (pcm := await queue.get()) is not None:
            await self.write_event(AudioChunk(rate, 2, 1, pcm).event())
        try:
            await task
        except Exception:
            _LOGGER.exception("Synthesis failed: %s", text)


def _concat(chunks: list[array.array]) -> array.array:
    out = array.array("f")
    for chunk in chunks:
        out.extend(chunk)
    return out
