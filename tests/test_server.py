import array

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeVoice,
)

from wyoming_nemo_speech.asr import asr_program
from wyoming_nemo_speech.names import HassNameCache, NameList
from wyoming_nemo_speech.server import SpeechEventHandler
from wyoming_nemo_speech.tts import tts_program


class FakeRecognizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.error: Exception | None = None

    def recognize(self, samples, sample_rate, language, phrases, boost) -> str:
        self.calls.append(
            {
                "samples": list(samples),
                "sample_rate": sample_rate,
                "language": language,
                "phrases": phrases,
                "boost": boost,
            }
        )
        if self.error is not None:
            raise self.error
        return "office lights"


class FakeSynthesizer:
    def __init__(self) -> None:
        self.sample_rate = 22050
        self.speakers = ["John", "Sofia"]
        self.calls: list[tuple] = []
        self.error: Exception | None = None

    def synthesize(self, text, voice, language, on_pcm) -> None:
        self.calls.append((text, voice, language))
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        on_pcm(b"\x01\x00\x02\x00")
        on_pcm(b"\x03\x00")


class FakeHass:
    async def get_names(self) -> NameList:
        return NameList(used_areas=["Office"], priority_entities=["Floor Lamp"])


class RecordingHandler(SpeechEventHandler):
    def __init__(
        self,
        recognizer: FakeRecognizer,
        synthesizer: FakeSynthesizer,
        names: HassNameCache | None = None,
    ) -> None:
        info = Info(
            asr=[asr_program("nemotron-en")],
            tts=[tts_program(synthesizer.speakers)],
        )
        super().__init__(recognizer, synthesizer, info, 2.5, names, None, None)
        self.sent: list[Event] = []

    async def write_event(self, event: Event) -> None:
        self.sent.append(event)


def pcm16(*samples: int) -> bytes:
    return array.array("h", samples).tobytes()


async def utterance(handler: RecordingHandler, audio: bytes, rate: int = 16000):
    assert await handler.handle_event(AudioStart(rate, 2, 1).event())
    assert await handler.handle_event(AudioChunk(rate, 2, 1, audio).event())
    assert await handler.handle_event(AudioStop().event()) is False


async def test_describe_advertises_both_programs() -> None:
    handler = RecordingHandler(FakeRecognizer(), FakeSynthesizer())
    assert await handler.handle_event(Describe().event())
    info = Info.from_event(handler.sent[0])
    assert info.asr[0].name == "nemotron"
    assert info.asr[0].requires_external_vad is True
    assert info.asr[0].models[0].languages == ["en"]
    assert [voice.name for voice in info.tts[0].voices] == ["John", "Sofia"]


async def test_utterance_is_scaled_and_transcribed() -> None:
    recognizer = FakeRecognizer()
    handler = RecordingHandler(recognizer, FakeSynthesizer())
    assert await handler.handle_event(Transcribe(language="es").event())
    await utterance(handler, pcm16(16384, -32768), rate=22050)
    call = recognizer.calls[0]
    assert call["sample_rate"] == 22050
    assert call["samples"] == pytest.approx([16384 / 32768, -1.0])
    assert call["language"] == "en-US"
    assert call["phrases"] is None
    assert call["boost"] == 2.5
    transcript = Transcript.from_event(handler.sent[-1])
    assert transcript.text == "office lights"
    assert transcript.language == "en-US"


async def test_chunks_are_concatenated_not_resampled() -> None:
    recognizer = FakeRecognizer()
    handler = RecordingHandler(recognizer, FakeSynthesizer())
    assert await handler.handle_event(AudioStart(8000, 2, 1).event())
    assert await handler.handle_event(AudioChunk(8000, 2, 1, pcm16(1, 2)).event())
    assert await handler.handle_event(AudioChunk(8000, 2, 1, pcm16(3)).event())
    assert await handler.handle_event(AudioStop().event()) is False
    assert len(recognizer.calls[0]["samples"]) == 3
    assert recognizer.calls[0]["sample_rate"] == 8000


async def test_stop_without_audio_sends_empty_transcript() -> None:
    recognizer = FakeRecognizer()
    handler = RecordingHandler(recognizer, FakeSynthesizer())
    assert await handler.handle_event(AudioStop().event()) is False
    assert recognizer.calls == []
    assert Transcript.from_event(handler.sent[-1]).text == ""


async def test_recognition_failure_sends_empty_transcript() -> None:
    recognizer = FakeRecognizer()
    recognizer.error = RuntimeError("boom")
    handler = RecordingHandler(recognizer, FakeSynthesizer())
    await utterance(handler, pcm16(0))
    assert Transcript.from_event(handler.sent[-1]).text == ""


async def test_rejects_bad_audio() -> None:
    handler = RecordingHandler(FakeRecognizer(), FakeSynthesizer())
    with pytest.raises(ValueError):
        await handler.handle_event(AudioChunk(16000, 4, 1, b"\x00" * 8).event())
    await handler.handle_event(AudioChunk(16000, 2, 1, pcm16(0)).event())
    with pytest.raises(ValueError):
        await handler.handle_event(AudioChunk(8000, 2, 1, pcm16(0)).event())


async def test_names_are_passed_as_phrases() -> None:
    recognizer = FakeRecognizer()
    names = HassNameCache(FakeHass())
    await names.start()
    handler = RecordingHandler(recognizer, FakeSynthesizer(), names)
    await utterance(handler, pcm16(0))
    assert recognizer.calls[0]["phrases"] == ["Office", "Floor Lamp"]


async def test_synthesize_streams_pcm() -> None:
    synthesizer = FakeSynthesizer()
    handler = RecordingHandler(FakeRecognizer(), synthesizer)
    request = Synthesize("Hello", voice=SynthesizeVoice(name="john"))
    assert await handler.handle_event(request.event())
    assert synthesizer.calls == [("Hello", "John", "en-US")]
    start = AudioStart.from_event(handler.sent[0])
    assert (start.rate, start.width, start.channels) == (22050, 2, 1)
    chunks = [AudioChunk.from_event(event).audio for event in handler.sent[1:-1]]
    assert chunks == [b"\x01\x00\x02\x00", b"\x03\x00"]
    assert AudioStop.is_type(handler.sent[-1].type)


async def test_synthesize_spells_out_numbers() -> None:
    synthesizer = FakeSynthesizer()
    handler = RecordingHandler(FakeRecognizer(), synthesizer)
    assert await handler.handle_event(Synthesize("It is 27°.").event())
    assert synthesizer.calls[0][0] == "It is twenty-seven degrees."


async def test_unknown_voice_uses_default() -> None:
    synthesizer = FakeSynthesizer()
    handler = RecordingHandler(FakeRecognizer(), synthesizer)
    request = Synthesize("Hi", voice=SynthesizeVoice(name="nope", language="de"))
    assert await handler.handle_event(request.event())
    assert synthesizer.calls == [("Hi", None, "en-US")]


async def test_synthesis_failure_still_sends_start_and_stop() -> None:
    synthesizer = FakeSynthesizer()
    synthesizer.error = RuntimeError("boom")
    handler = RecordingHandler(FakeRecognizer(), synthesizer)
    assert await handler.handle_event(Synthesize("Hello").event())
    assert [event.type for event in handler.sent] == ["audio-start", "audio-stop"]


async def test_synthesize_speaks_each_sentence() -> None:
    synthesizer = FakeSynthesizer()
    handler = RecordingHandler(FakeRecognizer(), synthesizer)
    assert await handler.handle_event(Synthesize("It is on. Anything else?").event())
    assert [call[0] for call in synthesizer.calls] == ["It is on.", "Anything else?"]
    types = [event.type for event in handler.sent]
    assert types == ["audio-start"] + ["audio-chunk"] * 4 + ["audio-stop"]


async def test_failed_sentence_does_not_stop_the_rest() -> None:
    synthesizer = FakeSynthesizer()
    synthesizer.error = RuntimeError("boom")
    handler = RecordingHandler(FakeRecognizer(), synthesizer)
    assert await handler.handle_event(Synthesize("It is on. Anything else?").event())
    assert len(synthesizer.calls) == 2
    types = [event.type for event in handler.sent]
    assert types == ["audio-start"] + ["audio-chunk"] * 2 + ["audio-stop"]


async def test_streaming_synthesis_splits_sentences_across_chunks() -> None:
    synthesizer = FakeSynthesizer()
    handler = RecordingHandler(FakeRecognizer(), synthesizer)
    start = SynthesizeStart(voice=SynthesizeVoice(name="sofia"))
    assert await handler.handle_event(start.event())
    for text in ["The light is on. It is 2", "1 degrees out", "side. Anything else?"]:
        assert await handler.handle_event(SynthesizeChunk(text).event())
    assert await handler.handle_event(SynthesizeStop().event())
    assert synthesizer.calls == [
        ("The light is on.", "Sofia", "en-US"),
        ("It is twenty-one degrees outside.", "Sofia", "en-US"),
        ("Anything else?", "Sofia", "en-US"),
    ]
    types = [event.type for event in handler.sent]
    assert types[0] == "audio-start"
    assert types.count("audio-start") == 1
    assert types.count("audio-chunk") == 6
    assert types[-2:] == ["audio-stop", "synthesize-stopped"]


async def test_full_text_synthesize_inside_stream_is_ignored() -> None:
    synthesizer = FakeSynthesizer()
    handler = RecordingHandler(FakeRecognizer(), synthesizer)
    text = "The kitchen light is on."
    assert await handler.handle_event(SynthesizeStart().event())
    assert await handler.handle_event(SynthesizeChunk(text).event())
    assert await handler.handle_event(Synthesize(text).event())
    assert await handler.handle_event(SynthesizeStop().event())
    assert [call[0] for call in synthesizer.calls] == [text]
    types = [event.type for event in handler.sent]
    assert types == [
        "audio-start",
        "audio-chunk",
        "audio-chunk",
        "audio-stop",
        "synthesize-stopped",
    ]
    assert await handler.handle_event(Synthesize("Hi there again.").event())
    assert len(synthesizer.calls) == 2


async def test_stream_events_without_start_are_ignored() -> None:
    synthesizer = FakeSynthesizer()
    handler = RecordingHandler(FakeRecognizer(), synthesizer)
    assert await handler.handle_event(SynthesizeChunk("Hello.").event())
    assert await handler.handle_event(SynthesizeStop().event())
    assert synthesizer.calls == []
    assert handler.sent == []
