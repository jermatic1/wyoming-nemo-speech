"""Load the config and models, then serve ASR and TTS on one socket."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from functools import partial

from wyoming.info import Info
from wyoming.server import AsyncServer

from wyoming_nemo_speech.asr import asr_program
from wyoming_nemo_speech.config import Speakers
from wyoming_nemo_speech.config import load as load_config
from wyoming_nemo_speech.libasr import gpu_index
from wyoming_nemo_speech.libasr import load as load_asr
from wyoming_nemo_speech.libspeaker import TitaNet
from wyoming_nemo_speech.libtts import load as load_tts
from wyoming_nemo_speech.models import asr_model, speaker_model, tts_models
from wyoming_nemo_speech.names import HassNameCache, HomeAssistant
from wyoming_nemo_speech.server import SpeechEventHandler
from wyoming_nemo_speech.speakers import SpeakerIdentifier, enroll, report
from wyoming_nemo_speech.tts import tts_program

_LOGGER = logging.getLogger(__name__)


def run() -> None:
    parser = argparse.ArgumentParser(prog="wyoming-nemo-speech")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser(
        "speakers", help="Show how well the enrolled speaker recordings separate"
    )
    args = parser.parse_args()
    if args.command == "speakers":
        sys.exit(report_speakers())
    asyncio.run(main())


def report_speakers() -> int:
    config = load_config()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not config.speakers.dir.is_dir():
        print(f"No speakers directory: {config.speakers.dir}", file=sys.stderr)
        return 1
    encoder = TitaNet(speaker_model(config.speakers.model))
    lines = report(config.speakers.dir, encoder)
    if not lines:
        print(f"No speakers enrolled under {config.speakers.dir}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


def load_speakers(settings: Speakers) -> SpeakerIdentifier | None:
    if not settings.enabled or not settings.dir.is_dir():
        return None
    if not any(path.is_dir() for path in settings.dir.iterdir()):
        _LOGGER.info("No speakers enrolled under %s", settings.dir)
        return None
    encoder = TitaNet(speaker_model(settings.model))
    profiles = enroll(settings.dir, encoder)
    if not profiles:
        _LOGGER.warning("No speakers enrolled under %s", settings.dir)
        return None
    _LOGGER.info("Enrolled speakers: %s", ", ".join(profiles))
    return SpeakerIdentifier(
        encoder, profiles, settings.threshold, settings.min_seconds
    )


async def main() -> None:
    config = load_config()
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    gpu = gpu_index(config.device)
    if config.device.lower().startswith("vulkan") and gpu >= 0:
        # ggml renumbers the visible device to 0.
        os.environ["GGML_VK_VISIBLE_DEVICES"] = str(gpu)
        gpu = 0

    asr_path = asr_model(config.asr.model, config.lib_dir)
    magpie, codec, tokenizer = tts_models(config.tts.model, config.lib_dir)
    asr_library = load_asr(config.lib_dir)
    tts_library = load_tts(config.lib_dir)
    recognizer = asr_library.create(asr_path, gpu)
    synthesizer = tts_library.create(magpie, codec, tokenizer, gpu)
    info = Info(
        asr=[asr_program(config.asr.model, asr_library.version())],
        tts=[tts_program(synthesizer.speakers, tts_library.version())],
    )
    speakers = load_speakers(config.speakers)

    names = None
    if config.home_assistant.token:
        hass = HomeAssistant(config.home_assistant.token, config.home_assistant.url)
        names = HassNameCache(hass)
        await names.start()

    _LOGGER.info("Serving %s and %s on %s", asr_path.name, magpie.name, config.uri)
    server = AsyncServer.from_uri(config.uri)
    handler = partial(
        SpeechEventHandler,
        recognizer,
        synthesizer,
        info,
        config.asr.speech_context_boost,
        names,
        speakers,
        config.speakers.prefix,
    )
    try:
        await server.run(handler)
    finally:
        if names is not None:
            await names.close()
        recognizer.close()
        synthesizer.close()
