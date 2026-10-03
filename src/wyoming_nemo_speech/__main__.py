"""Load the config and models, then serve ASR and TTS on one socket."""

from __future__ import annotations

import asyncio
import logging
import os
from functools import partial

from wyoming.info import Info
from wyoming.server import AsyncServer

from wyoming_nemo_speech.asr import asr_program
from wyoming_nemo_speech.config import load as load_config
from wyoming_nemo_speech.libasr import gpu_index
from wyoming_nemo_speech.libasr import load as load_asr
from wyoming_nemo_speech.libtts import load as load_tts
from wyoming_nemo_speech.models import asr_model, tts_models
from wyoming_nemo_speech.names import HassNameCache, HomeAssistant
from wyoming_nemo_speech.server import SpeechEventHandler
from wyoming_nemo_speech.tts import tts_program

_LOGGER = logging.getLogger(__name__)


def run() -> None:
    asyncio.run(main())


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
    )
    try:
        await server.run(handler)
    finally:
        if names is not None:
            await names.close()
        recognizer.close()
        synthesizer.close()
