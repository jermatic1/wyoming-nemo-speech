"""cffi bindings for libnemo_speech_tts."""

# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cffi import FFI

TTS_CDEF = """
typedef ... nemo_speech_tts_synthesizer;

typedef enum nemo_speech_tts_status {
    NEMO_SPEECH_TTS_OK = 0,
    NEMO_SPEECH_TTS_ERROR_INVALID_ARGUMENT = 1,
    NEMO_SPEECH_TTS_ERROR_OUT_OF_MEMORY = 2,
    NEMO_SPEECH_TTS_ERROR_RUNTIME = 3,
    NEMO_SPEECH_TTS_ERROR_CANCELLED = 4
} nemo_speech_tts_status;

typedef enum nemo_speech_tts_backend_preference {
    NEMO_SPEECH_TTS_BACKEND_AUTO = 0,
    NEMO_SPEECH_TTS_BACKEND_CPU = 1,
    NEMO_SPEECH_TTS_BACKEND_CUDA = 2
} nemo_speech_tts_backend_preference;

typedef enum nemo_speech_tts_uma_mode {
    NEMO_SPEECH_TTS_UMA_AUTO = 0,
    NEMO_SPEECH_TTS_UMA_OFF = 1,
    NEMO_SPEECH_TTS_UMA_ON = 2
} nemo_speech_tts_uma_mode;

typedef enum nemo_speech_tts_longform_mode {
    NEMO_SPEECH_TTS_LONGFORM_AUTO = 0,
    NEMO_SPEECH_TTS_LONGFORM_OFF = 1,
    NEMO_SPEECH_TTS_LONGFORM_ON = 2
} nemo_speech_tts_longform_mode;

typedef struct nemo_speech_tts_model_config {
    size_t size;
    const char* magpie_model;
    const char* codec_model;
    const char* tokenizer_model_dir;
    const char* text_normalizer_model_dir;
} nemo_speech_tts_model_config;

typedef struct nemo_speech_tts_runtime_config {
    size_t size;
    int32_t speaker;
    int32_t threads;
    int32_t codec_threads;
    int32_t seed;
    int32_t steps;
    int32_t top_k;
    int32_t chunk_frames;
    int32_t codec_queue_depth;
    int32_t codec_history_frames;
    int32_t codec_future_frames;
    int32_t window_ms;
    float temperature;
    _Bool override_temperature;
    float cfg_scale;
    _Bool override_cfg_scale;
    _Bool use_cfg;
    _Bool use_local_transformer;
    _Bool use_kv_cache;
    _Bool use_stateful_codec;
    _Bool codec_cpu;
    _Bool flush_partial_chunk;
    _Bool verbose;
    nemo_speech_tts_backend_preference lt_backend;
    nemo_speech_tts_backend_preference sampling_backend;
    nemo_speech_tts_uma_mode uma_mode;
    nemo_speech_tts_longform_mode longform_mode;
    _Bool lt_fp32;
} nemo_speech_tts_runtime_config;

typedef struct nemo_speech_tts_synthesizer_config {
    size_t size;
    const nemo_speech_tts_model_config* model;
    const nemo_speech_tts_runtime_config* runtime;
    const char* default_language_code;
    const char* default_voice_name;
} nemo_speech_tts_synthesizer_config;

typedef struct nemo_speech_tts_synthesis_options {
    size_t size;
    const char* request_id;
    const char* language_code;
    int32_t speaker;
    int32_t seed;
    int32_t steps;
    int32_t top_k;
    float temperature;
    _Bool override_temperature;
    float cfg_scale;
    _Bool override_cfg_scale;
    const char* voice_name;
    int32_t output_sample_rate;
} nemo_speech_tts_synthesis_options;

typedef _Bool (*nemo_speech_tts_pcm_callback)(
    const uint8_t* pcm, size_t n_bytes, void* user_data);

nemo_speech_tts_runtime_config nemo_speech_tts_runtime_config_default(void);
nemo_speech_tts_synthesis_options nemo_speech_tts_synthesis_options_default(void);

nemo_speech_tts_status nemo_speech_tts_create(
    const nemo_speech_tts_synthesizer_config* cfg,
    nemo_speech_tts_synthesizer** out);
void nemo_speech_tts_destroy(nemo_speech_tts_synthesizer* synthesizer);

int32_t nemo_speech_tts_sample_rate(const nemo_speech_tts_synthesizer* synthesizer);
int32_t nemo_speech_tts_speaker_count(const nemo_speech_tts_synthesizer* synthesizer);
const char* nemo_speech_tts_speaker_name(
    const nemo_speech_tts_synthesizer* synthesizer, size_t i);

nemo_speech_tts_status nemo_speech_tts_synthesize_text(
    nemo_speech_tts_synthesizer* synthesizer,
    const nemo_speech_tts_synthesis_options* options,
    const char* text,
    nemo_speech_tts_pcm_callback callback,
    void* user_data,
    void* stats_out);

const char* nemo_speech_tts_last_error(void);
const char* nemo_speech_tts_version(void);
"""

ffi = FFI()
ffi.cdef(TTS_CDEF)

PcmCallback = Callable[[bytes], bool]


class TtsError(RuntimeError):
    pass


def library_path(prefix: Path) -> Path | None:
    path = Path(prefix) / "lib" / "libnemo_speech_tts.so"
    return path if path.is_file() else None


class TtsLibrary:
    def __init__(self, path: Path) -> None:
        self.lib: Any = ffi.dlopen(str(path))

    def version(self) -> str:
        return ffi.string(self.lib.nemo_speech_tts_version()).decode()

    def create(
        self,
        magpie: Path,
        codec: Path,
        tokenizer: Path,
        gpu: int,
    ) -> Synthesizer:
        return Synthesizer(self, magpie, codec, tokenizer, gpu)


def speaker_map(tokenizer: Path) -> dict[str, int] | None:
    """Read the model's own speaker map from the tokenizer directory.

    The GGUF's speaker name list can disagree with its baked speaker rows (the
    v2607 package does), so voices are mapped through this file when it exists.
    """
    files = sorted(tokenizer.glob("*speakers.json"))
    if not files:
        return None
    data = json.loads(files[0].read_text())
    return {str(name): int(index) for name, index in data.items()}


class Synthesizer:
    def __init__(
        self,
        library: TtsLibrary,
        magpie: Path,
        codec: Path,
        tokenizer: Path,
        gpu: int,
    ) -> None:
        self._lib = library.lib
        self._lock = threading.Lock()
        model = ffi.new("nemo_speech_tts_model_config*")
        model.size = ffi.sizeof("nemo_speech_tts_model_config")
        magpie_path = ffi.new("char[]", str(magpie).encode())
        codec_path = ffi.new("char[]", str(codec).encode())
        tokenizer_path = ffi.new("char[]", str(tokenizer).encode())
        model.magpie_model = magpie_path
        model.codec_model = codec_path
        model.tokenizer_model_dir = tokenizer_path
        runtime = ffi.new("nemo_speech_tts_runtime_config*")
        runtime[0] = self._lib.nemo_speech_tts_runtime_config_default()
        # Longform (sentence-chunk) mode can abort on a history-cache size
        # mismatch in NeMo-Speech.cpp. The server sends one sentence per call.
        runtime.longform_mode = self._lib.NEMO_SPEECH_TTS_LONGFORM_OFF
        if gpu < 0:
            runtime.lt_backend = self._lib.NEMO_SPEECH_TTS_BACKEND_CPU
            runtime.sampling_backend = self._lib.NEMO_SPEECH_TTS_BACKEND_CPU
            runtime.codec_cpu = True
        language = ffi.new("char[]", b"en-US")
        config = ffi.new("nemo_speech_tts_synthesizer_config*")
        config.size = ffi.sizeof("nemo_speech_tts_synthesizer_config")
        config.model = model
        config.runtime = runtime
        config.default_language_code = language
        out = ffi.new("nemo_speech_tts_synthesizer**")
        status = self._lib.nemo_speech_tts_create(config, out)
        if status != self._lib.NEMO_SPEECH_TTS_OK:
            raise TtsError(self._last_error())
        self._handle = out[0]
        self._keepalive = [
            model,
            magpie_path,
            codec_path,
            tokenizer_path,
            runtime,
            language,
            config,
        ]
        try:
            self.sample_rate = int(self._lib.nemo_speech_tts_sample_rate(self._handle))
            count = self._lib.nemo_speech_tts_speaker_count(self._handle)
            self.speakers = [self._speaker_name(index) for index in range(count)]
            self._speaker_index = speaker_map(tokenizer) or {}
            if any(index >= count for index in self._speaker_index.values()):
                self._speaker_index = {}
            if self._speaker_index:
                by_index = sorted(self._speaker_index.items(), key=lambda item: item[1])
                self.speakers = [name for name, _ in by_index]
        except Exception:
            self.close()
            raise

    def _speaker_name(self, index: int) -> str:
        name = self._lib.nemo_speech_tts_speaker_name(self._handle, index)
        if name == ffi.NULL:
            raise TtsError(f"speaker {index} has no name")
        return ffi.string(name).decode()

    def synthesize(
        self,
        text: str,
        voice: str | None,
        language: str,
        on_pcm: PcmCallback,
    ) -> None:
        options = ffi.new("nemo_speech_tts_synthesis_options*")
        options[0] = self._lib.nemo_speech_tts_synthesis_options_default()
        options.speaker = -1
        encoded = ffi.new("char[]", language.encode())
        options.language_code = encoded
        keepalive: list[object] = [options, encoded]
        errors: list[BaseException] = []

        @ffi.callback("nemo_speech_tts_pcm_callback")
        def callback(pcm, n_bytes, _user_data):
            try:
                return on_pcm(bytes(ffi.buffer(pcm, n_bytes)))
            except BaseException as exc:
                errors.append(exc)
                return False

        keepalive.append(callback)
        if voice and voice in self._speaker_index:
            options.speaker = self._speaker_index[voice]
        elif voice:
            encoded_voice = ffi.new("char[]", voice.encode())
            keepalive.append(encoded_voice)
            options.voice_name = encoded_voice
        with self._lock:
            status = self._lib.nemo_speech_tts_synthesize_text(
                self._handle,
                options,
                text.encode(),
                callback,
                ffi.NULL,
                ffi.NULL,
            )
        if errors:
            raise errors[0]
        if status != self._lib.NEMO_SPEECH_TTS_OK:
            raise TtsError(self._last_error())

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle != ffi.NULL:
            self._lib.nemo_speech_tts_destroy(handle)
            self._handle = ffi.NULL

    def _last_error(self) -> str:
        message = self._lib.nemo_speech_tts_last_error()
        if message == ffi.NULL:
            return "TTS call failed"
        return ffi.string(message).decode()


def load(prefix: Path) -> TtsLibrary:
    path = library_path(prefix)
    if path is None:
        raise TtsError(f"libnemo_speech_tts.so not found under {prefix}")
    return TtsLibrary(path)
