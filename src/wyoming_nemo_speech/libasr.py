"""cffi bindings for libnemo_speech_asr_c."""

# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import array
import threading
from pathlib import Path
from typing import Any

from cffi import FFI

ASR_CDEF = """
typedef ... nemo_speech_asr_recognizer;
typedef ... nemo_speech_asr_result;

typedef enum nemo_speech_asr_status {
    NEMO_SPEECH_ASR_OK = 0,
    NEMO_SPEECH_ASR_ERROR_INVALID_ARGUMENT = 1,
    NEMO_SPEECH_ASR_ERROR_OUT_OF_MEMORY = 2,
    NEMO_SPEECH_ASR_ERROR_RUNTIME = 3,
    NEMO_SPEECH_ASR_ERROR_CANCELLED = 4
} nemo_speech_asr_status;

typedef struct nemo_speech_asr_backend_config {
    size_t size;
    int32_t gpu;
} nemo_speech_asr_backend_config;

typedef struct nemo_speech_asr_model_config {
    size_t size;
    const char* path;
    const char* name;
} nemo_speech_asr_model_config;

typedef struct nemo_speech_asr_recognizer_config {
    size_t size;
    const nemo_speech_asr_backend_config* backend;
    const nemo_speech_asr_model_config* model;
    const void* streaming;
    const void* decoder;
    const void* vad;
    const void* endpointing;
    const void* postproc;
    const void* diar;
    const void* batching;
} nemo_speech_asr_recognizer_config;

typedef struct nemo_speech_asr_speech_context {
    size_t size;
    const char* const* phrases;
    size_t phrase_count;
    float boost;
} nemo_speech_asr_speech_context;

typedef struct nemo_speech_asr_recognition_options {
    size_t size;
    const char* request_id;
    const char* language_code;
    _Bool interim_results;
    _Bool enable_word_time_offsets;
    _Bool enable_automatic_punctuation;
    _Bool verbatim_transcripts;
    _Bool profanity_filter;
    int32_t stop_history_eou_ms;
    const nemo_speech_asr_speech_context* speech_contexts;
    size_t speech_context_count;
    int32_t max_alternatives;
    _Bool enable_speaker_diarization;
    int32_t max_speaker_count;
} nemo_speech_asr_recognition_options;

nemo_speech_asr_recognition_options nemo_speech_asr_recognition_options_default(void);

nemo_speech_asr_status nemo_speech_asr_create(
    const nemo_speech_asr_recognizer_config* cfg,
    nemo_speech_asr_recognizer** out);
void nemo_speech_asr_destroy(nemo_speech_asr_recognizer* recognizer);

nemo_speech_asr_status nemo_speech_asr_recognize_f32(
    nemo_speech_asr_recognizer* recognizer,
    const nemo_speech_asr_recognition_options* options,
    const float* samples,
    size_t n_samples,
    int32_t sample_rate,
    nemo_speech_asr_result** out);

size_t nemo_speech_asr_result_alternative_count(const nemo_speech_asr_result* result);
const char* nemo_speech_asr_result_transcript(
    const nemo_speech_asr_result* result, size_t alt);
void nemo_speech_asr_result_destroy(nemo_speech_asr_result* result);

const char* nemo_speech_asr_last_error(void);
const char* nemo_speech_asr_version(void);
"""

ffi = FFI()
ffi.cdef(ASR_CDEF)


class AsrError(RuntimeError):
    pass


def gpu_index(device: str) -> int:
    """cpu is -1. vulkan:N and cuda:N are the device index."""
    text = device.strip().lower()
    if text == "cpu":
        return -1
    if ":" not in text:
        if text in {"vulkan", "cuda", "metal"}:
            return 0
        raise ValueError(f"unsupported device: {device}")
    backend, index = text.split(":", 1)
    if backend not in {"vulkan", "cuda", "metal"}:
        raise ValueError(f"unsupported device: {device}")
    try:
        parsed = int(index)
    except ValueError as exc:
        raise ValueError(f"unsupported device: {device}") from exc
    if parsed < 0:
        raise ValueError(f"unsupported device: {device}")
    return parsed


def library_path(prefix: Path) -> Path | None:
    path = Path(prefix) / "lib" / "libnemo_speech_asr_c.so"
    return path if path.is_file() else None


def pcm16_to_f32(audio: bytes, channels: int) -> array.array:
    """Scale by 32768. The library resamples. This does not."""
    if channels < 1:
        raise ValueError(f"unsupported channel count: {channels}")
    frame_bytes = 2 * channels
    if len(audio) % frame_bytes != 0:
        raise ValueError("audio length is not a whole number of frames")
    samples = array.array("h")
    samples.frombytes(audio)
    if channels == 1:
        return array.array("f", (sample / 32768.0 for sample in samples))
    frames = len(samples) // channels
    mixed = array.array("f")
    for frame in range(frames):
        start = frame * channels
        total = sum(samples[start : start + channels])
        mixed.append(total / channels / 32768.0)
    return mixed


class AsrLibrary:
    def __init__(self, path: Path) -> None:
        self.lib: Any = ffi.dlopen(str(path))

    def version(self) -> str:
        return ffi.string(self.lib.nemo_speech_asr_version()).decode()

    def create(self, model_path: Path, gpu: int) -> Recognizer:
        return Recognizer(self, model_path, gpu)


class Recognizer:
    def __init__(self, library: AsrLibrary, model_path: Path, gpu: int) -> None:
        self._lib = library.lib
        self._lock = threading.Lock()
        backend = ffi.new("nemo_speech_asr_backend_config*")
        backend.size = ffi.sizeof("nemo_speech_asr_backend_config")
        backend.gpu = gpu
        model = ffi.new("nemo_speech_asr_model_config*")
        model.size = ffi.sizeof("nemo_speech_asr_model_config")
        path = ffi.new("char[]", str(model_path).encode())
        model.path = path
        config = ffi.new("nemo_speech_asr_recognizer_config*")
        config.size = ffi.sizeof("nemo_speech_asr_recognizer_config")
        config.backend = backend
        config.model = model
        out = ffi.new("nemo_speech_asr_recognizer**")
        status = self._lib.nemo_speech_asr_create(config, out)
        if status != self._lib.NEMO_SPEECH_ASR_OK:
            raise AsrError(self._last_error())
        self._handle = out[0]
        self._keepalive = [backend, model, path, config]

    def recognize(
        self,
        samples: array.array,
        sample_rate: int,
        language: str,
        phrases: list[str] | None,
        boost: float,
    ) -> str:
        if not samples:
            return ""
        options = ffi.new("nemo_speech_asr_recognition_options*")
        options[0] = self._lib.nemo_speech_asr_recognition_options_default()
        keepalive: list[object] = [options]
        encoded = ffi.new("char[]", language.encode())
        keepalive.append(encoded)
        options.language_code = encoded
        if phrases:
            context = ffi.new("nemo_speech_asr_speech_context*")
            context.size = ffi.sizeof("nemo_speech_asr_speech_context")
            encoded_phrases = [ffi.new("char[]", phrase.encode()) for phrase in phrases]
            phrase_array = ffi.new("char*[]", encoded_phrases)
            context.phrases = phrase_array
            context.phrase_count = len(encoded_phrases)
            context.boost = boost
            keepalive.extend([context, phrase_array, *encoded_phrases])
            options.speech_contexts = context
            options.speech_context_count = 1
        out = ffi.new("nemo_speech_asr_result**")
        c_samples = ffi.cast("float*", ffi.from_buffer(samples))
        with self._lock:
            status = self._lib.nemo_speech_asr_recognize_f32(
                self._handle,
                options,
                c_samples,
                len(samples),
                sample_rate,
                out,
            )
        if status != self._lib.NEMO_SPEECH_ASR_OK:
            raise AsrError(self._last_error())
        result = out[0]
        if result == ffi.NULL:
            return ""
        try:
            if self._lib.nemo_speech_asr_result_alternative_count(result) == 0:
                return ""
            text = self._lib.nemo_speech_asr_result_transcript(result, 0)
            if text == ffi.NULL:
                return ""
            return ffi.string(text).decode()
        finally:
            self._lib.nemo_speech_asr_result_destroy(result)

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle != ffi.NULL:
            self._lib.nemo_speech_asr_destroy(handle)
            self._handle = ffi.NULL

    def _last_error(self) -> str:
        message = self._lib.nemo_speech_asr_last_error()
        if message == ffi.NULL:
            return "ASR call failed"
        return ffi.string(message).decode()


def load(prefix: Path) -> AsrLibrary:
    path = library_path(prefix)
    if path is None:
        raise AsrError(f"libnemo_speech_asr_c.so not found under {prefix}")
    return AsrLibrary(path)
