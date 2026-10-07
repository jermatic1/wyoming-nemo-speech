"""Find model files in the NeMo-Speech.cpp cache, pulling them if missing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import urllib.request
from collections.abc import Callable
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

# Speaker models are plain ONNX files from the sherpa-onnx release.
SPEAKER_RELEASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
)
SPEAKER_MODELS = {
    "titanet-large": (
        "nemo_en_titanet_large.onnx",
        "d51abcf31717ef28162f26acb9d44dd4127c3d44c9b8624f699f3425daca8e77",
    ),
}


class MissingModel(FileNotFoundError):
    pass


def asr_model(name: str, lib_dir: Path) -> Path:
    return _resolve_or_pull(name, lib_dir, resolve_asr_model)


def tts_models(
    name: str,
    lib_dir: Path,
    *,
    model_path: Path | None = None,
    tokenizer_dir: Path | None = None,
    codec_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """The Magpie GGUF, codec GGUF and tokenizer directory.

    With model_path and tokenizer_dir set, those are used as given and only the
    codec, which fine-tunes share with the stock model, comes from the index.
    """
    if model_path is None or tokenizer_dir is None:
        return _resolve_or_pull(name, lib_dir, resolve_tts_models)
    for path, kind in (
        (model_path, "tts.model_path"),
        (tokenizer_dir, "tts.tokenizer_dir"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{kind} does not exist: {path}")
    if codec_path is None:
        codec_path = _resolve_or_pull(name, lib_dir, resolve_tts_codec)
    elif not codec_path.is_file():
        raise FileNotFoundError(f"tts.codec_path does not exist: {codec_path}")
    return model_path, codec_path, tokenizer_dir


def speaker_model(name: str) -> Path:
    """An ONNX file by path or alias, downloaded into the cache if missing."""
    path = Path(name).expanduser()
    if path.is_file():
        return path
    key = name.strip().lower()
    if key not in SPEAKER_MODELS:
        raise FileNotFoundError(f"unknown speaker model: {name}")
    filename, sha256 = SPEAKER_MODELS[key]
    target = (
        _cache_root() / "k2-fsa" / "sherpa-onnx" / "speaker-recongition-models"
    ) / filename
    if not target.is_file():
        _download(SPEAKER_RELEASE + filename, target, sha256)
    return target


def _download(url: str, path: Path, sha256: str) -> None:
    _LOGGER.info("Downloading %s", url)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, part.open("wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        part.unlink()
        raise ValueError(f"checksum mismatch for {url}")
    part.replace(path)
    _LOGGER.info("Saved %s", path)


def pull(name: str, lib_dir: Path) -> None:
    _LOGGER.info("Pulling model %s", name)
    subprocess.run([str(lib_dir / "bin" / "nemo-speech"), "pull", name], check=True)


def resolve_asr_model(name: str, lib_dir: Path) -> Path:
    path = Path(name)
    if path.is_file():
        return path
    model = _find_model(_load_index(lib_dir), name)
    return _checked(_artifact_path(model, "asr"), name)


def resolve_tts_models(name: str, lib_dir: Path) -> tuple[Path, Path, Path]:
    index = _load_index(lib_dir)
    model = _find_model(index, name)
    codec = _find_model(index, model["companions"][0])
    return (
        _checked(_artifact_path(model, "tts"), name),
        _checked(_artifact_path(codec, "codec"), name),
        _checked(_artifact_path(model, "tokenizer"), name),
    )


def resolve_tts_codec(name: str, lib_dir: Path) -> Path:
    index = _load_index(lib_dir)
    codec = _find_model(index, _find_model(index, name)["companions"][0])
    return _checked(_artifact_path(codec, "codec"), codec["repo"])


def _resolve_or_pull[T](
    name: str, lib_dir: Path, resolve: Callable[[str, Path], T]
) -> T:
    try:
        return resolve(name, lib_dir)
    except MissingModel:
        pull(name, lib_dir)
        return resolve(name, lib_dir)


def _checked(path: Path, name: str) -> Path:
    if not path.exists():
        raise MissingModel(f"{path} is missing. Run: nemo-speech pull {name}")
    return path


def _load_index(lib_dir: Path) -> dict:
    path = Path(lib_dir) / "share" / "nemo-speech" / "model-index.json"
    if not path.is_file():
        raise FileNotFoundError(f"model index not found: {path}")
    return json.loads(path.read_text())


def _find_model(index: dict, name: str) -> dict:
    key = name.strip().lower()
    for model in index["models"]:
        names = [model["repo"], *model.get("aliases", [])]
        if key in {item.lower() for item in names}:
            return model
    raise FileNotFoundError(f"unknown model: {name}")


def _artifact_path(model: dict, role: str) -> Path:
    for artifact in model["artifacts"]:
        if artifact["role"] == role:
            break
    else:
        raise FileNotFoundError(f"{model['repo']} has no {role} artifact")
    owner, repo = model["repo"].split("/", 1)
    directory = _cache_root() / owner / repo / model["revision"]
    if artifact["type"] == "file":
        return directory / artifact["filename"]
    return directory / artifact["directory"]


def _cache_root() -> Path:
    override = os.environ.get("NEMO_SPEECH_MODEL_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "nemo-speech" / "models"
    return Path.home() / ".cache" / "nemo-speech" / "models"
