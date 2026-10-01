"""Find model files in the NeMo-Speech.cpp cache, pulling them if missing."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


class MissingModel(FileNotFoundError):
    pass


def asr_model(name: str, lib_dir: Path) -> Path:
    return _resolve_or_pull(name, lib_dir, resolve_asr_model)


def tts_models(name: str, lib_dir: Path) -> tuple[Path, Path, Path]:
    return _resolve_or_pull(name, lib_dir, resolve_tts_models)


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
