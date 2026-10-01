"""Settings from an optional config.toml in the working directory."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("config.toml")
LIB_DIRS = (Path("/opt/nemo-speech"), Path.home() / ".local" / "share" / "nemo-speech")


def default_lib_dir() -> Path:
    for path in LIB_DIRS:
        if path.is_dir():
            return path
    return LIB_DIRS[-1]


@dataclass(frozen=True)
class Asr:
    model: str = "nemotron-en"
    speech_context_boost: float = 2.5


@dataclass(frozen=True)
class Tts:
    model: str = "magpie"


@dataclass(frozen=True)
class HomeAssistant:
    url: str = "http://homeassistant.local:8123"
    token: str = ""


@dataclass(frozen=True)
class Config:
    lib_dir: Path = field(default_factory=default_lib_dir)
    device: str = "cpu"
    uri: str = "tcp://0.0.0.0:10400"
    debug: bool = False
    asr: Asr = field(default_factory=Asr)
    tts: Tts = field(default_factory=Tts)
    home_assistant: HomeAssistant = field(default_factory=HomeAssistant)


SECTIONS = {"asr": Asr, "tts": Tts, "home_assistant": HomeAssistant}


def load(path: Path = CONFIG_PATH) -> Config:
    if not path.is_file():
        return Config()
    data = tomllib.loads(path.read_text())
    for name, section in SECTIONS.items():
        if name in data:
            data[name] = _build(section, data[name], f"{name}.")
    if "lib_dir" in data:
        data["lib_dir"] = Path(data["lib_dir"]).expanduser()
    return _build(Config, data, "")


def _build(cls: type, data: dict[str, Any], prefix: str) -> Any:
    known = {item.name for item in fields(cls)}
    for key in data:
        if key not in known:
            raise ValueError(f"unknown config key: {prefix}{key}")
    return cls(**data)
