"""Settings from an optional config.toml in the working directory."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_args

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
    # Model files to use instead of looking `model` up in the index: the GGUF
    # and the extracted checkpoint directory that holds its tokenizer assets
    # and speaker map. The codec comes from the index unless given.
    model_path: Path | None = None
    tokenizer_dir: Path | None = None
    codec_path: Path | None = None

    def __post_init__(self) -> None:
        if (self.model_path is None) != (self.tokenizer_dir is None):
            raise ValueError(
                "tts.model_path and tts.tokenizer_dir must be set together"
            )


@dataclass(frozen=True)
class HomeAssistant:
    url: str = "http://homeassistant.local:8123"
    token: str = ""


@dataclass(frozen=True)
class Speakers:
    enabled: bool = True
    dir: Path = Path("speakers")
    model: str = "titanet-large"
    threshold: float = 0.6
    min_seconds: float = 0.5
    prefix: str = "I'm {name}. {text}"


@dataclass(frozen=True)
class Config:
    lib_dir: Path = field(default_factory=default_lib_dir)
    device: str = "cpu"
    uri: str = "tcp://0.0.0.0:10400"
    debug: bool = False
    asr: Asr = field(default_factory=Asr)
    tts: Tts = field(default_factory=Tts)
    home_assistant: HomeAssistant = field(default_factory=HomeAssistant)
    speakers: Speakers = field(default_factory=Speakers)


SECTIONS = {
    "asr": Asr,
    "tts": Tts,
    "home_assistant": HomeAssistant,
    "speakers": Speakers,
}


def load(path: Path = CONFIG_PATH) -> Config:
    if not path.is_file():
        return Config()
    data = tomllib.loads(path.read_text())
    for name, section in SECTIONS.items():
        if name in data:
            data[name] = _build(section, data[name], f"{name}.")
    return _build(Config, data, "")


def _build(cls: type, data: dict[str, Any], prefix: str) -> Any:
    known = {item.name: item.type for item in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key: {prefix}{key}")
        if _is_path(known[key]) and isinstance(value, str):
            data[key] = Path(value).expanduser()
    return cls(**data)


def _is_path(annotation: Any) -> bool:
    return annotation is Path or Path in get_args(annotation)
