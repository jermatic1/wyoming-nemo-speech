"""Home Assistant names, sent as a speech context to bias recognition."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse

import aiohttp

_LOGGER = logging.getLogger(__name__)

HASS_URL = "http://homeassistant.local:8123"
REFRESH_AFTER = 24 * 60 * 60

# Domains a speaker names out loud. Sensors fall to the last tier.
PRIORITY_DOMAINS = frozenset(
    {
        "light",
        "switch",
        "fan",
        "media_player",
        "climate",
        "scene",
        "todo",
    }
)

Command = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class HomeAssistantError(Exception):
    pass


def clean_names(*groups: Iterable[Any]) -> list[str]:
    """Strip, collapse whitespace, and de-dupe. First spelling wins."""
    names: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group or ():
            if not isinstance(value, str):
                continue
            name = " ".join(value.split())
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
    return names


@dataclass
class _Entity:
    entity_id: str
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    area_id: str | None = None

    @property
    def domain(self) -> str:
        return self.entity_id.split(".", 1)[0]

    @property
    def is_priority(self) -> bool:
        return self.domain in PRIORITY_DOMAINS


@dataclass
class NameList:
    """Names in the order a speech context should list them."""

    used_areas: list[str] = field(default_factory=list)
    priority_entities: list[str] = field(default_factory=list)
    empty_areas: list[str] = field(default_factory=list)
    other_entities: list[str] = field(default_factory=list)

    def phrases(self) -> list[str]:
        return clean_names(
            self.used_areas,
            self.priority_entities,
            self.empty_areas,
            self.other_entities,
        )


class NameSource(Protocol):
    async def get_names(self) -> NameList: ...


class HomeAssistant:
    """Read-only client for names a speaker can say."""

    def __init__(self, token: str, url: str = HASS_URL, timeout: float = 10.0) -> None:
        self.token = token
        self.timeout = timeout
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        scheme = "wss" if parsed.scheme == "https" else "ws"
        self.websocket_api_url = urlunparse(
            parsed._replace(
                scheme=scheme,
                path=f"{parsed.path.rstrip('/')}/api/websocket",
                params="",
                query="",
                fragment="",
            )
        )

    async def get_names(self) -> NameList:
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        client_timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.ws_connect(
                    self.websocket_api_url, max_msg_size=0
                ) as websocket:

                    async def command(payload: dict[str, Any]) -> dict[str, Any]:
                        await websocket.send_json({"id": next_id(), **payload})
                        msg = await websocket.receive_json()
                        if not msg.get("success"):
                            raise HomeAssistantError(f"{payload['type']} failed: {msg}")
                        return msg

                    await self._authenticate(websocket)
                    return await load_names(command)
        except HomeAssistantError:
            raise
        except Exception as exc:
            raise HomeAssistantError(f"Failed to load names: {exc}") from exc

    async def _authenticate(self, websocket: Any) -> None:
        msg = await websocket.receive_json()
        if msg.get("type") != "auth_required":
            raise HomeAssistantError(f"Expected auth_required, got {msg}")
        await websocket.send_json({"type": "auth", "access_token": self.token})
        msg = await websocket.receive_json()
        if msg.get("type") != "auth_ok":
            raise HomeAssistantError(f"Authentication failed: {msg}")


async def load_names(command: Command) -> NameList:
    exposed = await _exposed(command)
    friendly = await _friendly_names(command, exposed)
    entries = await _registry_entries(command, exposed)
    device_areas = await _device_areas(command, entries)
    entities = _entities(exposed, friendly, entries, device_areas)
    return await _name_list(command, entities)


async def _exposed(command: Command) -> set[str]:
    msg = await command({"type": "homeassistant/expose_entity/list"})
    return {
        entity_id
        for entity_id, info in (msg["result"]["exposed_entities"] or {}).items()
        if info.get("conversation")
    }


async def _friendly_names(command: Command, exposed: set[str]) -> dict[str, str]:
    msg = await command({"type": "get_states"})
    friendly: dict[str, str] = {}
    for state in msg["result"]:
        entity_id = state["entity_id"]
        if entity_id not in exposed:
            continue
        attributes = state.get("attributes") or {}
        name = " ".join((attributes.get("friendly_name") or "").split())
        if name:
            friendly[entity_id] = name
    return friendly


async def _registry_entries(
    command: Command, exposed: set[str]
) -> dict[str, dict[str, Any]]:
    if not exposed:
        return {}
    msg = await command(
        {
            "type": "config/entity_registry/get_entries",
            "entity_ids": sorted(exposed),
        }
    )
    return {key: value for key, value in msg["result"].items() if value}


async def _device_areas(
    command: Command, entries: dict[str, dict[str, Any]]
) -> dict[str, str]:
    needs_devices = any(
        entry.get("device_id") and not entry.get("area_id")
        for entry in entries.values()
    )
    if not needs_devices:
        return {}
    msg = await command({"type": "config/device_registry/list"})
    return {
        device["id"]: device["area_id"]
        for device in msg["result"]
        if device.get("id") and device.get("area_id")
    }


def _entities(
    exposed: set[str],
    friendly: dict[str, str],
    entries: dict[str, dict[str, Any]],
    device_areas: dict[str, str],
) -> list[_Entity]:
    entities: list[_Entity] = []
    for entity_id in sorted(exposed):
        entry = entries.get(entity_id) or {}
        if entry.get("disabled_by") is not None:
            continue
        name = friendly.get(entity_id) or entry.get("name") or ""
        if not name:
            name = entry.get("original_name") or ""
        entities.append(
            _Entity(
                entity_id=entity_id,
                name=name,
                aliases=list(entry.get("aliases") or []),
                area_id=entry.get("area_id")
                or device_areas.get(entry.get("device_id") or ""),
            )
        )
    return entities


async def _name_list(command: Command, entities: list[_Entity]) -> NameList:
    areas = list((await command({"type": "config/area_registry/list"}))["result"])
    floors = list((await command({"type": "config/floor_registry/list"}))["result"])
    used_area_ids = {entity.area_id for entity in entities if entity.area_id}
    used_floor_ids = {
        area.get("floor_id")
        for area in areas
        if area.get("floor_id") and area.get("area_id") in used_area_ids
    }
    return NameList(
        used_areas=clean_names(
            _place_names(areas, "area_id", used_area_ids, in_use=True),
            _place_names(floors, "floor_id", used_floor_ids, in_use=True),
        ),
        priority_entities=clean_names(
            _entity_names(entity for entity in entities if entity.is_priority)
        ),
        empty_areas=clean_names(
            _place_names(areas, "area_id", used_area_ids, in_use=False),
            _place_names(floors, "floor_id", used_floor_ids, in_use=False),
        ),
        other_entities=clean_names(
            _entity_names(entity for entity in entities if not entity.is_priority)
        ),
    )


def _entity_names(entities: Iterable[_Entity]) -> list[str]:
    """Aliases stay with the name they belong to."""
    names: list[str] = []
    for entity in entities:
        if entity.name:
            names.append(entity.name)
        names.extend(entity.aliases)
    return names


def _place_names(
    records: Iterable[dict[str, Any]],
    id_key: str,
    used_ids: set[Any],
    in_use: bool,
) -> list[str]:
    names: list[str] = []
    for record in records:
        if (record.get(id_key) in used_ids) != in_use:
            continue
        names.append(record.get("name") or "")
        names.extend(record.get("aliases") or [])
    return names


class HassNameCache:
    """Names fetched at startup and again, in the background, once a day old."""

    def __init__(self, hass: NameSource) -> None:
        self._hass = hass
        self._names: NameList | None = None
        self._fetched_at: float | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._refresh()

    def maybe_refresh(self) -> asyncio.Task[None] | None:
        if self._task is not None and not self._task.done():
            return None
        if self._fetched_at is not None:
            if time.monotonic() - self._fetched_at < REFRESH_AFTER:
                return None
        self._task = asyncio.create_task(self._refresh())
        return self._task

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def phrases(self) -> list[str]:
        if self._names is None:
            return []
        return self._names.phrases()

    async def _refresh(self) -> None:
        try:
            names = await self._hass.get_names()
        except Exception as exc:
            _LOGGER.warning("Failed to load names from Home Assistant: %s", exc)
            return
        self._names = names
        self._fetched_at = time.monotonic()
        _LOGGER.info("Loaded %d names from Home Assistant", len(names.phrases()))
