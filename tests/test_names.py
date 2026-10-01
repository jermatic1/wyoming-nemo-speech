import pytest

from wyoming_nemo_speech import names
from wyoming_nemo_speech.names import (
    HassNameCache,
    HomeAssistant,
    HomeAssistantError,
    NameList,
    load_names,
)


def _reply(result):
    return {"success": True, "result": result}


def _command(replies: dict):
    async def command(payload: dict) -> dict:
        return _reply(replies[payload["type"]])

    return command


async def test_names_keep_aliases_and_priority() -> None:
    replies = {
        "homeassistant/expose_entity/list": {
            "exposed_entities": {
                "light.lamp": {"conversation": True},
                "light.porch": {"conversation": True},
                "light.gone": {"conversation": True},
                "sensor.temp": {"conversation": True},
                "switch.unnamed": {"conversation": True},
                "fan.desk": {"conversation": True},
                "binary_sensor.hidden": {"conversation": False},
            }
        },
        "get_states": [
            {
                "entity_id": "light.lamp",
                "attributes": {"friendly_name": "Floor Lamp"},
            },
            {
                "entity_id": "light.porch",
                "attributes": {"friendly_name": "Porch Light"},
            },
            {
                "entity_id": "sensor.temp",
                "attributes": {"friendly_name": "Office Temp"},
            },
            {
                "entity_id": "fan.desk",
                "attributes": {"friendly_name": "Desk Fan"},
            },
            {"entity_id": "switch.unnamed", "attributes": {}},
            {"entity_id": "light.gone", "attributes": {"friendly_name": "Gone"}},
        ],
        "config/entity_registry/get_entries": {
            "light.lamp": {
                "aliases": ["standing light"],
                "area_id": "office",
            },
            "light.porch": {"aliases": []},
            "sensor.temp": {"area_id": "office"},
            "switch.unnamed": {"aliases": ["the switch"], "area_id": "office"},
            "fan.desk": {"device_id": "dev1"},
            "light.gone": {"disabled_by": "user", "area_id": "attic"},
        },
        "config/device_registry/list": [{"id": "dev1", "area_id": "office"}],
        "config/area_registry/list": [
            {
                "area_id": "office",
                "name": "Office",
                "aliases": ["study"],
                "floor_id": "up",
            },
            {"area_id": "attic", "name": "Attic", "aliases": []},
        ],
        "config/floor_registry/list": [
            {"floor_id": "up", "name": "Upstairs", "aliases": ["top floor"]},
            {"floor_id": "down", "name": "Basement", "aliases": []},
        ],
    }
    names = await load_names(_command(replies))
    assert names.phrases() == [
        "Office",
        "study",
        "Upstairs",
        "top floor",
        "Desk Fan",
        "Floor Lamp",
        "standing light",
        "Porch Light",
        "the switch",
        "Attic",
        "Basement",
        "Office Temp",
    ]
    assert names.priority_entities[1:3] == ["Floor Lamp", "standing light"]


class FakeHass:
    def __init__(self, names: NameList) -> None:
        self.names = names
        self.calls = 0
        self.error: Exception | None = None

    async def get_names(self) -> NameList:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.names


def test_websocket_url() -> None:
    hass = HomeAssistant("token", "https://ha.example/")
    assert hass.websocket_api_url == "wss://ha.example/api/websocket"


async def test_start_loads_names_once() -> None:
    hass = FakeHass(NameList(used_areas=["Office"]))
    cache = HassNameCache(hass)
    await cache.start()
    assert cache.phrases() == ["Office"]
    assert cache.maybe_refresh() is None
    assert hass.calls == 1


async def test_failed_start_retries_on_next_utterance() -> None:
    hass = FakeHass(NameList(used_areas=["Office"]))
    hass.error = HomeAssistantError("down")
    cache = HassNameCache(hass)
    await cache.start()
    assert cache.phrases() == []
    hass.error = None
    task = cache.maybe_refresh()
    assert task is not None
    await task
    assert cache.phrases() == ["Office"]


async def test_stale_names_refresh_in_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = FakeHass(NameList(used_areas=["Office"]))
    cache = HassNameCache(hass)
    await cache.start()
    hass.names = NameList(used_areas=["Kitchen"])
    monkeypatch.setattr(names, "REFRESH_AFTER", 0)
    task = cache.maybe_refresh()
    assert task is not None
    assert cache.phrases() == ["Office"]
    await task
    assert cache.phrases() == ["Kitchen"]
    await cache.close()
