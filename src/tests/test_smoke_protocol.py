"""Smoke tests — protocol/messages DTOs.

Verifica serialización/deserialización de todos los mensajes Unity↔Python
sin depender de red, SPADE ni Ollama.
"""

from __future__ import annotations

import pytest
from protocol.messages import (
    Pos2,
    MapBounds,
    ZoneBounds,
    ProfileZone,
    NPCProfilePayload,
    RecipePayload,
    RecipeItemPayload,
    ItemSpawnSummary,
    ActionResult,
    InventoryItem,
    InventoryUpdate,
    ZoneEntry,
    ValidationError,
    derive_entity_catalog,
)


# ---------------------------------------------------------------------------
# ZoneEntry (0.A1)
# ---------------------------------------------------------------------------

def test_zone_entry_roundtrip() -> None:
    raw = {"type": "ZoneEntry", "npc_id": "npc_001", "zone_tag": "bakeri", "entered": True}
    msg = ZoneEntry.from_dict(raw)
    assert msg.zone_tag == "bakeri"
    assert msg.entered is True
    assert msg.to_dict() == raw


def test_zone_entry_exit() -> None:
    raw = {"type": "ZoneEntry", "npc_id": "npc_001", "zone_tag": "bakeri", "entered": False}
    msg = ZoneEntry.from_dict(raw)
    assert msg.entered is False


def test_zone_entry_rejects_wrong_type() -> None:
    with pytest.raises(ValidationError):
        ZoneEntry.from_dict({"type": "Nope", "npc_id": "n", "zone_tag": "z", "entered": True})


def test_zone_entry_rejects_non_bool_entered() -> None:
    with pytest.raises(ValidationError):
        ZoneEntry.from_dict({"type": "ZoneEntry", "npc_id": "n", "zone_tag": "z", "entered": "yes"})


# ---------------------------------------------------------------------------
# Helper: minimal NPCProfile payload dict
# ---------------------------------------------------------------------------

def _profile_dict(**overrides) -> dict:
    base = {
        "role": "baker",
        "display_name": "Lars",
        "home_pos": {"x": 5, "y": 10},
        "background": "A local baker in the village.",
        "personality": "Friendly and hardworking.",
        "llm_context": "",
        "map_bounds": {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100},
        "zones": [
            {
                "tag": "bakery",
                "center": {"x": 50, "y": 50},
                "bounds": {"x_min": 40, "x_max": 60, "y_min": 40, "y_max": 60},
            }
        ],
        "delivery_points": {"tavern": {"x": 80, "y": 20}},
        "recipes": [
            {
                "recipeId": "bread",
                "zone": "bakery",
                "inputs": [{"itemId": "wheat", "qty": 2}],
                "outputs": [{"itemId": "bread", "qty": 1}],
            }
        ],
        "goals_nl": ["Deliver bread to the tavern"],
        "item_spawns": [
            {"itemId": "wheat", "zones": ["farmland"], "targetCount": 5}
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Pos2
# ---------------------------------------------------------------------------

def test_pos2_roundtrip() -> None:
    p = Pos2.from_dict({"x": 10, "y": 20})
    assert p.x == 10 and p.y == 20
    assert p.to_dict() == {"x": 10, "y": 20}


# ---------------------------------------------------------------------------
# NPCProfilePayload
# ---------------------------------------------------------------------------

def test_npc_profile_from_dict() -> None:
    profile = NPCProfilePayload.from_dict(_profile_dict())
    assert profile.role == "baker"
    assert profile.display_name == "Lars"
    assert len(profile.zones) == 1
    assert profile.zones[0].tag == "bakery"
    assert profile.goals_nl == ["Deliver bread to the tavern"]


def test_npc_profile_delivery_points() -> None:
    profile = NPCProfilePayload.from_dict(_profile_dict())
    assert "tavern" in profile.delivery_points
    tavern = profile.delivery_points["tavern"]
    assert tavern.x == 80 and tavern.y == 20


def test_npc_profile_recipes() -> None:
    profile = NPCProfilePayload.from_dict(_profile_dict())
    assert len(profile.recipes) == 1
    r = profile.recipes[0]
    assert r.recipeId == "bread"
    assert r.inputs[0].itemId == "wheat"


def test_npc_profile_item_spawns() -> None:
    profile = NPCProfilePayload.from_dict(_profile_dict())
    assert len(profile.item_spawns) == 1
    assert profile.item_spawns[0].itemId == "wheat"
    assert "farmland" in profile.item_spawns[0].zones


def test_npc_profile_roundtrip() -> None:
    """from_dict → to_dict preserves required fields."""
    data = _profile_dict()
    profile = NPCProfilePayload.from_dict(data)
    out = profile.to_dict()
    assert out["role"] == data["role"]
    assert out["goals_nl"] == data["goals_nl"]
    assert len(out["zones"]) == len(data["zones"])


def test_npc_profile_empty_lists_ok() -> None:
    """Listas opcionales vacías permitidas."""
    data = _profile_dict(zones=[], recipes=[], item_spawns=[], delivery_points={})
    profile = NPCProfilePayload.from_dict(data)
    assert profile.zones == []


def test_npc_profile_missing_role_raises() -> None:
    data = _profile_dict()
    del data["role"]
    with pytest.raises(ValidationError, match="role"):
        NPCProfilePayload.from_dict(data)


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------

def test_action_result_success() -> None:
    msg = {
        "type": "ActionResult",
        "msg_id": "msg-001",
        "commandId": "abc-123",
        "npcId": "npc_001",
        "actionType": "MoveTo",
        "status": "Success",
        "startedTick": 5,
        "endedTick": 15,
    }
    ar = ActionResult.from_dict(msg)
    assert ar.commandId == "abc-123"
    assert ar.status == "Success"


def test_action_result_failure() -> None:
    msg = {
        "type": "ActionResult",
        "msg_id": "msg-002",
        "commandId": "xyz-999",
        "npcId": "npc_001",
        "actionType": "PickUp",
        "status": "Failure",
        "startedTick": 1,
        "endedTick": 3,
        "errorCode": "ItemNotFound",
    }
    ar = ActionResult.from_dict(msg)
    assert ar.status == "Failure"
    assert ar.errorCode == "ItemNotFound"


# ---------------------------------------------------------------------------
# InventoryUpdate
# ---------------------------------------------------------------------------

def test_inventory_update_from_dict() -> None:
    msg = {
        "type": "InventoryUpdate",
        "msg_id": "msg-003",
        "npc_id": "npc_001",
        "items": [
            {"itemId": "wheat", "qty": 3},
            {"itemId": "bread", "qty": 1},
        ],
    }
    inv = InventoryUpdate.from_dict(msg)
    assert len(inv.items) == 2
    wheat = next(i for i in inv.items if i.itemId == "wheat")
    assert wheat.qty == 3


# ---------------------------------------------------------------------------
# derive_entity_catalog
# ---------------------------------------------------------------------------

class TestDeriveEntityCatalog:
    """Tests for derive_entity_catalog helper."""

    def _profile(self) -> NPCProfilePayload:
        return NPCProfilePayload.from_dict(_profile_dict())

    def test_zone_ids_from_zones(self) -> None:
        cat = derive_entity_catalog(self._profile())
        assert "bakery" in cat["zone_ids"]

    def test_item_ids_from_spawns(self) -> None:
        cat = derive_entity_catalog(self._profile())
        assert "wheat" in cat["item_ids"]

    def test_item_ids_include_recipe_outputs(self) -> None:
        cat = derive_entity_catalog(self._profile())
        # recipe output is bread
        assert "bread" in cat["item_ids"]

    def test_delivery_tags_from_delivery_points(self) -> None:
        cat = derive_entity_catalog(self._profile())
        assert "tavern" in cat["delivery_tags"]

    def test_recipe_ids_from_recipes(self) -> None:
        cat = derive_entity_catalog(self._profile())
        assert "bread" in cat["recipe_ids"]

    def test_lists_are_sorted(self) -> None:
        cat = derive_entity_catalog(self._profile())
        for key in ("zone_ids", "item_ids", "delivery_tags", "recipe_ids"):
            assert cat[key] == sorted(cat[key])

    def test_empty_profile_returns_empty_lists(self) -> None:
        raw = _profile_dict()
        raw["zones"] = []
        raw["item_spawns"] = []
        raw["recipes"] = []
        raw["delivery_points"] = {}
        profile = NPCProfilePayload.from_dict(raw)
        cat = derive_entity_catalog(profile)
        assert cat == {"zone_ids": [], "item_ids": [], "delivery_tags": [], "recipe_ids": []}

    def test_multi_zone_multi_item(self) -> None:
        raw = _profile_dict()
        raw["zones"] = [
            {"tag": "farmland", "center": {"x": 10, "y": 10},
             "bounds": {"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 20}},
            {"tag": "quarry", "center": {"x": 50, "y": 50},
             "bounds": {"x_min": 40, "x_max": 60, "y_min": 40, "y_max": 60}},
        ]
        raw["item_spawns"] = [
            {"itemId": "stone", "zones": ["quarry"], "targetCount": 3},
            {"itemId": "wheat", "zones": ["farmland"], "targetCount": 5},
        ]
        profile = NPCProfilePayload.from_dict(raw)
        cat = derive_entity_catalog(profile)
        assert cat["zone_ids"] == ["farmland", "quarry"]
        assert "stone" in cat["item_ids"]
        assert "wheat" in cat["item_ids"]


# ---------------------------------------------------------------------------
# search_range — tolerancia y ausencia
# ---------------------------------------------------------------------------

def test_npc_profile_tolerates_search_range_from_unity() -> None:
    """Unity sigue enviando search_range; Python lo ignora silenciosamente."""
    data = _profile_dict()
    data["search_range"] = 20  # Unity aun lo manda
    profile = NPCProfilePayload.from_dict(data)
    # El perfil parsea correctamente y search_range no es un atributo
    assert profile.role == "baker"
    assert not hasattr(profile, "search_range")


def test_npc_profile_parses_without_search_range() -> None:
    """NPCProfilePayload.from_dict funciona sin search_range en el payload."""
    data = _profile_dict()
    data.pop("search_range", None)
    profile = NPCProfilePayload.from_dict(data)
    assert profile.role == "baker"


def test_npc_profile_to_dict_no_search_range() -> None:
    """to_dict() no emite search_range."""
    profile = NPCProfilePayload.from_dict(_profile_dict())
    out = profile.to_dict()
    assert "search_range" not in out
