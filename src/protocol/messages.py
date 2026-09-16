from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ValidationError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object.")
    return value


def _require_field(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValidationError(f"Missing required field '{key}'.")
    return data[key]


def _require_str(data: dict[str, Any], key: str, allow_empty: bool = False) -> str:
    value = _require_field(data, key)
    if not isinstance(value, str):
        raise ValidationError(f"Field '{key}' must be a string.")
    if not allow_empty and value.strip() == "":
        raise ValidationError(f"Field '{key}' must not be empty.")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"Field '{key}' must be a string when present.")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Field '{key}' must be an integer when present.")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = _require_field(data, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"Field '{key}' must be an integer.")
    return value


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = _require_field(data, key)
    if not isinstance(value, list):
        raise ValidationError(f"Field '{key}' must be a list.")
    return value


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = _require_field(data, key)
    if not isinstance(value, dict):
        raise ValidationError(f"Field '{key}' must be an object.")
    return value


def _optional_dict(data: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValidationError(f"Field '{key}' must be an object when present.")
    return value


def _optional_list(data: dict[str, Any], key: str) -> list[Any] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValidationError(f"Field '{key}' must be a list when present.")
    return value


def _expect_type(data: dict[str, Any], expected_type: str) -> None:
    msg_type = _require_str(data, "type")
    if msg_type != expected_type:
        raise ValidationError(f"Expected message type '{expected_type}', got '{msg_type}'.")


# ---------------------------------------------------------------------------
# Primitive structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Pos2:
    x: int
    y: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pos2":
        payload = _require_mapping(data, "pos")
        return cls(x=_require_int(payload, "x"), y=_require_int(payload, "y"))

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y}


@dataclass(slots=True)
class MapBounds:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapBounds":
        payload = _require_mapping(data, "map_bounds")
        return cls(
            x_min=_require_int(payload, "x_min"),
            x_max=_require_int(payload, "x_max"),
            y_min=_require_int(payload, "y_min"),
            y_max=_require_int(payload, "y_max"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"x_min": self.x_min, "x_max": self.x_max,
                "y_min": self.y_min, "y_max": self.y_max}


@dataclass(slots=True)
class ZoneBounds:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneBounds":
        payload = _require_mapping(data, "bounds")
        return cls(
            x_min=_require_int(payload, "x_min"),
            x_max=_require_int(payload, "x_max"),
            y_min=_require_int(payload, "y_min"),
            y_max=_require_int(payload, "y_max"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"x_min": self.x_min, "x_max": self.x_max,
                "y_min": self.y_min, "y_max": self.y_max}


@dataclass(slots=True)
class ProfileZone:
    tag: str
    center: Pos2
    bounds: ZoneBounds

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileZone":
        payload = _require_mapping(data, "profile.zone")

        def _safe_int(value: Any, default: int = 0) -> int:
            if isinstance(value, bool):
                return default
            if isinstance(value, int):
                return value
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        bounds_raw = payload.get("bounds")
        if isinstance(bounds_raw, dict):
            bounds = ZoneBounds.from_dict(bounds_raw)
        else:
            x1, x2 = _safe_int(payload.get("x1", 0)), _safe_int(payload.get("x2", 0))
            y1, y2 = _safe_int(payload.get("y1", 0)), _safe_int(payload.get("y2", 0))
            bounds = ZoneBounds(
                x_min=min(x1, x2), x_max=max(x1, x2),
                y_min=min(y1, y2), y_max=max(y1, y2),
            )

        center_raw = payload.get("center")
        if isinstance(center_raw, dict):
            center = Pos2.from_dict(center_raw)
        else:
            center = Pos2(
                x=(bounds.x_min + bounds.x_max) // 2,
                y=(bounds.y_min + bounds.y_max) // 2,
            )

        return cls(tag=_require_str(payload, "tag"), center=center, bounds=bounds)

    def to_dict(self) -> dict[str, Any]:
        return {"tag": self.tag, "center": self.center.to_dict(), "bounds": self.bounds.to_dict()}


@dataclass(slots=True)
class ZoneDiscoveryInfo:
    tag: str
    center: Pos2
    bounds: ZoneBounds
    observed_cells: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneDiscoveryInfo":
        payload = _require_mapping(data, "zone_discovery.zone")

        def _safe_int(value: Any, default: int = 0) -> int:
            if isinstance(value, bool):
                return default
            if isinstance(value, int):
                return value
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        bounds_raw = payload.get("bounds")
        if isinstance(bounds_raw, dict):
            bounds = ZoneBounds.from_dict(bounds_raw)
        else:
            x1, x2 = _safe_int(payload.get("x1", 0)), _safe_int(payload.get("x2", 0))
            y1, y2 = _safe_int(payload.get("y1", 0)), _safe_int(payload.get("y2", 0))
            bounds = ZoneBounds(
                x_min=min(x1, x2), x_max=max(x1, x2),
                y_min=min(y1, y2), y_max=max(y1, y2),
            )

        center_raw = payload.get("center")
        if isinstance(center_raw, dict):
            center = Pos2.from_dict(center_raw)
        else:
            center = Pos2(
                x=(bounds.x_min + bounds.x_max) // 2,
                y=(bounds.y_min + bounds.y_max) // 2,
            )

        obs = payload.get("observed_cells", 0)
        observed_cells = obs if isinstance(obs, int) and not isinstance(obs, bool) else 0

        return cls(tag=_require_str(payload, "tag"), center=center,
                   bounds=bounds, observed_cells=observed_cells)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "center": self.center.to_dict(),
            "bounds": self.bounds.to_dict(),
            "observed_cells": self.observed_cells,
        }


# ---------------------------------------------------------------------------
# Profile structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RecipeItemPayload:
    itemId: str
    qty: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecipeItemPayload":
        payload = _require_mapping(data, "recipe.item")
        return cls(itemId=_require_str(payload, "itemId"), qty=_require_int(payload, "qty"))

    def to_dict(self) -> dict[str, Any]:
        return {"itemId": self.itemId, "qty": self.qty}


@dataclass(slots=True)
class RecipePayload:
    recipeId: str
    zone: str
    inputs: list[RecipeItemPayload]
    outputs: list[RecipeItemPayload]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecipePayload":
        payload = _require_mapping(data, "recipe")
        return cls(
            recipeId=_require_str(payload, "recipeId"),
            zone=_require_str(payload, "zone"),
            inputs=[RecipeItemPayload.from_dict(_require_mapping(i, "recipe.inputs[]"))
                    for i in _require_list(payload, "inputs")],
            outputs=[RecipeItemPayload.from_dict(_require_mapping(o, "recipe.outputs[]"))
                     for o in _require_list(payload, "outputs")],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipeId": self.recipeId,
            "zone": self.zone,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
        }


@dataclass(slots=True)
class ItemSpawnSummary:
    itemId: str
    zones: list[str]
    targetCount: int
    weight: int = 1  # 0 = item does not exist in this world; default 1 for backward compat

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ItemSpawnSummary":
        payload = _require_mapping(data, "item_spawn")
        zones_raw = _require_list(payload, "zones")
        zones: list[str] = []
        for idx, zone in enumerate(zones_raw):
            if not isinstance(zone, str) or zone.strip() == "":
                raise ValidationError(f"item_spawns[].zones[{idx}] must be a non-empty string.")
            zones.append(zone)
        return cls(
            itemId=_require_str(payload, "itemId"),
            zones=zones,
            targetCount=_require_int(payload, "targetCount"),
            weight=int(payload.get("weight", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"itemId": self.itemId, "zones": self.zones, "targetCount": self.targetCount, "weight": self.weight}


@dataclass(slots=True)
class NPCProfilePayload:
    role: str
    display_name: str
    home_pos: Pos2
    background: str
    personality: str
    llm_context: str
    map_bounds: MapBounds
    zones: list[ProfileZone]
    delivery_points: dict[str, Pos2]
    recipes: list[RecipePayload]
    goals_nl: list[str]
    item_spawns: list[ItemSpawnSummary]
    # Condiciones de éxito esperadas por el diseñador, en paralelo con goals_nl.
    # Formato ASL: "has_item(wheat, 1)", "at_location(bakeri)", etc.
    # Opcional: si no se define, se usa solo la condición derivada por el LLM.
    goal_conditions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NPCProfilePayload":
        payload = _require_mapping(data, "profile")

        zones_raw = payload.get("zones", [])
        if not isinstance(zones_raw, list):
            zones_raw = []

        recipes_raw = payload.get("recipes", [])
        if not isinstance(recipes_raw, list):
            recipes_raw = []

        goals_nl_raw = payload.get("goals_nl", [])
        if not isinstance(goals_nl_raw, list):
            goals_nl_raw = []

        item_spawns_raw = payload.get("item_spawns", [])
        if not isinstance(item_spawns_raw, list):
            item_spawns_raw = []

        delivery_points_raw = payload.get("delivery_points", {})
        if not isinstance(delivery_points_raw, dict):
            delivery_points_raw = {}

        zones = [ProfileZone.from_dict(_require_mapping(z, "profile.zones[]")) for z in zones_raw]

        delivery_points: dict[str, Pos2] = {}
        for key, value in delivery_points_raw.items():
            if not isinstance(key, str) or key.strip() == "":
                raise ValidationError("delivery_points keys must be non-empty strings.")
            delivery_points[key] = Pos2.from_dict(_require_mapping(value, f"delivery_points['{key}']"))

        recipes = [RecipePayload.from_dict(_require_mapping(r, "profile.recipes[]")) for r in recipes_raw]

        goals_nl: list[str] = []
        for idx, goal in enumerate(goals_nl_raw):
            if not isinstance(goal, str) or goal.strip() == "":
                raise ValidationError(f"goals_nl[{idx}] must be a non-empty string.")
            goals_nl.append(goal)

        item_spawns = [
            ItemSpawnSummary.from_dict(_require_mapping(s, "profile.item_spawns[]"))
            for s in item_spawns_raw
        ]

        goal_conditions_raw = payload.get("goal_conditions", [])
        if not isinstance(goal_conditions_raw, list):
            goal_conditions_raw = []
        goal_conditions: list[str] = []
        for idx, cond in enumerate(goal_conditions_raw):
            if isinstance(cond, str) and cond.strip():
                goal_conditions.append(cond.strip())

        return cls(
            role=_require_str(payload, "role"),
            display_name=_require_str(payload, "display_name"),
            home_pos=Pos2.from_dict(_require_dict(payload, "home_pos")),
            background=_require_str(payload, "background", allow_empty=True),
            personality=_require_str(payload, "personality", allow_empty=True),
            llm_context=_require_str(payload, "llm_context", allow_empty=True),
            map_bounds=MapBounds.from_dict(_require_dict(payload, "map_bounds")),
            zones=zones,
            delivery_points=delivery_points,
            recipes=recipes,
            goals_nl=goals_nl,
            item_spawns=item_spawns,
            goal_conditions=goal_conditions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "display_name": self.display_name,
            "home_pos": self.home_pos.to_dict(),
            "background": self.background,
            "personality": self.personality,
            "llm_context": self.llm_context,
            "map_bounds": self.map_bounds.to_dict(),
            "zones": [z.to_dict() for z in self.zones],
            "delivery_points": {k: v.to_dict() for k, v in self.delivery_points.items()},
            "recipes": [r.to_dict() for r in self.recipes],
            "goals_nl": self.goals_nl,
            "item_spawns": [s.to_dict() for s in self.item_spawns],
            "goal_conditions": self.goal_conditions,
        }


# ---------------------------------------------------------------------------
# Unity → Python messages
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegisterNPC:
    type: str
    msg_id: str
    npc_id: str
    pos: Pos2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegisterNPC":
        payload = _require_mapping(data, "RegisterNPC")
        _expect_type(payload, "RegisterNPC")
        return cls(
            type="RegisterNPC",
            msg_id=_require_str(payload, "msg_id"),
            npc_id=_require_str(payload, "npc_id"),
            pos=Pos2.from_dict(_require_dict(payload, "pos")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "msg_id": self.msg_id,
                "npc_id": self.npc_id, "pos": self.pos.to_dict()}


@dataclass(slots=True)
class NPCProfile:
    type: str
    msg_id: str
    npc_id: str
    profile: NPCProfilePayload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NPCProfile":
        payload = _require_mapping(data, "NPCProfile")
        _expect_type(payload, "NPCProfile")
        return cls(
            type="NPCProfile",
            msg_id=_require_str(payload, "msg_id"),
            npc_id=_require_str(payload, "npc_id"),
            profile=NPCProfilePayload.from_dict(_require_dict(payload, "profile")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "msg_id": self.msg_id,
                "npc_id": self.npc_id, "profile": self.profile.to_dict()}


_ACTION_RESULT_STATUSES = {"Running", "Success", "Failure"}


@dataclass(slots=True)
class ActionResult:
    type: str
    msg_id: str
    commandId: str
    npcId: str
    actionType: str
    status: str
    startedTick: int
    endedTick: int
    errorCode: str | None = None
    errorSeverity: str | None = None
    errorMessage: str | None = None
    payload: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResult":
        payload = _require_mapping(data, "ActionResult")
        _expect_type(payload, "ActionResult")
        status = _require_str(payload, "status")
        if status not in _ACTION_RESULT_STATUSES:
            raise ValidationError(
                f"ActionResult.status must be one of {sorted(_ACTION_RESULT_STATUSES)}, got '{status}'."
            )
        return cls(
            type="ActionResult",
            msg_id=_require_str(payload, "msg_id"),
            commandId=_require_str(payload, "commandId"),
            npcId=_require_str(payload, "npcId"),
            actionType=_require_str(payload, "actionType"),
            status=status,
            startedTick=_require_int(payload, "startedTick"),
            endedTick=_require_int(payload, "endedTick"),
            errorCode=_optional_str(payload, "errorCode"),
            errorSeverity=_optional_str(payload, "errorSeverity"),
            errorMessage=_optional_str(payload, "errorMessage"),
            payload=_optional_dict(payload, "payload"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "msg_id": self.msg_id,
            "commandId": self.commandId,
            "npcId": self.npcId,
            "actionType": self.actionType,
            "status": self.status,
            "startedTick": self.startedTick,
            "endedTick": self.endedTick,
        }
        if self.errorCode is not None:
            data["errorCode"] = self.errorCode
        if self.errorSeverity is not None:
            data["errorSeverity"] = self.errorSeverity
        if self.errorMessage is not None:
            data["errorMessage"] = self.errorMessage
        if self.payload is not None:
            data["payload"] = self.payload
        return data


@dataclass(slots=True)
class ZoneDiscovery:
    type: str
    msg_id: str
    npc_id: str
    zones: list[ZoneDiscoveryInfo]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneDiscovery":
        payload = _require_mapping(data, "ZoneDiscovery")
        _expect_type(payload, "ZoneDiscovery")
        zones_raw = _require_list(payload, "zones")
        return cls(
            type="ZoneDiscovery",
            msg_id=_require_str(payload, "msg_id"),
            npc_id=_require_str(payload, "npc_id"),
            zones=[ZoneDiscoveryInfo.from_dict(_require_mapping(z, "ZoneDiscovery.zones[]"))
                   for z in zones_raw],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "msg_id": self.msg_id,
                "npc_id": self.npc_id, "zones": [z.to_dict() for z in self.zones]}


@dataclass(slots=True)
class InventoryItem:
    itemId: str
    qty: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InventoryItem":
        payload = _require_mapping(data, "InventoryItem")
        return cls(itemId=_require_str(payload, "itemId"), qty=_require_int(payload, "qty"))

    def to_dict(self) -> dict[str, Any]:
        return {"itemId": self.itemId, "qty": self.qty}


@dataclass(slots=True)
class InventoryUpdate:
    type: str
    msg_id: str
    npc_id: str
    items: list[InventoryItem]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InventoryUpdate":
        payload = _require_mapping(data, "InventoryUpdate")
        _expect_type(payload, "InventoryUpdate")
        items_raw = _require_list(payload, "items")
        return cls(
            type="InventoryUpdate",
            msg_id=_require_str(payload, "msg_id"),
            npc_id=_require_str(payload, "npc_id"),
            items=[InventoryItem.from_dict(_require_mapping(i, "InventoryUpdate.items[]"))
                   for i in items_raw],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "msg_id": self.msg_id,
                "npc_id": self.npc_id, "items": [i.to_dict() for i in self.items]}


@dataclass(slots=True)
class WanderRetry:
    type: str
    commandId: str
    npcId: str
    attempt: int
    maxAttempts: int
    reason: str
    x: int
    y: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WanderRetry":
        payload = _require_mapping(data, "WanderRetry")
        _expect_type(payload, "WanderRetry")
        return cls(
            type="WanderRetry",
            commandId=_require_str(payload, "commandId"),
            npcId=_require_str(payload, "npcId"),
            attempt=_require_int(payload, "attempt"),
            maxAttempts=_require_int(payload, "maxAttempts"),
            reason=_require_str(payload, "reason", allow_empty=True),
            x=_require_int(payload, "x"),
            y=_require_int(payload, "y"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "commandId": self.commandId,
            "npcId": self.npcId,
            "attempt": self.attempt,
            "maxAttempts": self.maxAttempts,
            "reason": self.reason,
            "x": self.x,
            "y": self.y,
        }


@dataclass(slots=True)
class ZoneEntry:
    """El NPC ha entrado o salido de una zona con trigger (emitido por Unity).

    `entered=True`  → assertar at_zone(zone_tag).
    `entered=False` → retractar at_zone(zone_tag) (equivale a un ZoneExit).
    """
    type: str
    npc_id: str
    zone_tag: str
    entered: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneEntry":
        payload = _require_mapping(data, "ZoneEntry")
        _expect_type(payload, "ZoneEntry")
        entered = payload.get("entered", True)
        if not isinstance(entered, bool):
            raise ValidationError("Field 'entered' must be a boolean.")
        return cls(
            type="ZoneEntry",
            npc_id=_require_str(payload, "npc_id"),
            zone_tag=_require_str(payload, "zone_tag"),
            entered=entered,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "npc_id": self.npc_id,
                "zone_tag": self.zone_tag, "entered": self.entered}


@dataclass(slots=True)
class Ping:
    type: str
    msg_id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ping":
        payload = _require_mapping(data, "Ping")
        _expect_type(payload, "Ping")
        return cls(type="Ping", msg_id=_require_str(payload, "msg_id"))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "msg_id": self.msg_id}


# ---------------------------------------------------------------------------
# Python → Unity messages
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ActionCommand:
    type: str
    msg_id: str
    commandId: str
    npcId: str
    actionType: str
    issuedTicks: int
    timeoutTicks: int
    args: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionCommand":
        payload = _require_mapping(data, "ActionCommand")
        _expect_type(payload, "ActionCommand")
        return cls(
            type="ActionCommand",
            msg_id=_require_str(payload, "msg_id"),
            commandId=_require_str(payload, "commandId"),
            npcId=_require_str(payload, "npcId"),
            actionType=_require_str(payload, "actionType"),
            issuedTicks=_require_int(payload, "issuedTicks"),
            timeoutTicks=_require_int(payload, "timeoutTicks"),
            args=_require_dict(payload, "args"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "msg_id": self.msg_id,
            "commandId": self.commandId,
            "npcId": self.npcId,
            "actionType": self.actionType,
            "issuedTicks": self.issuedTicks,
            "timeoutTicks": self.timeoutTicks,
            "args": self.args,
        }


@dataclass(slots=True)
class NPCRegistered:
    type: str
    npc_id: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "npc_id": self.npc_id, "status": self.status}


@dataclass(slots=True)
class NPCReady:
    type: str
    npc_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "npc_id": self.npc_id}


@dataclass(slots=True)
class NPCProfileAck:
    type: str
    msg_id: str
    npc_id: str
    status: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "msg_id": self.msg_id,
            "npc_id": self.npc_id,
            "status": self.status,
        }
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(slots=True)
class Pong:
    type: str
    msg_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "msg_id": self.msg_id}


@dataclass(slots=True)
class PythonDisconnecting:
    type: str = "PythonDisconnecting"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type}


# ---------------------------------------------------------------------------
# Derived catalog helper
# ---------------------------------------------------------------------------

def derive_entity_catalog(profile: "NPCProfilePayload") -> dict[str, list[str]]:
    """Derives the entity vocabulary from an NPCProfilePayload.

    Returns a dict with four lists of valid string identifiers grouped by
    semantic role.  All values are **sorted** for deterministic prompts:

      zone_ids       — zone tag constants (e.g. farmland, bakeri)
      item_ids       — item type constants (e.g. wheat, bread)
      delivery_tags  — delivery point keys (e.g. bakeri_delivery)
      recipe_ids     — recipe identifiers  (e.g. bread_recipe)

    The catalog is injected into LLM prompts at Step 2 and Step 3 so the
    model can only generate entity names that actually exist in the current
    game world.  It is derived purely from fields already present in the
    NPCProfile — no separate Unity-side changes are required.
    """
    zone_ids: list[str] = sorted({z.tag for z in profile.zones})

    # Only include items that actually exist in the world (weight > 0) or that
    # appear as recipe inputs/outputs (they must be reachable via crafting).
    recipe_item_ids: set[str] = set()
    for recipe in profile.recipes:
        for item in recipe.inputs:
            recipe_item_ids.add(item.itemId)
        for item in recipe.outputs:
            recipe_item_ids.add(item.itemId)

    item_ids_set: set[str] = {
        s.itemId for s in profile.item_spawns
        if s.weight > 0 or s.itemId in recipe_item_ids
    } | recipe_item_ids  # always include recipe items even if not in spawns
    item_ids: list[str] = sorted(item_ids_set)

    delivery_tags: list[str] = sorted(profile.delivery_points.keys())
    recipe_ids: list[str] = sorted({r.recipeId for r in profile.recipes})

    return {
        "zone_ids": zone_ids,
        "item_ids": item_ids,
        "delivery_tags": delivery_tags,
        "recipe_ids": recipe_ids,
    }
